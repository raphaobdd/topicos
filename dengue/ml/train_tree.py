"""Análise da base DENGBR24 + treino da árvore de decisão.

Uso:
    python -m dengue.ml.train_tree --csv data/DENGBR24.csv.zip --explorar
    python -m dengue.ml.train_tree --csv data/DENGBR24.csv.zip --max-linhas 500000
    python -m dengue.ml.train_tree --sintetico          # teste do pipeline (não usar na aplicação)

Etapas (numeração da especificação):
  1-5  analisar a base: variáveis, tipos, ausentes, duplicados, distribuição das classes
  6    alvo: CLASSI_FIN 11/12 ("alarme/grave") vs 10 ("dengue") — NÃO é o grupo A/B/C/D
  7    só variáveis disponíveis na triagem (sem ALRM_*/GRAV_*/HOSPITALIZ/EVOLUCAO...)
  8-9  preparo + separação treino/teste estratificada
  10   árvore de decisão com busca de hiperparâmetros (validação cruzada)
  11   AUC, average precision vs. baseline, matriz de confusão no teste
  12   marca se o modelo é "viável" para integração (AUC mínimo); o app só usa se for.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

from .. import config
from .features import ALVO_DESCRICAO, COLUNAS_SINAN, FEATURES, alvo_sinan, preparar_sinan


def carregar(csv: str, sep: str = ",", max_linhas: int | None = None) -> pd.DataFrame:
    usar = set(COLUNAS_SINAN) | {"CLASSI_FIN"}
    return pd.read_csv(
        csv,
        sep=sep,
        usecols=lambda c: c in usar,
        low_memory=False,
        nrows=max_linhas,
        encoding_errors="replace",
    )


def carregar_amostra_aleatoria(csv: str, tamanho: int, sep: str = ",", semente: int = 42) -> pd.DataFrame:
    """Amostra uniforme e reproduzivel sem carregar todo o CSV em memoria."""
    if tamanho < 1:
        raise ValueError("tamanho da amostra deve ser positivo")
    usar = set(COLUNAS_SINAN) | {"CLASSI_FIN"}
    rng = np.random.default_rng(semente)
    amostra = pd.DataFrame()
    for bloco in pd.read_csv(csv, sep=sep, usecols=lambda c: c in usar,
                             low_memory=False, chunksize=100000, encoding_errors="replace"):
        bloco = bloco.copy()
        bloco["_prioridade"] = rng.random(len(bloco))
        amostra = pd.concat([amostra, bloco], ignore_index=True)
        if len(amostra) > tamanho:
            amostra = amostra.nsmallest(tamanho, "_prioridade")
    return amostra.drop(columns="_prioridade").reset_index(drop=True)


def explorar(df: pd.DataFrame) -> None:
    print("\n=== 1-3. Estrutura ===")
    print(f"Linhas: {len(df):,} | Colunas lidas: {df.shape[1]}")
    faltando = [c for c in COLUNAS_SINAN if c not in df.columns]
    if faltando:
        print(f"!! Colunas esperadas que NÃO existem na base: {faltando}")
    print(df.dtypes.to_string())
    print("\n% de ausentes por coluna:")
    print((df.isna().mean() * 100).round(1).sort_values(ascending=False).to_string())

    print("\n=== 4. Duplicados ===")
    chaves = [c for c in ["NU_NOTIFIC", "DT_NOTIFIC", "NU_IDADE_N", "CS_SEXO", "ID_MUNICIP", "SG_UF_NOT"] if c in df.columns]
    print(f"Registros duplicados em {chaves}: {df.duplicated(subset=chaves).sum():,}")

    print("\n=== 5. Distribuição de CLASSI_FIN (5=descartado, 8=inconclusivo, 10=dengue, 11=alarme, 12=grave) ===")
    print(df["CLASSI_FIN"].value_counts(dropna=False).sort_index().to_string())


def treinar(df: pd.DataFrame, saida: Path, auc_minimo: float, sintetico: bool, semente: int = 42,
            origem: dict | None = None) -> dict:
    elegiveis, y_all = alvo_sinan(df)
    X = preparar_sinan(df)[elegiveis]
    y = y_all[elegiveis]
    # remove duplicados exatos de notificação, se houver chave
    if "NU_NOTIFIC" in df.columns:
        dup = df.loc[elegiveis].duplicated(subset=[c for c in ["NU_NOTIFIC", "DT_NOTIFIC", "ID_MUNICIP", "SG_UF_NOT"] if c in df.columns])
        X, y = X[~dup.values], y[~dup.values]

    print("\n=== 6-7. Alvo e variáveis ===")
    print(f"Alvo: {ALVO_DESCRICAO}")
    print(f"Variáveis ({len(FEATURES)}): {', '.join(FEATURES)}")
    print(f"Registros elegíveis: {len(X):,} | prevalência do alvo: {y.mean():.3%}")
    if y.nunique() < 2 or y.sum() < 50:
        raise SystemExit("Poucos casos positivos para treinar. Verifique os códigos de CLASSI_FIN.")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=semente)
    print(f"Treino: {len(Xtr):,} | Teste: {len(Xte):,}")

    print("\n=== 10. Treino (busca de hiperparâmetros, CV=3, métrica AUC) ===")
    folhas = [max(20, int(0.001 * len(Xtr))), max(50, int(0.005 * len(Xtr)))]
    busca = GridSearchCV(
        DecisionTreeClassifier(random_state=semente),
        {"max_depth": [3, 4, 5, 6], "min_samples_leaf": folhas},
        scoring="roc_auc",
        cv=3,
    )
    busca.fit(Xtr, ytr)
    arvore = busca.best_estimator_
    print(f"Melhores parâmetros: {busca.best_params_} | AUC (CV): {busca.best_score_:.3f}")

    print("\n=== 11. Avaliação no teste ===")
    p = arvore.predict_proba(Xte)[:, 1]
    prevalencia = float(ytr.mean())
    auc = float(roc_auc_score(yte, p))
    ap = float(average_precision_score(yte, p))
    pred = (p > prevalencia).astype(int)  # limiar = prevalência (probabilidades não são "balanceadas")
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    recall, precisao = tp / max(tp + fn, 1), tp / max(tp + fp, 1)
    print(f"AUC: {auc:.3f} | Average precision: {ap:.3f} (baseline = prevalência {prevalencia:.3f})")
    print(f"Limiar = prevalência -> recall {recall:.2%}, precisão {precisao:.2%}")
    print(f"Matriz de confusão [[TN FP] [FN TP]]: [[{tn} {fp}] [{fn} {tp}]]")
    print("\nRegras aprendidas (até profundidade 4):")
    print(export_text(arvore, feature_names=FEATURES, max_depth=4))

    viavel = (auc >= auc_minimo) and not sintetico
    print("=== 12. Viabilidade ===")
    print(f"AUC mínimo exigido: {auc_minimo} -> {'VIÁVEL' if viavel else 'NÃO usar na aplicação'}")

    pacote = {
        "modelo": arvore,
        "features": FEATURES,
        "alvo": ALVO_DESCRICAO,
        "prevalencia": prevalencia,
        "metricas": {"auc_teste": auc, "average_precision": ap, "recall_limiar": recall, "precisao_limiar": precisao},
        "n_treino": int(len(Xtr)),
        "viavel": bool(viavel),
        "sintetico": bool(sintetico),
        "sklearn": sklearn.__version__,
        "origem": origem or {},
    }
    saida.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pacote, saida)
    (saida.with_suffix(".metricas.json")).write_text(
        json.dumps({k: v for k, v in pacote.items() if k not in ("modelo",)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nModelo salvo em {saida}")
    return pacote


def main() -> None:
    ap = argparse.ArgumentParser(description="EDA e treino da árvore de decisão (DENGBR24)")
    ap.add_argument("--csv", help="caminho do DENGBR24.csv ou .csv.zip")
    ap.add_argument("--sep", default=",")
    ap.add_argument("--max-linhas", type=int, default=None, help="lê só as N primeiras linhas (testes rápidos)")
    ap.add_argument("--explorar", action="store_true", help="só análise exploratória, sem treinar")
    ap.add_argument("--amostra-aleatoria", type=int, default=None, help="amostra uniforme N linhas de todo o CSV")
    ap.add_argument("--auc-minimo", type=float, default=0.65)
    ap.add_argument("--saida", default=str(config.MODELO_ARVORE))
    ap.add_argument("--sintetico", action="store_true", help="usa base sintética (teste do pipeline)")
    args = ap.parse_args()

    if args.sintetico:
        from .sintetico import gerar

        df = gerar(8000)
        saida = Path(args.saida).with_name("arvore_sintetica.joblib")
    else:
        if not args.csv:
            raise SystemExit("Informe --csv data/DENGBR24.csv.zip (ou use --sintetico).")
        df = (carregar_amostra_aleatoria(args.csv, args.amostra_aleatoria, args.sep)
              if args.amostra_aleatoria else carregar(args.csv, args.sep, args.max_linhas))
        saida = Path(args.saida)

    explorar(df)
    if not args.explorar:
        origem = {"arquivo": args.csv or "sintetico", "amostra_aleatoria": args.amostra_aleatoria,
                  "max_linhas": args.max_linhas, "semente": 42}
        if args.csv:
            from .import_sinan import sha256_arquivo
            origem["sha256"] = sha256_arquivo(Path(args.csv))
        treinar(df, saida, args.auc_minimo, sintetico=args.sintetico, origem=origem)


if __name__ == "__main__":
    main()
