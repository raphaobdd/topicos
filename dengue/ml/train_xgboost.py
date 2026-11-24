"""Treina XGBoost com seleção no treino, validação de limiar e teste independente."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
import xgboost
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from .features import ALVO_DESCRICAO, FEATURES, alvo_sinan, preparar_sinan
from .train_tree import carregar, carregar_amostra_aleatoria


def treinar(df, saida: Path, *, auc_minimo=0.65, semente=42, origem=None):
    elegiveis, y = alvo_sinan(df)
    X = preparar_sinan(df).loc[elegiveis]
    y = y.loc[elegiveis]
    if y.nunique() != 2 or y.sum() < 50:
        raise ValueError("Poucos positivos elegíveis para treino")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=semente)
    Xfit, Xval, yfit, yval = train_test_split(Xtr, ytr, test_size=.25, stratify=ytr, random_state=semente)
    peso = float((yfit == 0).sum() / (yfit == 1).sum())
    busca = GridSearchCV(
        XGBClassifier(n_estimators=200, objective="binary:logistic", eval_metric="logloss",
                      tree_method="hist", n_jobs=2, random_state=semente, scale_pos_weight=peso),
        {"max_depth": [3, 5], "learning_rate": [.05, .1], "min_child_weight": [5, 20]},
        scoring="average_precision", cv=StratifiedKFold(3, shuffle=True, random_state=semente), n_jobs=2,
    )
    print("Buscando hiperparâmetros XGBoost...", flush=True)
    busca.fit(Xfit, yfit)
    pv = busca.predict_proba(Xval)[:, 1]
    precisao, recall, limites = precision_recall_curve(yval, pv)
    f1 = 2 * precisao[:-1] * recall[:-1] / np.maximum(precisao[:-1] + recall[:-1], 1e-12)
    limiar = float(limites[int(np.argmax(f1))])
    modelo = busca.best_estimator_
    modelo.set_params(scale_pos_weight=float((ytr == 0).sum() / (ytr == 1).sum()))
    modelo.fit(Xtr, ytr)
    p = modelo.predict_proba(Xte)[:, 1]
    auc = float(roc_auc_score(yte, p))
    ap = float(average_precision_score(yte, p))
    matriz = confusion_matrix(yte, p >= limiar, labels=[0, 1]).tolist()
    pacote = {"modelo": modelo, "tipo_modelo": "XGBoost", "features": FEATURES,
              "alvo": ALVO_DESCRICAO, "prevalencia": float(ytr.mean()), "limiar": limiar,
              "metricas": {"auc_teste": auc, "average_precision": ap, "matriz_confusao": matriz,
                          "cv_average_precision": float(busca.best_score_), "limiar_validacao": limiar},
              "n_treino": len(Xtr), "n_teste": len(Xte), "positivos_teste": int(yte.sum()),
              "parametros": busca.best_params_, "viavel": auc >= auc_minimo, "sintetico": False,
              "sklearn": sklearn.__version__, "xgboost": xgboost.__version__, "origem": origem or {}}
    saida.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pacote, saida)
    saida.with_suffix(".metricas.json").write_text(
        json.dumps({k: v for k, v in pacote.items() if k != "modelo"}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"XGBoost: AUC={auc:.4f}, AP={ap:.4f}, matriz={matriz}, viavel={pacote['viavel']}", flush=True)
    return pacote


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--amostra-aleatoria", type=int)
    ap.add_argument("--sep", default=",")
    ap.add_argument("--auc-minimo", type=float, default=.65)
    ap.add_argument("--saida", default="models/xgboost_dengue.joblib")
    args = ap.parse_args()
    df = (carregar_amostra_aleatoria(args.csv, args.amostra_aleatoria, args.sep)
          if args.amostra_aleatoria else carregar(args.csv, args.sep))
    treinar(df, Path(args.saida), auc_minimo=args.auc_minimo,
            origem={"arquivo": args.csv, "amostra_aleatoria": args.amostra_aleatoria, "semente": 42})


if __name__ == "__main__":
    main()
