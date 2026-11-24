"""Comparação experimental em um teste intocado; execução: python -m dengue.ml.compare --csv ARQUIVO."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

from .features import FEATURES, alvo_sinan, preparar_sinan
from .train_tree import carregar, carregar_amostra_aleatoria


def _curva(x, y, max_pontos=500):
    indices = np.unique(np.linspace(0, len(x) - 1, min(max_pontos, len(x)), dtype=int))
    return x[indices].tolist(), y[indices].tolist()


def comparar(df, seed=42):
    elegiveis, alvo = alvo_sinan(df)
    X, y = preparar_sinan(df).loc[elegiveis], alvo.loc[elegiveis]
    if y.nunique() != 2 or y.sum() < 20:
        raise ValueError("São necessários pelo menos 20 casos positivos elegíveis.")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=seed)
    Xfit, Xval, yfit, yval = train_test_split(Xtr, ytr, test_size=.25, stratify=ytr, random_state=seed)
    leaf = max(20, int(.001 * len(Xfit)))
    modelos = {
        "Decision Tree": (DecisionTreeClassifier(random_state=seed), {"max_depth": [3, 5], "min_samples_leaf": [leaf, leaf * 3]}),
        "Decision Tree balanceada": (DecisionTreeClassifier(class_weight="balanced", random_state=seed), {"max_depth": [3, 5], "min_samples_leaf": [leaf, leaf * 3]}),
        "Random Forest": (make_pipeline(SimpleImputer(add_indicator=True), RandomForestClassifier(n_estimators=40, class_weight="balanced_subsample", random_state=seed, n_jobs=1)), {"randomforestclassifier__max_depth": [5, 10], "randomforestclassifier__min_samples_leaf": [leaf, leaf * 3]}),
        "Logistic Regression": (make_pipeline(SimpleImputer(add_indicator=True), LogisticRegression(class_weight="balanced", max_iter=1000)), {"logisticregression__C": [.1, 1.0]}),
        "HistGradientBoosting": (HistGradientBoostingClassifier(random_state=seed), {"max_leaf_nodes": [15, 31], "learning_rate": [.05, .1]}),
    }
    opcionais = [
        ("XGBoost", "xgboost", "XGBClassifier", {"max_depth": [3, 5], "n_estimators": [100]}),
        ("LightGBM", "lightgbm", "LGBMClassifier", {"num_leaves": [15, 31]}),
        ("CatBoost", "catboost", "CatBoostClassifier", {"depth": [4, 6]}),
    ]
    for nome, modulo, classe, grade in opcionais:
        try:
            import importlib
            cls = getattr(importlib.import_module(modulo), classe)
            kwargs = {"random_state": seed} if nome != "CatBoost" else {"random_seed": seed, "verbose": False}
            if nome == "XGBoost":
                kwargs.update(scale_pos_weight=float((yfit == 0).sum() / (yfit == 1).sum()), eval_metric="logloss")
            elif nome == "LightGBM":
                kwargs.update(class_weight="balanced", verbosity=-1)
            else:
                kwargs.update(auto_class_weights="Balanced")
            modelos[nome] = (make_pipeline(SimpleImputer(add_indicator=True), cls(**kwargs)),
                             {classe.lower() + "__" + k: v for k, v in grade.items()})
        except ImportError:
            pass
    resultados = []
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    for nome, (estimador, grade) in modelos.items():
        print(f"Treinando {nome}...", flush=True)
        busca = GridSearchCV(estimador, grade, cv=cv, scoring="average_precision", n_jobs=2)
        busca.fit(Xfit, yfit)
        pv = busca.predict_proba(Xval)[:, 1]
        precisao, recall, limites = precision_recall_curve(yval, pv)
        f1 = 2 * precisao[:-1] * recall[:-1] / np.maximum(precisao[:-1] + recall[:-1], 1e-12)
        limiar = float(limites[int(np.argmax(f1))]) if len(limites) else .5
        # Reajuste somente no treino; teste é acessado uma única vez após seleção do limiar.
        busca.best_estimator_.fit(Xtr, ytr)
        p = busca.best_estimator_.predict_proba(Xte)[:, 1]
        pred = (p >= limiar).astype(int)
        fpr, tpr, _ = roc_curve(yte, p)
        pr, rc, _ = precision_recall_curve(yte, p)
        fpr, tpr = _curva(fpr, tpr)
        rc, pr = _curva(rc, pr)
        resultados.append({"modelo": nome, "auc_teste": float(roc_auc_score(yte, p)),
                           "average_precision": float(average_precision_score(yte, p)),
                           "limiar_validacao": limiar, "cv_ap": float(busca.best_score_),
                           "parametros": busca.best_params_,
                           "matriz_confusao": confusion_matrix(yte, pred, labels=[0, 1]).tolist(),
                           "roc": {"fpr": fpr, "tpr": tpr},
                           "pr": {"recall": rc, "precision": pr}})
    return {"gerado_em": datetime.now(timezone.utc).isoformat(), "n_treino": len(Xtr),
            "n_teste": len(Xte), "positivos_teste": int(yte.sum()), "prevalencia_teste": float(yte.mean()),
            "alvo": "CLASSI_FIN 11/12 vs 10; não representa grupos A/B/C/D", "features": FEATURES,
            "resultados": resultados}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--sep", default=",")
    ap.add_argument("--amostra-aleatoria", type=int)
    ap.add_argument("--saida", default="models/comparacao.json")
    args = ap.parse_args()
    df = (carregar_amostra_aleatoria(args.csv, args.amostra_aleatoria, args.sep)
          if args.amostra_aleatoria else carregar(args.csv, args.sep))
    resultado = comparar(df)
    resultado["origem"] = {"arquivo": args.csv, "amostra_aleatoria": args.amostra_aleatoria,
                           "semente": 42, "separador": args.sep}
    destino = Path(args.saida)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Comparação salva em {destino}")


if __name__ == "__main__":
    main()
