"""Integração da árvore de decisão na aplicação (Agente 2)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from .. import config
from ..schemas import DadosPaciente, PrevisaoArvore
from .features import features_paciente


class PreditorArvore:
    def __init__(self, pacote: dict):
        self.pacote = pacote
        self.modelo = pacote["modelo"]
        self.features: list[str] = pacote["features"]

    @classmethod
    def carregar(cls, caminho: Path | str | None = None) -> Optional["PreditorArvore"]:
        caminho = Path(caminho or config.MODELO_ARVORE)
        if not caminho.exists():
            return None
        try:
            return cls(joblib.load(caminho))
        except Exception:
            return None

    @property
    def utilizavel(self) -> bool:
        return bool(self.pacote.get("viavel")) and not self.pacote.get("sintetico")

    @property
    def motivo_nao_utilizavel(self) -> str:
        if self.pacote.get("sintetico"):
            return "modelo treinado em dados sintéticos (apenas teste de pipeline)"
        if not self.pacote.get("viavel"):
            auc = self.pacote.get("metricas", {}).get("auc_teste")
            return f"desempenho insuficiente no teste (AUC {auc:.2f}) para integração" if auc else "modelo não considerado viável"
        return ""

    def prever(self, dados: DadosPaciente) -> PrevisaoArvore:
        linha = features_paciente(dados)
        x = pd.DataFrame([[linha[f] for f in self.features]], columns=self.features).astype("float32")
        prob = float(self.modelo.predict_proba(x)[0, 1])
        prev = float(self.pacote["prevalencia"])
        razao = prob / prev if prev > 0 else math.nan
        nivel = "abaixo da média" if razao < 0.75 else "acima da média" if razao > 1.5 else "semelhante à média"
        ausentes = sum(1 for f in self.features if math.isnan(linha[f]))
        avisos = [
            "Estimativa estatística baseada em casos notificados no SINAN; não é diagnóstico nem prognóstico individual.",
            "O modelo usa só dados disponíveis na triagem (sem sinais de alarme): ele NÃO substitui a verificação de sinais de alerta.",
        ]
        if ausentes:
            avisos.append(f"{ausentes} variável(is) do modelo não foram informadas e entraram como ausentes.")
        return PrevisaoArvore(
            alvo=self.pacote["alvo"],
            probabilidade=prob,
            prevalencia_base=prev,
            razao_vs_base=razao,
            nivel=nivel,
            auc_teste=self.pacote.get("metricas", {}).get("auc_teste"),
            n_variaveis_ausentes=ausentes,
            avisos=avisos,
        )


class PreditorXGBoost(PreditorArvore):
    """Mant?m o contrato de sa?da existente para o modelo XGBoost."""

    @classmethod
    def carregar(cls, caminho: Path | str | None = None) -> Optional["PreditorXGBoost"]:
        return super().carregar(caminho or config.MODELO_XGBOOST)

    def prever(self, dados: DadosPaciente) -> PrevisaoArvore:
        resultado = super().prever(dados)
        resultado.avisos[0] = (
            "Escore estat?stico XGBoost treinado com pondera??o de classes em casos SINAN; "
            "n?o ? probabilidade cl?nica calibrada, diagn?stico ou progn?stico individual."
        )
        return resultado
