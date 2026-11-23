"""Variáveis da árvore de decisão — UMA definição usada no treino e na inferência
(evita train/serve skew).

Pergunta do modelo (alvo): entre casos de dengue NOTIFICADOS e classificados como
dengue / dengue com sinais de alarme / dengue grave (CLASSI_FIN 10, 11, 12), qual a
chance de o caso ser classificado como "com sinais de alarme ou grave" (11 ou 12),
usando APENAS o que existe na hora da triagem:
  idade, sexo, gestação, dias de sintoma, sintomas iniciais e comorbidades.

Ficam FORA de propósito (data leakage / indisponíveis na triagem):
  ALRM_*, GRAV_* (sinais observados durante a evolução e que definem o alvo),
  HOSPITALIZ, EVOLUCAO, CRITERIO, resultados de exames e datas posteriores à notificação.
Valores desconhecidos (código 9 / vazio) viram NaN — "ignorado" != "não".

ATENÇÃO: confira os códigos no dicionário de dados do SINAN-Dengue (DENGBR24).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..schemas import DadosPaciente, Resposta

FEATURES = [
    "idade_anos",
    "sexo_feminino",
    "gestante",
    "dias_sintomas",
    "febre",
    "cefaleia",
    "dor_retro_orbitaria",
    "mialgia",
    "artralgia",
    "nausea",
    "vomito",
    "exantema",
    "diabetes",
    "hipertensao",
    "doenca_hematologica",
    "doenca_hepatica",
    "doenca_renal",
    "doenca_autoimune",
]

# feature -> coluna SINAN (1 = sim, 2 = não, 9/vazio = ignorado)
SINAN_BINARIAS = {
    "febre": "FEBRE",
    "cefaleia": "CEFALEIA",
    "dor_retro_orbitaria": "DOR_RETRO",
    "mialgia": "MIALGIA",
    "artralgia": "ARTRALGIA",
    "nausea": "NAUSEA",
    "vomito": "VOMITO",
    "exantema": "EXANTEMA",
    "diabetes": "DIABETES",
    "hipertensao": "HIPERTENSA",
    "doenca_hematologica": "HEMATOLOG",
    "doenca_hepatica": "HEPATOPAT",
    "doenca_renal": "RENAL",
    "doenca_autoimune": "AUTO_IMUNE",
}

COLUNAS_SINAN = sorted(
    set(SINAN_BINARIAS.values())
    | {"NU_IDADE_N", "CS_SEXO", "CS_GESTANT", "DT_NOTIFIC", "DT_SIN_PRI", "CLASSI_FIN", "NU_NOTIFIC", "ID_MUNICIP", "SG_UF_NOT"}
)

ALVO_DESCRICAO = (
    "Caso de dengue notificado classificado como 'com sinais de alarme' ou 'grave' "
    "(CLASSI_FIN 11/12) vs. 'dengue' (10)"
)


# --------------------------------------------------------------------------- #
# Treino: SINAN -> X, y
# --------------------------------------------------------------------------- #
def _sn(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").map({1.0: 1.0, 2.0: 0.0})


def idade_em_anos(s: pd.Series) -> pd.Series:
    """NU_IDADE_N: 1º dígito = unidade (1 h, 2 dia, 3 mês, 4 ano), demais = valor."""
    n = pd.to_numeric(s, errors="coerce")
    unidade, valor = n // 1000, n % 1000
    anos = np.select(
        [unidade == 4, unidade == 3, unidade == 2, unidade == 1],
        [valor, valor / 12, valor / 365.25, valor / 8766],
        default=np.nan,
    )
    return pd.Series(anos, index=s.index)


def _gestante(s: pd.Series) -> pd.Series:
    n = pd.to_numeric(s, errors="coerce")
    out = pd.Series(np.nan, index=s.index)
    out[n.isin([1, 2, 3, 4])] = 1.0
    out[n.isin([5, 6])] = 0.0  # 5 = não, 6 = não se aplica
    return out


def preparar_sinan(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame(index=df.index)
    d["idade_anos"] = idade_em_anos(df["NU_IDADE_N"])
    d["sexo_feminino"] = df["CS_SEXO"].map({"F": 1.0, "M": 0.0})
    d["gestante"] = _gestante(df["CS_GESTANT"])
    notif = pd.to_datetime(df["DT_NOTIFIC"], errors="coerce")
    inicio = pd.to_datetime(df["DT_SIN_PRI"], errors="coerce")
    dias = (notif - inicio).dt.days
    d["dias_sintomas"] = dias.where(dias.between(0, 20))
    for feat, col in SINAN_BINARIAS.items():
        d[feat] = _sn(df[col])
    return d[FEATURES].astype("float32")


def alvo_sinan(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(máscara de linhas elegíveis, y). Elegíveis: CLASSI_FIN em {10, 11, 12}."""
    cf = pd.to_numeric(df["CLASSI_FIN"], errors="coerce")
    elegiveis = cf.isin([10, 11, 12])
    y = cf.isin([11, 12]).astype(int)
    return elegiveis, y


# --------------------------------------------------------------------------- #
# Inferência: DadosPaciente -> linha de features
# --------------------------------------------------------------------------- #
def _r(v: Resposta) -> float:
    return 1.0 if v == Resposta.SIM else 0.0 if v == Resposta.NAO else math.nan


def features_paciente(d: DadosPaciente) -> dict[str, float]:
    i, e, s, h = d.identificacao, d.evolucao, d.sintomas, d.historico
    return {
        "idade_anos": float(i.idade) if i.idade is not None else math.nan,
        "sexo_feminino": {"F": 1.0, "M": 0.0}.get(i.sexo, math.nan),
        "gestante": _r(i.gravidez),
        "dias_sintomas": float(e.dias_sintomas) if e.dias_sintomas is not None else math.nan,
        "febre": _r(e.febre),
        "cefaleia": _r(s.dor_cabeca),
        "dor_retro_orbitaria": _r(s.dor_atras_olhos),
        "mialgia": _r(s.dor_muscular),
        "artralgia": _r(s.dor_articulacoes),
        "nausea": _r(s.nausea),
        "vomito": _r(s.vomitos),
        "exantema": _r(s.manchas_pele),
        "diabetes": _r(h.diabetes),
        "hipertensao": _r(h.hipertensao),
        "doenca_hematologica": _r(h.doenca_hematologica),
        "doenca_hepatica": _r(h.doenca_hepatica),
        "doenca_renal": _r(h.doenca_renal),
        "doenca_autoimune": _r(h.doenca_autoimune),
    }
