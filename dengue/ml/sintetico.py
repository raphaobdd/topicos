"""Base SINTÉTICA no formato do DENGBR24 — só para testar o pipeline.
Modelos treinados nela são marcados como `sintetico` e NÃO são usados pela aplicação."""
from __future__ import annotations

import numpy as np
import pandas as pd


def gerar(n: int = 5000, semente: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(semente)

    def sn(p):  # 1 = sim, 2 = não, 9 = ignorado (~3%)
        v = np.where(rng.random(n) < p, 1, 2)
        return np.where(rng.random(n) < 0.03, 9, v)

    idade = rng.integers(1, 90, n)
    notif = pd.Timestamp("2024-03-01") + pd.to_timedelta(rng.integers(0, 90, n), unit="D")
    dias = rng.integers(0, 8, n)
    df = pd.DataFrame(
        {
            "NU_NOTIFIC": np.arange(n),
            "DT_NOTIFIC": notif.strftime("%Y-%m-%d"),
            "DT_SIN_PRI": (notif - pd.to_timedelta(dias, unit="D")).strftime("%Y-%m-%d"),
            "NU_IDADE_N": 4000 + idade,
            "CS_SEXO": rng.choice(["F", "M", "I"], n, p=[0.52, 0.47, 0.01]),
            "ID_MUNICIP": rng.integers(350000, 351000, n),
            "SG_UF_NOT": 35,
        }
    )
    df["CS_GESTANT"] = np.where(df["CS_SEXO"] == "F", rng.choice([1, 2, 5, 6, 9], n, p=[0.02, 0.02, 0.5, 0.4, 0.06]), 6)
    df["FEBRE"], df["CEFALEIA"], df["MIALGIA"] = sn(0.9), sn(0.7), sn(0.7)
    df["EXANTEMA"], df["VOMITO"], df["NAUSEA"] = sn(0.2), sn(0.25), sn(0.35)
    df["DOR_RETRO"], df["ARTRALGIA"] = sn(0.35), sn(0.3)
    df["DIABETES"], df["HEMATOLOG"], df["HEPATOPAT"] = sn(0.06), sn(0.01), sn(0.01)
    df["RENAL"], df["HIPERTENSA"], df["AUTO_IMUNE"] = sn(0.01), sn(0.15), sn(0.01)

    # risco latente: idade extrema, vômito, comorbidades, consulta mais tardia
    z = (
        -3.2
        + 0.8 * ((idade >= 65) | (idade < 3))
        + 0.9 * (df["VOMITO"] == 1)
        + 0.7 * (df["DIABETES"] == 1)
        + 0.5 * (df["HIPERTENSA"] == 1)
        + 0.15 * dias
    )
    p = 1 / (1 + np.exp(-z))
    grave = rng.random(n) < p
    cf = np.where(grave, rng.choice([11, 12], n, p=[0.9, 0.1]), 10)
    descartado = rng.random(n) < 0.25
    df["CLASSI_FIN"] = np.where(descartado, rng.choice([5, 8], n), cf)
    return df
