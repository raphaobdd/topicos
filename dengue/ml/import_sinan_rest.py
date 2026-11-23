"""Carrega uma amostra da DENGBR24 pela API HTTPS do Supabase, sem URL PostgreSQL.

Uso: python -m dengue.ml.import_sinan_rest --csv data/DENGBR24.csv.zip
Usa SUPABASE_URL e SUPABASE_SERVICE_KEY do .env. A seleção é reproduzível.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from .. import config
from .import_sinan import COLUNAS, linhas_csv, sha256_arquivo


def selecionar(caminho: Path, fracao: float, semente: int = 42):
    if not 0 < fracao <= 1:
        raise ValueError("fracao deve estar entre 0 e 1")
    rng = random.Random(semente)
    for numero, valores in linhas_csv(caminho):
        if rng.random() < fracao:
            yield numero, valores


def importar(caminho: Path, client, fracao: float = 0.015, lote: int = 250) -> dict:
    if lote < 1:
        raise ValueError("lote deve ser positivo")
    sha = sha256_arquivo(caminho)
    enviados = 0
    bloco = []

    def enviar(registros):
        for tentativa in range(4):
            try:
                client.table("sinan_dengue_2024").upsert(
                    registros, on_conflict="fonte_sha256,linha_arquivo").execute()
                return
            except Exception:
                if tentativa == 3:
                    raise
                time.sleep(2 ** tentativa)

    for numero, valores in selecionar(caminho, fracao):
        bloco.append({"fonte_sha256": sha, "linha_arquivo": numero,
                      **dict(zip(COLUNAS, valores))})
        if len(bloco) >= lote:
            enviar(bloco)
            enviados += len(bloco)
            if enviados % 5000 < lote:
                print(f"Enviadas {enviados:,} linhas da amostra", flush=True)
            bloco = []
    if bloco:
        enviar(bloco)
        enviados += len(bloco)
    return {"fonte_sha256": sha, "fracao": fracao, "semente": 42,
            "linhas_amostra": enviados, "tipo": "amostra; nao e a base completa"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="data/DENGBR24.csv.zip")
    ap.add_argument("--fracao", type=float, default=0.015,
                    help="proporção de linhas enviada, padrão 0.015 (~98 mil linhas)")
    ap.add_argument("--lote", type=int, default=250)
    args = ap.parse_args()
    if not config.SUPABASE_URL or not config.SUPABASE_KEY:
        raise SystemExit("Configure SUPABASE_URL e SUPABASE_SERVICE_KEY no .env")
    from supabase import create_client
    resultado = importar(Path(args.csv), create_client(config.SUPABASE_URL, config.SUPABASE_KEY),
                         args.fracao, args.lote)
    print(json.dumps(resultado, ensure_ascii=False))


if __name__ == "__main__":
    main()
