"""Importa apenas as colunas usadas da DENGBR24 para PostgreSQL/Supabase.

Uso: python -m dengue.ml.import_sinan --csv data/DENGBR24.csv.zip
Requer SUPABASE_DB_URL (conexao direta ou session pooler, nao transaction pooler).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

from dotenv import load_dotenv

from .features import COLUNAS_SINAN

COLUNAS = sorted(COLUNAS_SINAN)
TABELA = "sinan_dengue_2024"


def sha256_arquivo(caminho: Path) -> str:
    h = hashlib.sha256()
    with caminho.open("rb") as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloco)
    return h.hexdigest()


def linhas_csv(caminho: Path):
    """Lê ZIP/CSV em fluxo, sem carregar a base inteira na memoria."""
    if caminho.suffix.lower() == ".zip":
        with zipfile.ZipFile(caminho) as z:
            nomes = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if len(nomes) != 1:
                raise ValueError(f"ZIP deve conter um CSV; encontrados: {nomes}")
            with z.open(nomes[0]) as bruto, io.TextIOWrapper(bruto, encoding="utf-8-sig", errors="replace", newline="") as f:
                yield from _ler(f)
    else:
        with caminho.open(encoding="utf-8-sig", errors="replace", newline="") as f:
            yield from _ler(f)


def _ler(f):
    leitor = csv.DictReader(f)
    # O recurso CSV oficial de 2024 nao publica NU_NOTIFIC.
    opcionais = {"NU_NOTIFIC"}
    faltantes = set(COLUNAS) - set(leitor.fieldnames or []) - opcionais
    if faltantes:
        raise ValueError(f"Colunas ausentes no CSV: {sorted(faltantes)}")
    for numero, linha in enumerate(leitor, 2):
        yield numero, [linha.get(c) or None for c in COLUNAS]


def importar(caminho: Path, dsn: str, lote: int = 10000) -> dict:
    import psycopg

    sha = sha256_arquivo(caminho)
    colunas_sql = ", ".join(f'"{c}"' for c in COLUNAS)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {TABELA} WHERE fonte_sha256 = %s", (sha,))
            existentes = cur.fetchone()[0]
            if existentes:
                return {"sha256": sha, "linhas": existentes, "estado": "ja_importado"}
            cur.execute(f"DELETE FROM {TABELA} WHERE fonte_sha256 = %s", (sha,))
            # Uma transacao: falha deixa a tabela sem carga parcial dessa versao.
            total = 0
            for bloco in _lotes(linhas_csv(caminho), lote):
                with cur.copy(f"COPY {TABELA} (fonte_sha256, linha_arquivo, {colunas_sql}) FROM STDIN") as cp:
                    for numero, valores in bloco:
                        cp.write_row((sha, numero, *valores))
                total += len(bloco)
                print(f"Importadas {total:,} linhas", flush=True)
            cur.execute(f"SELECT count(*) FROM {TABELA} WHERE fonte_sha256 = %s", (sha,))
            if cur.fetchone()[0] != total:
                raise RuntimeError("Contagem no banco difere do CSV")
        conn.commit()
    return {"sha256": sha, "linhas": total, "estado": "importado"}


def _lotes(iteravel, tamanho):
    bloco = []
    for item in iteravel:
        bloco.append(item)
        if len(bloco) >= tamanho:
            yield bloco
            bloco = []
    if bloco:
        yield bloco


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="data/DENGBR24.csv.zip")
    ap.add_argument("--lote", type=int, default=10000)
    args = ap.parse_args()
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    dsn = os.getenv("SUPABASE_DB_URL")
    if not dsn:
        raise SystemExit("Configure SUPABASE_DB_URL no .env; execute supabase_schema.sql antes da carga.")
    print(json.dumps(importar(Path(args.csv), dsn, args.lote), ensure_ascii=False))


if __name__ == "__main__":
    main()
