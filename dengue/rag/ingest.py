"""Ingestão da base de conhecimento do RAG.

Uso:
    python -m dengue.rag.ingest                 # lê data/docs/*.pdf|txt|md
    python -m dengue.rag.ingest --dir outra/pasta
    python -m dengue.rag.ingest --local         # força armazenamento local (sem Supabase)

Coloque em data/docs os documentos oficiais (ex.: "Dengue: diagnóstico e manejo
clínico — adulto e criança", do Ministério da Saúde) e reexecute quando atualizar.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterator

from .. import config
from ..llm import LLM
from .capitulos import CLINICO, extrair_secoes_clinicas, marcar_capitulo
from .store import LocalStore, VectorStore, criar_store


def dividir_texto(texto: str, tamanho: int = 1000, sobreposicao: int = 150) -> list[str]:
    texto = re.sub(r"\s+", " ", texto).strip()
    chunks: list[str] = []
    i = 0
    while i < len(texto):
        fim = min(i + tamanho, len(texto))
        if fim < len(texto):  # tenta cortar no fim de uma frase
            corte = texto.rfind(". ", i + tamanho // 2, fim)
            if corte != -1:
                fim = corte + 1
        chunks.append(texto[i:fim].strip())
        if fim >= len(texto):
            break
        i = max(fim - sobreposicao, i + 1)
    return [c for c in chunks if len(c) > 50]


def ler_documentos(pasta: Path) -> Iterator[tuple[str, int | None, str, str]]:
    """Gera (fonte, página, texto). PDFs são lidos página a página."""
    for arq in sorted(pasta.glob("**/*")):
        sufixo = arq.suffix.lower()
        if sufixo == ".pdf":
            from pypdf import PdfReader

            leitor = PdfReader(str(arq))
            if arq.stem == CLINICO:
                for pagina, capitulo, texto in extrair_secoes_clinicas(leitor):
                    yield arq.stem, pagina, capitulo, texto
                continue
            for n, pagina in enumerate(leitor.pages, start=1):
                yield arq.stem, n, "integral", pagina.extract_text() or ""
        elif sufixo in (".txt", ".md"):
            yield arq.stem, None, "integral", arq.read_text(encoding="utf-8", errors="replace")


def indexar(llm: LLM, store: VectorStore, pasta: Path, tamanho: int = 1000, sobreposicao: int = 150) -> int:
    por_fonte: dict[str, list[dict]] = {}
    for fonte, pagina, capitulo, texto in ler_documentos(pasta):
        for chunk in dividir_texto(texto, tamanho, sobreposicao):
            por_fonte.setdefault(fonte, []).append(
                {"fonte": fonte, "pagina": pagina, "conteudo": marcar_capitulo(capitulo, chunk)})

    total = 0
    for fonte, registros in por_fonte.items():
        vetores: list[list[float]] = []
        textos = [r["conteudo"] for r in registros]
        for i in range(0, len(textos), 50):
            vetores.extend(llm.embed(textos[i : i + 50], consulta=False))
        for r, v in zip(registros, vetores):
            r["embedding"] = v
        store.limpar_fonte(fonte)  # reindexação idempotente
        store.adicionar(registros)
        total += len(registros)
        print(f"  {fonte}: {len(registros)} trechos")
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description="Indexa documentos para o RAG")
    ap.add_argument("--dir", default=str(config.DOCS_DIR))
    ap.add_argument("--local", action="store_true", help="usa o armazenamento local (JSON)")
    ap.add_argument("--supabase", action="store_true", help="força armazenamento no Supabase")
    args = ap.parse_args()
    if args.local and args.supabase:
        ap.error("Escolha apenas --local ou --supabase")

    from ..llm import LocalEmbedder

    pasta = Path(args.dir)
    if not pasta.exists() or not any(pasta.iterdir()):
        raise SystemExit(f"Nenhum documento em {pasta}. Coloque PDFs/TXT do Ministério da Saúde lá.")
    if args.supabase:
        if not (config.SUPABASE_URL and config.SUPABASE_KEY):
            raise SystemExit("Configure SUPABASE_URL e SUPABASE_SERVICE_KEY para --supabase")
        from .store import SupabaseStore
        store = SupabaseStore()
    else:
        store = LocalStore() if args.local else criar_store()
    print(f"Indexando {pasta} em: {store.nome}")
    n = indexar(LocalEmbedder(), store, pasta)
    print(f"Pronto: {n} trechos indexados.")


if __name__ == "__main__":
    main()
