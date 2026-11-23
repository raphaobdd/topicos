"""Armazenamento vetorial do RAG.

- SupabaseStore: Postgres + pgvector (tabela `documentos` + função `match_documentos`,
  veja supabase_schema.sql).
- LocalStore: arquivo JSON + numpy. Serve para desenvolver/testar o RAG antes de
  configurar o Supabase (ou sem internet).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Protocol

import numpy as np
from pydantic import BaseModel

from .. import config


class Trecho(BaseModel):
    fonte: str
    pagina: Optional[int] = None
    conteudo: str
    similaridade: float = 0.0


class VectorStore(Protocol):
    nome: str

    def adicionar(self, registros: list[dict]) -> None: ...
    def buscar(self, embedding: list[float], k: int) -> list[Trecho]: ...
    def limpar_fonte(self, fonte: str) -> None: ...
    def contar(self) -> int: ...


class LocalStore:
    nome = "local (JSON)"

    def __init__(self, caminho: Path | str | None = None):
        self.caminho = Path(caminho or config.LOCAL_STORE)
        self._dados: list[dict] = []
        if self.caminho.exists():
            self._dados = json.loads(self.caminho.read_text(encoding="utf-8"))

    def _salvar(self) -> None:
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        self.caminho.write_text(json.dumps(self._dados, ensure_ascii=False), encoding="utf-8")

    def adicionar(self, registros: list[dict]) -> None:
        self._dados.extend(registros)
        self._salvar()

    def limpar_fonte(self, fonte: str) -> None:
        self._dados = [r for r in self._dados if r["fonte"] != fonte]
        self._salvar()

    def contar(self) -> int:
        return len(self._dados)

    def buscar(self, embedding: list[float], k: int) -> list[Trecho]:
        if not self._dados:
            return []
        m = np.array([r["embedding"] for r in self._dados], dtype=float)
        q = np.array(embedding, dtype=float)
        sim = (m @ q) / ((np.linalg.norm(m, axis=1) * np.linalg.norm(q)) + 1e-12)
        idx = np.argsort(-sim)[:k]
        return [
            Trecho(
                fonte=self._dados[i]["fonte"],
                pagina=self._dados[i].get("pagina"),
                conteudo=self._dados[i]["conteudo"],
                similaridade=float(sim[i]),
            )
            for i in idx
        ]


class SupabaseStore:
    nome = "Supabase (pgvector)"

    def __init__(self, url: str | None = None, key: str | None = None):
        from supabase import create_client

        self.client = create_client(url or config.SUPABASE_URL, key or config.SUPABASE_KEY)

    def adicionar(self, registros: list[dict]) -> None:
        for i in range(0, len(registros), 100):
            self.client.table("documentos").insert(registros[i : i + 100]).execute()

    def limpar_fonte(self, fonte: str) -> None:
        self.client.table("documentos").delete().eq("fonte", fonte).execute()

    def contar(self) -> int:
        r = self.client.table("documentos").select("id", count="exact").limit(1).execute()
        return int(r.count or 0)

    def buscar(self, embedding: list[float], k: int) -> list[Trecho]:
        r = self.client.rpc("match_documentos", {"query_embedding": embedding, "match_count": k}).execute()
        return [
            Trecho(
                fonte=row["fonte"],
                pagina=row.get("pagina"),
                conteudo=row["conteudo"],
                similaridade=float(row.get("similaridade", 0.0)),
            )
            for row in (r.data or [])
        ]


def criar_store() -> VectorStore:
    if config.RAG_STORE not in {"auto", "local", "supabase"}:
        raise ValueError("RAG_STORE deve ser auto, local ou supabase")
    if config.RAG_STORE == "local" or (config.RAG_STORE == "auto" and config.LOCAL_STORE.exists()):
        return LocalStore()
    if config.SUPABASE_URL and config.SUPABASE_KEY:
        return SupabaseStore()
    if config.RAG_STORE == "supabase":
        raise ValueError("RAG_STORE=supabase requer SUPABASE_URL e SUPABASE_SERVICE_KEY")
    return LocalStore()
