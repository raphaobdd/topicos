"""Recuperação de trechos relevantes (RAG)."""
from __future__ import annotations

from ..llm import LLM
from .capitulos import permitido
from .store import Trecho, VectorStore


class Retriever:
    def __init__(self, llm: LLM, store: VectorStore, k: int = 4):
        self.llm = llm
        self.store = store
        self.k = k

    def disponivel(self) -> bool:
        try:
            return self.store.contar() > 0
        except Exception:
            return False

    def buscar(self, consulta: str, k: int | None = None, agente: int | None = None) -> list[Trecho]:
        emb = self.llm.embed([consulta], consulta=True)[0]
        limite = k or self.k
        if agente is None:
            return self.store.buscar(emb, limite)
        # A tabela atual não tem coluna de capítulo. Os trechos são marcados no
        # conteúdo, então filtramos antes de entregar o contexto ao agente.
        candidatos = self.store.buscar(emb, max(1000, self.store.contar()))
        return [t for t in candidatos if permitido(agente, t.fonte, t.conteudo)][:limite]

    def buscar_varias(self, consultas: list[str], k_total: int | None = None,
                     agente: int | None = None) -> list[Trecho]:
        """Busca para várias consultas, remove duplicados e devolve os melhores por similaridade."""
        if not consultas:
            return []
        embs = self.llm.embed(consultas, consulta=True)
        vistos: dict[tuple, Trecho] = {}
        limite_busca = max(1000, self.store.contar()) if agente is not None else self.k
        for emb in embs:
            for t in self.store.buscar(emb, limite_busca):
                if agente is not None and not permitido(agente, t.fonte, t.conteudo):
                    continue
                chave = (t.fonte, t.pagina, t.conteudo[:80])
                if chave not in vistos or t.similaridade > vistos[chave].similaridade:
                    vistos[chave] = t
        ordenados = sorted(vistos.values(), key=lambda t: -t.similaridade)
        return ordenados[: (k_total or self.k * 2)]
