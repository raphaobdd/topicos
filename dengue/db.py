"""Persistência de sessões (Supabase/PostgreSQL).

Privacidade: o app não coleta nome/CPF/endereço; a sessão é identificada por UUID.
Use dados sintéticos ou desidentificados em testes e demonstrações. A chave
`service_role` só deve existir no backend (nunca no navegador).
"""
from __future__ import annotations

from typing import Protocol

from . import config
from .schemas import ColetaEstado, ResultadoSessao


class Repositorio(Protocol):
    nome: str

    def salvar(self, estado: ColetaEstado, resultado: ResultadoSessao) -> None: ...


class RepositorioNulo:
    nome = "desativado (sem Supabase)"

    def salvar(self, estado: ColetaEstado, resultado: ResultadoSessao) -> None:
        return None


class RepositorioSupabase:
    nome = "Supabase"

    def __init__(self, url: str | None = None, key: str | None = None):
        from supabase import create_client

        self.client = create_client(url or config.SUPABASE_URL, key or config.SUPABASE_KEY)

    def salvar(self, estado: ColetaEstado, resultado: ResultadoSessao) -> None:
        linha = {
            "id": estado.sessao_id,
            "criada_em": estado.criada_em.isoformat(),
            "interrompido": resultado.interrompido,
            "dados": estado.dados.model_dump(mode="json"),
            "mensagens": [m.model_dump() for m in estado.mensagens],
            "seguranca": resultado.seguranca.model_dump(mode="json"),
            "analise": resultado.analise.model_dump(mode="json"),
            "orientacoes": resultado.orientacoes.model_dump(mode="json"),
            "relatorio_md": resultado.relatorio_md,
        }
        self.client.table("sessoes").upsert(linha).execute()


def criar_repositorio() -> Repositorio:
    if config.SUPABASE_URL and config.SUPABASE_KEY:
        return RepositorioSupabase()
    return RepositorioNulo()
