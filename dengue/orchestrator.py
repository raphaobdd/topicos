"""Orquestrador — controla o fluxo e a comunicação entre os agentes.

    Paciente → Agente 1 → Orquestrador → Agente 2 → Agente 3 → Agente 4 → Relatório

Segurança: após CADA turno de coleta o orquestrador aplica as regras explícitas
(safety.py). Se houver sinal de alerta, interrompe a sequência normal, mostra a
orientação de procurar atendimento e gera um relatório só com o que foi coletado.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from . import config
from .agents.analise import AgenteAnalise
from .agents.coleta import AgenteColeta
from .agents.orientacoes import AgenteOrientacoes
from .agents.relatorio import AgenteRelatorio
from .db import Repositorio, RepositorioNulo
from .llm import LLM
from .ml.predict import PreditorArvore
from .rag.retriever import Retriever
from .safety import mensagem_urgente, verificar_seguranca
from .schemas import ColetaEstado, ResultadoSessao


@dataclass
class Turno:
    mensagem: str
    etapa: Literal["coletando", "alerta", "coleta_concluida"]


class Orquestrador:
    def __init__(
        self,
        llm: LLM,
        retriever: Optional[Retriever] = None,
        preditor: Optional[PreditorArvore] = None,
        repositorio: Optional[Repositorio] = None,
    ):
        self.coleta = AgenteColeta(llm)
        self.analise = AgenteAnalise(llm, preditor)
        self.orientacoes = AgenteOrientacoes(llm, retriever)
        self.relatorio = AgenteRelatorio(llm)
        self.repositorio = repositorio or RepositorioNulo()
        self._retriever, self._preditor = retriever, preditor

    # ---- conversa ------------------------------------------------------ #
    def nova_sessao(self) -> ColetaEstado:
        return ColetaEstado()

    def iniciar(self, estado: ColetaEstado) -> str:
        return self.coleta.iniciar(estado)

    def turno(self, estado: ColetaEstado, texto: str) -> Turno:
        if estado.concluida:
            return Turno("A coleta já foi concluída.", "alerta" if estado.interrompida else "coleta_concluida")

        msg = self.coleta.responder(estado, texto)

        seg = verificar_seguranca(estado.dados)  # regras explícitas, sem LLM
        if seg.interromper:
            estado.interrompida = True
            estado.concluida = True
            urgente = mensagem_urgente(seg)
            estado.mensagens[-1].content = urgente  # substitui a próxima pergunta
            return Turno(urgente, "alerta")
        if estado.concluida:
            return Turno(msg, "coleta_concluida")
        return Turno(msg, "coletando")

    def encerrar_agora(self, estado: ColetaEstado) -> None:
        """Usuário quer parar: gera relatório com o que foi informado."""
        estado.concluida = True

    # ---- pipeline pós-coleta -------------------------------------------- #
    def executar(self, estado: ColetaEstado) -> ResultadoSessao:
        etapas = ["Agente 1: coleta de informações"]
        seg = verificar_seguranca(estado.dados)
        etapas.append("Orquestrador: verificação de sinais de alerta (regras explícitas)")
        if seg.nao_respondidos:
            etapas.append(f"Orquestrador: {len(seg.nao_respondidos)} sinal(is) de alerta sem resposta")

        urgente: Optional[str] = None
        if seg.interromper:
            estado.interrompida = True
            etapas.append("Fluxo normal INTERROMPIDO por sinal de alerta")
            analise = self.analise.analisar_urgente(estado.dados, seg)
            orient = self.orientacoes.orientacao_urgente(seg)
            urgente = mensagem_urgente(seg)
        else:
            analise = self.analise.analisar(estado.dados, seg)
            etapas.append(f"Agente 2: classificação experimental Grupo {analise.classificacao.grupo.value}"
                          + (" + árvore de decisão" if analise.arvore else ""))
            orient = self.orientacoes.orientar(estado.dados, analise)
            etapas.append("Agente 3: orientações" + (" (sem contexto de documentos)" if orient.contexto_insuficiente else " com RAG"))

        resumo = self.relatorio.resumo_ia(estado, seg, analise)
        md = self.relatorio.gerar(estado, seg, analise, orient, resumo)
        etapas.append("Agente 4: relatório final")

        resultado = ResultadoSessao(
            sessao_id=estado.sessao_id,
            interrompido=estado.interrompida,
            seguranca=seg,
            analise=analise,
            orientacoes=orient,
            mensagem_urgente=urgente,
            resumo_ia=resumo,
            relatorio_md=md,
            etapas=etapas,
        )
        try:
            self.repositorio.salvar(estado, resultado)
            etapas.append(f"Armazenamento: {self.repositorio.nome}")
        except Exception as e:  # persistência nunca derruba o atendimento
            etapas.append(f"Armazenamento falhou ({type(e).__name__})")
        return resultado

    # ---- status para a interface ---------------------------------------- #
    def status(self) -> dict[str, str]:
        rag = "indisponível"
        if self._retriever is not None:
            rag = f"{self._retriever.store.nome}" + ("" if self._retriever.disponivel() else " — vazio (rode a ingestão)")
        return {
            "LLM": config.DEEPSEEK_MODEL,
            "RAG": rag,
            "Árvore de decisão": (
                "ativa" if self._preditor and self._preditor.utilizavel
                else f"inativa ({self._preditor.motivo_nao_utilizavel})" if self._preditor
                else "sem modelo treinado"
            ),
            "Banco de dados": self.repositorio.nome,
        }


def criar_orquestrador() -> Orquestrador:
    from .db import criar_repositorio
    from .llm import DeepSeekLLM
    from .rag.store import criar_store

    llm = DeepSeekLLM()
    retriever = Retriever(llm, criar_store(), k=config.RAG_TOP_K)
    return Orquestrador(
        llm,
        retriever=retriever,
        preditor=PreditorArvore.carregar(),
        repositorio=criar_repositorio(),
    )
