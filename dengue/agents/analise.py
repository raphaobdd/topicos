"""Agente 2 — análise e classificação inicial.

Combina: (1) regras do protocolo (safety.classificar_manejo), (2) a árvore de decisão,
quando existir e for viável, e (3) um resumo em linguagem simples escrito pelo LLM.

O GRUPO (A/B/C/D) vem SEMPRE das regras. O LLM só redige o resumo e não pode alterá-lo.
"""
from __future__ import annotations

from typing import Optional

from ..llm import LLM
from ..ml.predict import PreditorArvore
from ..safety import classificar_manejo
from ..schemas import AnaliseInicial, DadosPaciente, ResultadoSeguranca

SISTEMA = """Você é o AGENTE DE ANÁLISE de um protótipo acadêmico de triagem de dengue.
Escreva UM parágrafo curto (3 a 5 frases), em português simples, resumindo a análise automática.
Regras:
- Use somente os dados fornecidos. Não invente fatos, não faça diagnóstico, não indique medicamentos.
- NÃO altere o grupo (A/B/C/D) nem a justificativa: apenas explique-os com clareza.
- Se houver estimativa da árvore de decisão, descreva-a como estatística de casos notificados, nunca como previsão individual.
- Deixe claro que a avaliação é experimental e que a avaliação profissional é indispensável.
"""


class AgenteAnalise:
    def __init__(self, llm: LLM, preditor: Optional[PreditorArvore] = None):
        self.llm = llm
        self.preditor = preditor

    # fluxo normal
    def analisar(self, dados: DadosPaciente, seg: ResultadoSeguranca) -> AnaliseInicial:
        classif = classificar_manejo(dados, seg)

        arvore, motivo = None, None
        if self.preditor is None:
            motivo = "modelo de árvore de decisão não disponível"
        elif not self.preditor.utilizavel:
            motivo = self.preditor.motivo_nao_utilizavel
        else:
            try:
                arvore = self.preditor.prever(dados)
            except Exception as e:  # nunca derruba o fluxo por causa do modelo
                motivo = f"falha ao executar o modelo ({type(e).__name__})"

        analise = AnaliseInicial(classificacao=classif, arvore=arvore, arvore_motivo_ausencia=motivo)
        analise.resumo = self._resumo(analise)
        return analise

    # fluxo interrompido por sinal de alerta: sem LLM e sem árvore (só regras)
    def analisar_urgente(self, dados: DadosPaciente, seg: ResultadoSeguranca) -> AnaliseInicial:
        classif = classificar_manejo(dados, seg)
        analise = AnaliseInicial(
            classificacao=classif,
            arvore=None,
            arvore_motivo_ausencia="fluxo interrompido por sinal de alerta (apenas regras de segurança foram aplicadas)",
        )
        analise.resumo = self._resumo_fixo(analise)
        return analise

    # ------------------------------------------------------------------ #
    def _resumo(self, a: AnaliseInicial) -> str:
        c = a.classificacao
        prompt = (
            f"Grupo (regras do protocolo): {c.grupo.value}\n"
            f"Justificativas: {' '.join(c.justificativas)}\n"
            f"Atenções: {' | '.join(c.atencoes[:4])}\n"
            f"Informações faltantes: {', '.join(c.dados_incompletos) or 'nenhuma'}\n"
        )
        if a.arvore:
            t = a.arvore
            prompt += (
                f"Árvore de decisão: probabilidade estimada {t.probabilidade:.1%} de o caso ser "
                f"classificado como dengue com sinais de alarme/grave; prevalência geral {t.prevalencia_base:.1%}; "
                f"nível {t.nivel}.\n"
            )
        else:
            prompt += f"Árvore de decisão: não utilizada ({a.arvore_motivo_ausencia}).\n"
        try:
            texto = self.llm.texto(SISTEMA, prompt, temperatura=0.2)
            return texto if texto else self._resumo_fixo(a)
        except Exception:
            return self._resumo_fixo(a)

    @staticmethod
    def _resumo_fixo(a: AnaliseInicial) -> str:
        c = a.classificacao
        txt = (
            f"Pelas regras do protocolo, a classificação automática e experimental é Grupo {c.grupo.value}. "
            + " ".join(c.justificativas)
            + " Esta avaliação não substitui a classificação feita por um profissional de saúde."
        )
        return txt
