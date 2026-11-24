"""Agente 4 — relatório final.

O corpo do relatório é montado de forma DETERMINÍSTICA a partir dos dados
estruturados (nada é "reescrito" pelo LLM, então nada é inventado). O LLM só
escreve um resumo curto em linguagem simples, claramente identificado como
gerado por IA. Cada seção indica a origem da informação:
  [Paciente]     relatado pelo paciente
  [Sistema]      produzido pelas regras / modelo / agentes
  [Profissional] depende de avaliação profissional
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from ..llm import LLM
from ..rag.retriever import Retriever
from ..perguntas import PERGUNTAS, ROTULOS, todos_campos_perguntados
from ..safety import ALERTAS
from ..schemas import (
    AnaliseInicial,
    ColetaEstado,
    DESCRICAO_GRUPOS,
    Orientacoes,
    Resposta,
    ResultadoSeguranca,
)

SISTEMA_RESUMO = """Você escreve o "resumo para o paciente" no topo de um relatório de triagem de dengue (protótipo acadêmico).
Escreva 3 a 4 frases em português simples e acolhedor sobre O QUE FAZER AGORA (próximos passos), SOMENTE com os dados fornecidos.
Não repita a análise técnica, não faça diagnóstico, não indique medicamentos, não invente dados.
Termine lembrando que a avaliação profissional é indispensável."""

AVISO = (
    "Este relatório foi gerado por um protótipo acadêmico e experimental. Não é diagnóstico, não substitui "
    "consulta médica e não deve ser usado para decisões clínicas."
)

_RESP = {Resposta.SIM: "Sim", Resposta.NAO: "Não", Resposta.NAO_INFORMADO: "Não informado"}
_VALORES = {"F": "Feminino", "M": "Masculino", "melhorando": "Melhorando", "piorando": "Piorando", "igual": "Permanecem iguais"}


class AgenteRelatorio:
    def __init__(self, llm: Optional[LLM] = None, retriever: Optional[Retriever] = None):
        self.llm = llm
        self.retriever = retriever

    def resumo_ia(self, estado: ColetaEstado, seg: ResultadoSeguranca, analise: AnaliseInicial) -> Optional[str]:
        if self.llm is None:
            return None
        d = estado.dados
        prompt = (
            f"Grupo (regras): {analise.classificacao.grupo.value}\n"
            f"Sinais de alerta relatados: {', '.join(seg.sinais_presentes) or 'nenhum'}\n"
            f"Idade: {d.identificacao.idade}; dias de sintomas: {d.evolucao.dias_sintomas}\n"
            f"Fluxo interrompido por sinal de alerta: {'sim' if estado.interrompida else 'não'}\n"
            f"Justificativas: {' '.join(analise.classificacao.justificativas)}"
        )
        try:
            return self.llm.texto(SISTEMA_RESUMO, prompt, temperatura=0.2) or None
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    def gerar(
        self,
        estado: ColetaEstado,
        seg: ResultadoSeguranca,
        analise: AnaliseInicial,
        orient: Orientacoes,
        resumo_ia: Optional[str] = None,
        agora: Optional[datetime] = None,
    ) -> str:
        d = estado.dados
        i, e, s, h, x = d.identificacao, d.evolucao, d.sintomas, d.historico, d.exames
        agora = agora or datetime.now()
        L: list[str] = []

        def tab(campos: list[str]) -> None:
            L.append("| Item | Resposta |\n|---|---|")
            for c in campos:
                L.append(f"| {ROTULOS[c]} | {self._fmt(d, c, estado)} |")
            L.append("")

        L += [
            "# Relatório de avaliação inicial — suspeita de dengue",
            f"> ⚠️ {AVISO}",
            "",
            f"**Sessão:** `{estado.sessao_id}` · **Gerado em:** {agora:%d/%m/%Y %H:%M}",
            "",
            "**Origem das informações:** `[Paciente]` relatado pelo paciente · `[Sistema]` produzido por regras/modelo/agentes · `[Profissional]` depende de avaliação profissional.",
            "",
        ]
        if estado.interrompida:
            L += [
                "## ⚠️ FLUXO INTERROMPIDO — SINAL(IS) DE ALERTA",
                "A conversa foi interrompida porque foram relatados sinais de alerta. **Procure atendimento médico imediatamente.** "
                "Itens não perguntados constam como *não perguntado*.",
                "",
            ]
        if resumo_ia:
            L += ["## Resumo para o paciente (gerado por IA)", resumo_ia, ""]

        L += ["## 1. Identificação básica `[Paciente]`"]
        tab(["identificacao.idade", "identificacao.sexo", "identificacao.peso_kg", "identificacao.gravidez"])

        L += [
            "## 2. Motivo da interação `[Paciente]`",
            "Avaliação inicial de suspeita de dengue por assistente conversacional (protótipo acadêmico). "
            "Nenhum dado de identificação pessoal (nome, CPF, endereço) é solicitado.",
            "",
            "## 3. Início e evolução dos sintomas `[Paciente]`",
        ]
        tab(["evolucao.dias_sintomas", "evolucao.febre", "evolucao.febre_max_c", "evolucao.febre_continua", "evolucao.tendencia"])
        if e.data_inicio:
            L += [f"Data aproximada de início (calculada a partir dos dias informados): {e.data_inicio:%d/%m/%Y}.", ""]

        L += ["## 4. Sintomas relatados `[Paciente]`"]
        tab([c for c in d.caminhos() if c.startswith("sintomas.")])

        L += ["## 5. Sinais de alerta investigados `[Paciente]`"]
        tab(list(ALERTAS))
        if seg.sinais_presentes:
            L += ["**Sinais de alerta presentes:** " + "; ".join(seg.sinais_presentes) + ".", ""]

        L += ["## 6. Histórico de saúde e condições preexistentes `[Paciente]`"]
        tab(
            [
                "historico.doenca_preexistente", "historico.doenca_preexistente_detalhe",
                "historico.uso_continuo_medicamento", "historico.medicamento_detalhe",
                "historico.alergia", "historico.alergia_detalhe", "historico.teve_dengue_antes",
                "historico.hipertensao", "historico.diabetes", "historico.doenca_cardiaca",
                "historico.doenca_renal", "historico.doenca_pulmonar", "historico.doenca_hepatica",
                "historico.doenca_hematologica", "historico.doenca_autoimune", "historico.anticoagulantes",
                "historico.outra_condicao", "historico.outra_condicao_detalhe",
            ]
        )

        L += ["## 7. Exames informados `[Paciente]`"]
        tab(["exames.hemograma_disponivel", "exames.hematocrito_pct", "exames.plaquetas"])
        L += ["Resultados de exames não são presumidos: o que não foi informado consta como *não informado*.", ""]

        c = analise.classificacao
        L += [
            "## 8. Resultado da análise (Agente 2) `[Sistema]`",
            f"**Classificação automatizada experimental (regras do protocolo): Grupo {c.grupo.value}** — {DESCRICAO_GRUPOS[c.grupo.value]}",
            "",
            "**Justificativa:**",
            *[f"- {j}" for j in c.justificativas],
            "",
            "**Pontos de atenção:**",
            *[f"- {a}" for a in c.atencoes],
            "",
            analise.resumo,
            "",
            "> A classificação definitiva depende de avaliação clínica e dos critérios do protocolo.",
            "",
            "## 9. XGBoost `[Sistema]`",
        ]
        if analise.arvore:
            t = analise.arvore
            L += [
                f"- Alvo do modelo: {t.alvo}.",
                f"- Escore XGBoost ponderado: **{t.probabilidade:.1%}** (não é probabilidade clínica calibrada).",
                f"- AUC no teste: {t.auc_teste:.2f}" if t.auc_teste else "- AUC no teste: não disponível",
                *[f"- ⚠️ {a}" for a in t.avisos],
                "",
            ]
        else:
            L += [f"Não utilizada: {analise.arvore_motivo_ausencia}.", ""]

        L += ["## 10. Orientações gerais (Agente 3) `[Sistema, com base em documentos]`"]
        if orient.aviso:
            L += [f"> {orient.aviso}", ""]
        L += self._itens(orient.orientacoes)
        if orient.quando_procurar_atendimento:
            L += ["", "**Procure atendimento se houver:**", *self._itens(orient.quando_procurar_atendimento)]
        L += [""]

        L += ["## 11. Referências utilizadas"]
        if analise.referencias_documentais:
            L.append("**Apoio documental da análise (Agente 2):**")
            L.extend(f"- {r}" for r in analise.referencias_documentais)
        if orient.referencias:
            for r in orient.referencias:
                pg = f", p. {r.pagina}" if r.pagina else ""
                L.append(f"{r.numero}. {r.fonte}{pg} — “{r.trecho.strip()}…”")
        else:
            L.append("Nenhum documento foi citado (orientações fixas e de segurança).")
        if self.retriever and not estado.interrompida:
            try:
                trechos = self.retriever.buscar_varias(
                    ["ficha de notificação dengue SINAN dados de identificação e notificação",
                     "instruções de preenchimento da ficha SINAN dengue",
                     "definição e classificação dos casos de dengue"],
                    k_total=6, agente=4)
                if trechos:
                    L.append("**Apoio documental para registro (Agente 4; não é uma notificação oficial):**")
                    for t in trechos:
                        L.append(f"- {t.fonte}, p. {t.pagina}: {t.conteudo[:180]}")
            except Exception:
                pass
        L += [""]

        sem_resp, nao_perg = [], []
        for campo in todos_campos_perguntados():
            if campo.endswith("_detalhe") or campo in ("exames.hematocrito_pct", "exames.plaquetas"):
                continue
            if d.resolvido(campo):
                continue
            (sem_resp if campo in d.campos_sem_resposta else nao_perg).append(ROTULOS[campo])
        L += [
            "## 12. Informações não fornecidas e que dependem de avaliação profissional",
            "**Perguntado, sem resposta clara `[Paciente]`:** " + ("; ".join(sem_resp) or "nenhum") + ".",
            "",
            "**Não perguntado (fluxo interrompido/encerrado antes) `[Sistema]`:** " + ("; ".join(nao_perg) or "nenhum") + ".",
            "",
            "**Dependem de exame físico ou avaliação profissional `[Profissional]`:** hipotensão postural, aumento do fígado, "
            "acúmulo de líquidos, prova do laço/petéquias, variação do hematócrito, risco social, diagnóstico e conduta.",
            "",
            "---",
            f"*{AVISO}*",
        ]
        if estado.inconsistencias:
            pos = L.index("---")
            L[pos:pos] = ["**Relatos que precisam de esclarecimento `[Paciente]`:**",
                          *[f"- {item}" for item in estado.inconsistencias], ""]
        return "\n".join(L)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _itens(itens) -> list[str]:
        out = []
        for it in itens:
            ref = f" _(ref. {', '.join(map(str, it.refs))})_" if it.refs else ""
            out.append(f"- {it.texto}{ref}")
        return out

    @staticmethod
    def _fmt(d, campo: str, estado: ColetaEstado) -> str:
        v = d.obter(campo)
        if isinstance(v, Resposta):
            if v == Resposta.NAO_INFORMADO:
                pergunta_feita = any(
                    campo in p.campos + p.extras and p.id in estado.perguntadas for p in PERGUNTAS
                )
                return "Não informado" if pergunta_feita else "Não perguntado"
            return _RESP[v]
        if v is None:
            if campo.endswith("_detalhe") or campo in ("exames.hematocrito_pct", "exames.plaquetas"):
                return "—"
            pergunta_feita = any(campo in p.campos + p.extras and p.id in estado.perguntadas for p in PERGUNTAS)
            return "Não informado" if pergunta_feita else "Não perguntado"
        return _VALORES.get(v, str(v))
