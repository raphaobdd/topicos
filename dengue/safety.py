"""Regras explícitas (determinísticas) de segurança e classificação.

IMPORTANTE: a detecção de sinais de alerta NUNCA depende do LLM. O LLM só extrai
as respostas; quem decide interromper o fluxo são estas regras, baseadas nos
sinais de alarme/gravidade do protocolo de manejo clínico da dengue (Ministério
da Saúde). Revise as listas abaixo com o protocolo oficial e com seu orientador
antes de qualquer uso além do acadêmico.
"""
from __future__ import annotations

from .perguntas import ROTULOS
from .schemas import (
    ClassificacaoRegras,
    DadosPaciente,
    Grupo,
    Resposta,
    ResultadoSeguranca,
)

# Os 8 sinais investigados pelo Agente 1 (seção D da especificação)
ALERTAS = (
    "sinais_alerta.dor_abdominal_intensa",
    "sinais_alerta.vomitos_persistentes",
    "sinais_alerta.sangramento_mucosa",
    "sinais_alerta.sangue_fezes_outro_sangramento",
    "sinais_alerta.tontura_desmaio",
    "sinais_alerta.sonolencia_irritabilidade",
    "sinais_alerta.dificuldade_respirar",
    "sinais_alerta.inchaco_abdominal",
)

# Relatos que sugerem possível gravidade (-> "Grupo D?", orientar SAMU 192).
# Escolha conservadora: o conjunto é pequeno e fácil de ajustar.
POSSIVEL_GRAVIDADE = frozenset(
    {
        "sinais_alerta.dificuldade_respirar",
        "sinais_alerta.sangue_fezes_outro_sangramento",
    }
)

COMORBIDADES = (
    "historico.doenca_preexistente",
    "historico.hipertensao",
    "historico.diabetes",
    "historico.doenca_cardiaca",
    "historico.doenca_renal",
    "historico.doenca_pulmonar",
    "historico.doenca_hepatica",
    "historico.doenca_hematologica",
    "historico.doenca_autoimune",
    "historico.anticoagulantes",
)

SIM = Resposta.SIM
NAO = Resposta.NAO


def verificar_seguranca(dados: DadosPaciente) -> ResultadoSeguranca:
    presentes = [c for c in ALERTAS if dados.obter(c) == SIM]
    return ResultadoSeguranca(
        campos_presentes=presentes,
        sinais_presentes=[ROTULOS[c] for c in presentes],
        possivel_gravidade=any(c in POSSIVEL_GRAVIDADE for c in presentes),
        nao_respondidos=[ROTULOS[c] for c in ALERTAS if not dados.resolvido(c)],
        interromper=bool(presentes),
    )


def mensagem_urgente(seg: ResultadoSeguranca) -> str:
    itens = "\n".join(f"- {s}" for s in seg.sinais_presentes)
    samu = (
        "\n\n**Ligue 192 (SAMU) agora** ou peça para alguém levar você com urgência — "
        "o que você descreveu pode indicar uma situação grave."
        if seg.possivel_gravidade
        else "\n\nSe tiver dificuldade para respirar, desmaio, sangramento importante, "
        "muita sonolência ou confusão, ligue **192 (SAMU)**."
    )
    return (
        "⚠️ **Procure atendimento médico agora.**\n\n"
        "Pelo que você me contou, há sinal(is) de alerta que precisam de avaliação presencial imediata:\n"
        f"{itens}\n\n"
        "Vá a uma UPA, pronto-socorro ou hospital o quanto antes — não espere os sintomas passarem. "
        "Se possível, vá acompanhado(a)."
        f"{samu}\n\n"
        "Vou encerrar a avaliação por aqui e gerar um relatório com o que você informou, "
        "que você pode levar ao atendimento. Esta é uma ferramenta experimental e não substitui avaliação médica."
    )


def _precisa_gravidez(d: DadosPaciente) -> bool:
    i = d.identificacao
    return i.sexo != "M" and (i.idade is None or 10 <= i.idade <= 60)


def classificar_manejo(
    dados: DadosPaciente, seg: ResultadoSeguranca | None = None
) -> ClassificacaoRegras:
    """Classificação experimental A/B/C/D por regras. Não substitui avaliação profissional."""
    seg = seg or verificar_seguranca(dados)
    ident, ev, hist = dados.identificacao, dados.evolucao, dados.historico

    just: list[str] = []
    atenc: list[str] = []

    criticos_para_a = list(ALERTAS) + ["identificacao.idade", "evolucao.febre"] + list(COMORBIDADES)
    if _precisa_gravidez(dados):
        criticos_para_a.append("identificacao.gravidez")
    faltantes = [c for c in criticos_para_a if not dados.resolvido(c)]
    incompletos = [ROTULOS[c] for c in faltantes]

    # --- grupo ------------------------------------------------------------- #
    if seg.possivel_gravidade:
        grupo = Grupo.D
        just.append(
            "Relato compatível com possível sinal de gravidade: "
            + ", ".join(s.lower() for s in seg.sinais_presentes)
            + "."
        )
    elif seg.sinais_presentes:
        grupo = Grupo.C
        just.append("Sinal(is) de alarme relatado(s): " + ", ".join(s.lower() for s in seg.sinais_presentes) + ".")
    else:
        especiais: list[str] = []
        if ident.idade is not None and ident.idade < 2:
            especiais.append("criança menor de 2 anos")
        if ident.idade is not None and ident.idade >= 65:
            especiais.append("idade igual ou superior a 65 anos")
        if ident.gravidez == SIM:
            especiais.append("gestação")
        for c in COMORBIDADES:
            if dados.obter(c) == SIM:
                especiais.append(ROTULOS[c].lower())
        if dados.sintomas.manchas_pele == SIM:
            especiais.append("manchas na pele relatadas (é preciso avaliar petéquias, o que exige exame)")

        if especiais:
            grupo = Grupo.B
            just.append("Sem sinais de alarme relatados, mas com: " + "; ".join(especiais) + ".")
        elif faltantes:
            grupo = Grupo.B
            just.append(
                "Sem sinais de alarme relatados, porém há informações essenciais sem resposta "
                "(" + ", ".join(incompletos[:6]) + ("…" if len(incompletos) > 6 else "") + "). "
                "Por precaução, o caso não foi classificado como Grupo A."
            )
        else:
            grupo = Grupo.A
            just.append(
                "Sem sinais de alarme, condições especiais ou comorbidades informados "
                "(a ausência de relato não confirma ausência de risco)."
            )

    # --- atenções ---------------------------------------------------------- #
    if ev.febre == SIM and ev.febre_continua == NAO and ev.dias_sintomas is not None and 3 <= ev.dias_sintomas <= 7:
        atenc.append(
            "A febre passou entre o 3º e o 7º dia de sintomas: é a fase em que sinais de alarme "
            "podem surgir. Atenção redobrada, mesmo se sentir melhora."
        )
    if ev.dias_sintomas is not None and ev.dias_sintomas > 7:
        atenc.append("Sintomas há mais de 7 dias: recomenda-se avaliação presencial.")
    if ev.tendencia == "piorando":
        atenc.append("Os sintomas foram descritos como piorando.")
    if hist.teve_dengue_antes == SIM:
        atenc.append("Relato de dengue anterior: nova infecção merece acompanhamento atento.")
    if dados.exames.hemograma_disponivel != SIM:
        atenc.append("Sem resultado de hemograma/hematócrito: esses exames dependem de serviço de saúde.")
    atenc.append(
        "Risco social (ex.: dificuldade de acesso ao serviço de saúde, morar sozinho) não foi avaliado e pode alterar o grupo."
    )
    atenc.append(
        "Hipotensão postural, aumento do fígado, acúmulo de líquidos e variação do hematócrito "
        "dependem de exame físico/laboratorial; esta conversa não os substitui."
    )

    return ClassificacaoRegras(
        grupo=grupo, justificativas=just, atencoes=atenc, dados_incompletos=incompletos
    )
