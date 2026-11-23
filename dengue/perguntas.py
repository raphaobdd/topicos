"""Roteiro de perguntas do Agente 1 (seções A–E da especificação).

Cada pergunta cobre um ou mais campos. Perguntas agrupadas (ex.: sinais de alerta)
mantêm UM campo por sinal no objeto estruturado; só a conversa fica mais curta.
"""
from __future__ import annotations

import types
import typing
from dataclasses import dataclass
from datetime import date
from typing import Callable, Optional

from .schemas import DadosPaciente, Resposta, GRUPOS_DADOS

SEC_A = "identificação e informações gerais"
SEC_B = "início e evolução dos sintomas"
SEC_C = "sintomas mais comuns"
SEC_D = "sinais de alerta e exames"
SEC_E = "histórico e fatores de risco"


def _sempre(_d: DadosPaciente) -> bool:
    return True


def _teve_febre(d: DadosPaciente) -> bool:
    return d.evolucao.febre == Resposta.SIM


def _gravidez_aplicavel(d: DadosPaciente) -> bool:
    i = d.identificacao
    return i.sexo != "M" and (i.idade is None or 10 <= i.idade <= 60)


@dataclass(frozen=True)
class Pergunta:
    id: str
    secao: str
    texto: str
    campos: tuple  # campos que a pergunta precisa resolver
    extras: tuple = ()  # campos de detalhe preenchidos "se der"
    aplicavel: Callable[[DadosPaciente], bool] = _sempre


PERGUNTAS: list[Pergunta] = [
    # A. Identificação e informações gerais
    Pergunta("idade", SEC_A, "Qual é a sua idade?", ("identificacao.idade",)),
    Pergunta(
        "sexo", SEC_A,
        "Qual é o seu sexo (feminino ou masculino)? Pergunto porque algumas condições, como a gravidez, mudam a avaliação.",
        ("identificacao.sexo",),
    ),
    Pergunta("peso", SEC_A, "Qual é o seu peso aproximado, em kg?", ("identificacao.peso_kg",)),
    Pergunta(
        "gravidez", SEC_A, "Você está grávida ou existe possibilidade de gravidez?",
        ("identificacao.gravidez",), aplicavel=_gravidez_aplicavel,
    ),
    Pergunta(
        "doenca_pre", SEC_A,
        "Você possui alguma doença ou condição de saúde preexistente? Se sim, qual?",
        ("historico.doenca_preexistente",), ("historico.doenca_preexistente_detalhe",),
    ),
    Pergunta(
        "medicamento", SEC_A, "Faz uso contínuo de algum medicamento? Se sim, qual?",
        ("historico.uso_continuo_medicamento",), ("historico.medicamento_detalhe",),
    ),
    Pergunta(
        "alergia", SEC_A, "Possui alguma alergia conhecida? Se sim, a quê?",
        ("historico.alergia",), ("historico.alergia_detalhe",),
    ),
    # B. Início e evolução
    Pergunta(
        "dias", SEC_B,
        "Quando começaram os primeiros sintomas? (Pode me dizer há quantos dias você está com sintomas.)",
        ("evolucao.dias_sintomas",),
    ),
    Pergunta("febre", SEC_B, "Você teve febre?", ("evolucao.febre",)),
    Pergunta(
        "febre_max", SEC_B,
        "Sabe informar qual foi a temperatura mais alta que você chegou a medir?",
        ("evolucao.febre_max_c",), aplicavel=_teve_febre,
    ),
    Pergunta(
        "febre_continua", SEC_B, "A febre continua ou já diminuiu/passou?",
        ("evolucao.febre_continua",), aplicavel=_teve_febre,
    ),
    Pergunta(
        "tendencia", SEC_B,
        "De modo geral, os sintomas estão melhorando, piorando ou permanecem iguais?",
        ("evolucao.tendencia",),
    ),
    # C. Sintomas comuns (agrupados; itens não citados em resposta clara = "não")
    Pergunta(
        "sint_dor", SEC_C,
        "Você está com algum destes? Dor de cabeça, dor atrás dos olhos, dor muscular, dor nas articulações. (Pode listar só os que tem.)",
        ("sintomas.dor_cabeca", "sintomas.dor_atras_olhos", "sintomas.dor_muscular", "sintomas.dor_articulacoes"),
    ),
    Pergunta(
        "sint_dig", SEC_C,
        "E estes? Náuseas/enjoo, vômitos, manchas vermelhas na pele ou erupções.",
        ("sintomas.nausea", "sintomas.vomitos", "sintomas.manchas_pele"),
    ),
    Pergunta(
        "sint_geral", SEC_C, "E cansaço/prostração ou falta de apetite?",
        ("sintomas.cansaco_prostracao", "sintomas.falta_apetite"),
    ),
    # D. Sinais de alerta (etapa mais importante)
    Pergunta(
        "alerta_1", SEC_D,
        "Agora, sobre sinais de alerta — são importantes. Você está sentindo: dor abdominal (na barriga) intensa e contínua? Vômitos persistentes (que não param)? Inchaço ou sensação de líquido acumulado no abdômen?",
        ("sinais_alerta.dor_abdominal_intensa", "sinais_alerta.vomitos_persistentes", "sinais_alerta.inchaco_abdominal"),
    ),
    Pergunta(
        "alerta_2", SEC_D,
        "Percebeu algum sangramento no nariz ou na gengiva? E sangue nas fezes ou qualquer outro sangramento?",
        ("sinais_alerta.sangramento_mucosa", "sinais_alerta.sangue_fezes_outro_sangramento"),
    ),
    Pergunta(
        "alerta_3", SEC_D,
        "Sentiu tontura ao levantar ou sensação de desmaio? Sonolência excessiva, dificuldade de ficar alerta ou irritabilidade incomum? Dificuldade para respirar?",
        ("sinais_alerta.tontura_desmaio", "sinais_alerta.sonolencia_irritabilidade", "sinais_alerta.dificuldade_respirar"),
    ),
    Pergunta(
        "exames", SEC_D,
        "Você fez algum exame de sangue (hemograma) nesta doença? Se tiver o resultado, me diga o hematócrito e as plaquetas. Se não tiver, tudo bem.",
        ("exames.hemograma_disponivel",), ("exames.hematocrito_pct", "exames.plaquetas"),
    ),
    # E. Histórico e fatores de risco
    Pergunta("dengue_antes", SEC_E, "Você já teve dengue antes?", ("historico.teve_dengue_antes",)),
    Pergunta(
        "hist_1", SEC_E, "Você tem hipertensão ou diabetes?",
        ("historico.hipertensao", "historico.diabetes"),
    ),
    Pergunta(
        "hist_2", SEC_E, "Tem alguma doença do coração, dos rins, do pulmão ou do fígado?",
        ("historico.doenca_cardiaca", "historico.doenca_renal", "historico.doenca_pulmonar", "historico.doenca_hepatica"),
    ),
    Pergunta(
        "hist_3", SEC_E,
        "Tem alguma doença do sangue ou autoimune? Usa anticoagulantes (remédios que afinam o sangue)?",
        ("historico.doenca_hematologica", "historico.doenca_autoimune", "historico.anticoagulantes"),
    ),
    Pergunta(
        "hist_4", SEC_E, "Existe alguma outra condição de saúde que você considere importante informar?",
        ("historico.outra_condicao",), ("historico.outra_condicao_detalhe",),
    ),
]

PERGUNTA_POR_ID = {p.id: p for p in PERGUNTAS}

# --------------------------------------------------------------------------- #
# Rótulos legíveis (relatório, mensagens e catálogo do extrator)
# --------------------------------------------------------------------------- #
ROTULOS = {
    "identificacao.idade": "Idade (anos)",
    "identificacao.sexo": "Sexo",
    "identificacao.peso_kg": "Peso (kg)",
    "identificacao.gravidez": "Gravidez ou possibilidade de gravidez",
    "evolucao.dias_sintomas": "Dias de sintomas",
    "evolucao.data_inicio": "Data aproximada de início",
    "evolucao.febre": "Teve febre",
    "evolucao.febre_max_c": "Temperatura mais alta (°C)",
    "evolucao.febre_continua": "Ainda está com febre",
    "evolucao.tendencia": "Evolução dos sintomas",
    "sintomas.dor_cabeca": "Dor de cabeça",
    "sintomas.dor_atras_olhos": "Dor atrás dos olhos",
    "sintomas.dor_muscular": "Dor muscular",
    "sintomas.dor_articulacoes": "Dor nas articulações",
    "sintomas.nausea": "Náuseas ou enjoo",
    "sintomas.vomitos": "Vômitos",
    "sintomas.manchas_pele": "Manchas vermelhas ou erupções na pele",
    "sintomas.cansaco_prostracao": "Cansaço ou prostração",
    "sintomas.falta_apetite": "Falta de apetite",
    "sinais_alerta.dor_abdominal_intensa": "Dor abdominal intensa e contínua",
    "sinais_alerta.vomitos_persistentes": "Vômitos persistentes",
    "sinais_alerta.sangramento_mucosa": "Sangramento no nariz ou na gengiva",
    "sinais_alerta.sangue_fezes_outro_sangramento": "Sangue nas fezes ou outro sangramento",
    "sinais_alerta.tontura_desmaio": "Tontura ao levantar ou sensação de desmaio",
    "sinais_alerta.sonolencia_irritabilidade": "Sonolência excessiva, dificuldade de se manter alerta ou irritabilidade incomum",
    "sinais_alerta.dificuldade_respirar": "Dificuldade para respirar",
    "sinais_alerta.inchaco_abdominal": "Inchaço ou sensação de líquido acumulado no abdômen",
    "historico.doenca_preexistente": "Doença ou condição preexistente",
    "historico.doenca_preexistente_detalhe": "Qual doença/condição",
    "historico.uso_continuo_medicamento": "Uso contínuo de medicamento",
    "historico.medicamento_detalhe": "Qual medicamento",
    "historico.alergia": "Alergia conhecida",
    "historico.alergia_detalhe": "Alergia a quê",
    "historico.teve_dengue_antes": "Já teve dengue",
    "historico.hipertensao": "Hipertensão",
    "historico.diabetes": "Diabetes",
    "historico.doenca_cardiaca": "Doença cardíaca",
    "historico.doenca_renal": "Doença renal",
    "historico.doenca_pulmonar": "Doença pulmonar",
    "historico.doenca_hepatica": "Doença hepática",
    "historico.doenca_hematologica": "Doença hematológica",
    "historico.doenca_autoimune": "Doença autoimune",
    "historico.anticoagulantes": "Uso de anticoagulantes",
    "historico.outra_condicao": "Outra condição relevante",
    "historico.outra_condicao_detalhe": "Qual outra condição",
    "exames.hemograma_disponivel": "Tem resultado de hemograma",
    "exames.hematocrito_pct": "Hematócrito (%)",
    "exames.plaquetas": "Plaquetas (/mm³)",
}

DICAS = {
    "evolucao.dias_sintomas": "número de dias desde o início dos primeiros sintomas",
    "evolucao.febre_continua": '"sim" se ainda está com febre; "nao" se a febre já passou',
    "evolucao.tendencia": 'estado geral dos sintomas',
    "sinais_alerta.vomitos_persistentes": "vômitos repetidos, que não param",
    "sinais_alerta.dor_abdominal_intensa": "dor na barriga forte e que não passa",
    "exames.hemograma_disponivel": '"sim" se tem resultado em mãos',
    "exames.plaquetas": "contagem por mm³, ex.: 100000 (se disser '100 mil', use 100000)",
}


def _tipo(ann) -> str:
    args = typing.get_args(ann)
    origem = typing.get_origin(ann)
    if ann is Resposta:
        return '"sim" ou "nao"'
    if origem is typing.Literal:
        return " ou ".join(f'"{a}"' for a in args)
    if origem is typing.Union or origem is types.UnionType:
        for a in args:
            if a is not type(None):
                return _tipo(a)
    if ann is int:
        return "número inteiro"
    if ann is float:
        return "número (ponto decimal)"
    if ann is date:
        return "data AAAA-MM-DD"
    return "texto curto"


def descrever_campo(caminho: str) -> str:
    """Linha de catálogo para o extrator: 'caminho: tipo — rótulo (dica)'."""
    grupo, campo = caminho.split(".", 1)
    modelo = DadosPaciente.model_fields[grupo].annotation
    ann = modelo.model_fields[campo].annotation
    linha = f"- {caminho}: {_tipo(ann)} — {ROTULOS.get(caminho, campo)}"
    if caminho in DICAS:
        linha += f" ({DICAS[caminho]})"
    return linha


def todos_campos_perguntados() -> list[str]:
    vistos: list[str] = []
    for p in PERGUNTAS:
        for c in p.campos + p.extras:
            if c not in vistos:
                vistos.append(c)
    return vistos


# sanity check em import: todo campo das perguntas existe no schema
for _c in todos_campos_perguntados():
    _g, _f = _c.split(".", 1)
    assert _g in GRUPOS_DADOS and _c in ROTULOS, f"campo inconsistente: {_c}"
