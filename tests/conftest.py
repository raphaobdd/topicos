"""Fixtures: um LLM falso (sem rede) e um paciente simulado que responde ao extrator."""
from __future__ import annotations

import hashlib
import math
import re

import pytest

from dengue.schemas import DadosPaciente, Resposta


def embedding_hash(texto: str, dim: int = 64) -> list[float]:
    """Embedding lexical determinístico (bag-of-words em hash) — suficiente p/ testar o RAG."""
    v = [0.0] * dim
    for palavra in re.findall(r"\w+", texto.lower()):
        h = int(hashlib.md5(palavra.encode()).hexdigest(), 16)
        v[h % dim] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class FakeLLM:
    """paciente: dict {campo: valor}. Para a pergunta atual, devolve os campos que o paciente
    'sabe'; campos de lista não citados viram `padrao` ('nao') se `padrao` estiver definido."""

    def __init__(self, paciente=None, padrao="nao", orientacao=None, texto="Resumo de teste.", falhar_extracao=False):
        self.paciente = paciente or {}
        self.padrao = padrao
        self.orientacao = orientacao
        self.texto_fixo = texto
        self.falhar_extracao = falhar_extracao
        self.sistemas_chamados: list[str] = []

    def json(self, sistema, prompt, temperatura=0.0):
        self.sistemas_chamados.append(sistema[:40])
        if "EXTRATOR" in sistema:
            if self.falhar_extracao:
                return {"campos": {}, "nao_sabe": []}
            linha = re.search(r"CAMPOS DA PERGUNTA ATUAL: (.*)", prompt).group(1)
            campos = [c.strip() for c in linha.split(",") if c.strip()]
            saida = {}
            for c in campos:
                if c in self.paciente:
                    saida[c] = self.paciente[c]
                elif self.padrao is not None and c.startswith(("sintomas.", "sinais_alerta.", "historico.", "exames.hemograma")):
                    saida[c] = self.padrao
            return {"campos": saida, "nao_sabe": [], "mensagem": ""}
        if "ORIENTAÇÕES" in sistema:
            return self.orientacao or {
                "orientacoes": [{"texto": "Mantenha hidratação conforme orientação do serviço de saúde.", "refs": [1]}],
                "quando_procurar_atendimento": [{"texto": "Procure atendimento se surgir dor abdominal intensa.", "refs": [1]}],
                "contexto_insuficiente": False,
            }
        raise AssertionError(f"sistema inesperado: {sistema[:60]}")

    def texto(self, sistema, prompt, temperatura=0.2):
        self.sistemas_chamados.append(sistema[:40])
        return self.texto_fixo

    def embed(self, textos, consulta=False):
        return [embedding_hash(t) for t in textos]


# Paciente "típico" que responde todas as perguntas; sobrescreva campos por teste.
BASE = {
    "identificacao.idade": 30,
    "identificacao.sexo": "F",
    "identificacao.peso_kg": 65,
    "identificacao.gravidez": "nao",
    "historico.doenca_preexistente": "nao",
    "historico.uso_continuo_medicamento": "nao",
    "historico.alergia": "nao",
    "evolucao.dias_sintomas": 3,
    "evolucao.febre": "sim",
    "evolucao.febre_max_c": 39.2,
    "evolucao.febre_continua": "sim",
    "evolucao.tendencia": "igual",
    "sintomas.dor_cabeca": "sim",
    "sintomas.dor_muscular": "sim",
    "historico.teve_dengue_antes": "nao",
}


@pytest.fixture
def paciente_base():
    return dict(BASE)


@pytest.fixture
def fake_llm_factory():
    return FakeLLM


def dados_com(**campos) -> DadosPaciente:
    """Monta DadosPaciente preenchido: todos os campos Resposta = 'nao', salvo os informados
    (use '__' no lugar de '.': sinais_alerta__dificuldade_respirar='sim')."""
    d = DadosPaciente()
    for c in d.caminhos():
        if isinstance(d.obter(c), Resposta):
            d.definir(c, "nao")
    d.definir("identificacao.idade", 30)
    d.definir("identificacao.sexo", "F")
    for k, v in campos.items():
        assert d.definir(k.replace("__", "."), v), f"não aplicou {k}={v}"
    return d
