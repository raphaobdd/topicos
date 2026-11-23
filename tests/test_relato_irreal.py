from dengue.agents.coleta import AgenteColeta
from dengue.perguntas import PERGUNTAS
from dengue.schemas import ColetaEstado
from tests.conftest import FakeLLM


def _pergunta(campo):
    return next(p for p in PERGUNTAS if campo in p.campos)


def test_metafora_nao_confirma_acumulo_de_liquido():
    coleta = AgenteColeta(FakeLLM())
    estado = ColetaEstado()
    atual = _pergunta("sinais_alerta.inchaco_abdominal")
    extraido, aviso = coleta._validar_extracao(
        estado, "parece que tem um boi no meu abdômen",
        atual, {"campos": {"sinais_alerta.inchaco_abdominal": "sim"}})
    assert "sinais_alerta.inchaco_abdominal" not in extraido["campos"]
    assert "procure atendimento" in aviso.lower()


def test_febre_36_e_duracao_de_anos_ficam_pendentes():
    coleta = AgenteColeta(FakeLLM())
    estado = ColetaEstado()
    extraido, _ = coleta._validar_extracao(
        estado, "tive febre de 36 graus", _pergunta("evolucao.febre"),
        {"campos": {"evolucao.febre": "sim", "evolucao.febre_max_c": 36}})
    assert "evolucao.febre" not in extraido["campos"]
    extraido, _ = coleta._validar_extracao(
        estado, "2190 dias", _pergunta("evolucao.dias_sintomas"),
        {"campos": {"evolucao.dias_sintomas": 2190}})
    assert "evolucao.dias_sintomas" not in extraido["campos"]
    assert len(estado.inconsistencias) == 2


def test_sintomas_nao_mencionados_nao_viram_ausentes():
    coleta = AgenteColeta(FakeLLM())
    estado = ColetaEstado()
    atual = _pergunta("sintomas.dor_cabeca")
    extraido, _ = coleta._validar_extracao(
        estado, "minha uretra dói", atual,
        {"campos": {campo: "nao" for campo in atual.campos}})
    assert not extraido["campos"]
