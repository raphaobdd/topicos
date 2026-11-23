from dengue.schemas import DadosPaciente, Resposta
from tests.conftest import dados_com


def test_padrao_e_nao_informado_nunca_nao():
    d = DadosPaciente()
    assert d.sintomas.vomitos == Resposta.NAO_INFORMADO
    assert d.sinais_alerta.dificuldade_respirar == Resposta.NAO_INFORMADO
    assert not d.resolvido("sintomas.vomitos")


def test_definir_normaliza_acentos_e_booleanos():
    d = DadosPaciente()
    assert d.definir("sintomas.vomitos", "Não")
    assert d.sintomas.vomitos == Resposta.NAO
    assert d.definir("sintomas.nausea", True)
    assert d.sintomas.nausea == Resposta.SIM
    assert d.definir("identificacao.idade", "34")
    assert d.identificacao.idade == 34


def test_definir_rejeita_invalido_e_nao_sobrescreve_com_vazio():
    d = DadosPaciente()
    assert not d.definir("identificacao.idade", 300)          # fora do intervalo
    assert not d.definir("evolucao.febre_max_c", 20)           # implausível
    assert not d.definir("campo.inexistente", "sim")
    assert not d.definir("sintomas.vomitos", "talvez")
    d.definir("sintomas.vomitos", "sim")
    assert not d.definir("sintomas.vomitos", "nao_informado")  # não apaga informação
    assert d.sintomas.vomitos == Resposta.SIM


def test_dados_com_helper():
    d = dados_com(sinais_alerta__dificuldade_respirar="sim")
    assert d.sinais_alerta.dificuldade_respirar == Resposta.SIM
    assert d.sintomas.vomitos == Resposta.NAO
