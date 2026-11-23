from dengue.safety import ALERTAS, classificar_manejo, mensagem_urgente, verificar_seguranca
from dengue.schemas import DadosPaciente, Grupo
from tests.conftest import dados_com


def test_sem_alertas_grupo_a():
    d = dados_com()
    seg = verificar_seguranca(d)
    assert not seg.interromper and not seg.nao_respondidos
    assert classificar_manejo(d, seg).grupo == Grupo.A


def test_cada_sinal_de_alerta_interrompe():
    for campo in ALERTAS:
        d = dados_com(**{campo.replace(".", "__"): "sim"})
        seg = verificar_seguranca(d)
        assert seg.interromper, campo
        assert classificar_manejo(d, seg).grupo in (Grupo.C, Grupo.D), campo


def test_vomitos_persistentes_e_grupo_c():
    d = dados_com(sinais_alerta__vomitos_persistentes="sim")
    assert classificar_manejo(d).grupo == Grupo.C


def test_dificuldade_respirar_grupo_d_e_samu():
    d = dados_com(sinais_alerta__dificuldade_respirar="sim")
    seg = verificar_seguranca(d)
    assert seg.possivel_gravidade
    assert classificar_manejo(d, seg).grupo == Grupo.D
    assert "192" in mensagem_urgente(seg)


def test_comorbidade_idade_e_gestacao_grupo_b():
    assert classificar_manejo(dados_com(historico__hipertensao="sim")).grupo == Grupo.B
    assert classificar_manejo(dados_com(identificacao__gravidez="sim")).grupo == Grupo.B
    d = dados_com()
    d.definir("identificacao.idade", 70)
    assert classificar_manejo(d).grupo == Grupo.B
    d.definir("identificacao.idade", 1)
    assert classificar_manejo(d).grupo == Grupo.B


def test_manchas_na_pele_exigem_avaliacao_grupo_b():
    assert classificar_manejo(dados_com(sintomas__manchas_pele="sim")).grupo == Grupo.B


def test_dados_incompletos_nao_viram_grupo_a():
    d = dados_com()
    d.sinais_alerta.vomitos_persistentes = DadosPaciente().sinais_alerta.vomitos_persistentes  # nao informado
    seg = verificar_seguranca(d)
    assert seg.nao_respondidos
    c = classificar_manejo(d, seg)
    assert c.grupo == Grupo.B and c.dados_incompletos


def test_defervescencia_gera_atencao():
    d = dados_com(evolucao__dias_sintomas=4, evolucao__febre="sim", evolucao__febre_continua="nao")
    c = classificar_manejo(d)
    assert any("febre passou" in a for a in c.atencoes)
