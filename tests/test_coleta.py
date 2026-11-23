from dengue.orchestrator import Orquestrador
from dengue.perguntas import PERGUNTAS
from dengue.schemas import Resposta
from tests.conftest import BASE, FakeLLM


def conversar(orq, estado, max_turnos=60):
    n = 0
    while not estado.concluida and n < max_turnos:
        orq.turno(estado, "resposta do paciente")
        n += 1
    return n


def test_fluxo_completo_estrutura_dados():
    orq = Orquestrador(FakeLLM(BASE))
    estado = orq.nova_sessao()
    primeira = orq.iniciar(estado)
    assert "Qual é a sua idade" in primeira
    conversar(orq, estado)
    assert estado.concluida and not estado.interrompida
    d = estado.dados
    assert d.identificacao.idade == 30 and d.evolucao.febre == Resposta.SIM
    assert d.sintomas.dor_cabeca == Resposta.SIM
    assert d.sintomas.vomitos == Resposta.NAO          # lista respondida: item não citado = não
    assert d.evolucao.data_inicio is not None           # derivada de dias_sintomas
    assert d.sinais_alerta.dificuldade_respirar == Resposta.NAO


def test_sem_febre_pula_perguntas_dependentes():
    paciente = {**BASE, "evolucao.febre": "nao"}
    orq = Orquestrador(FakeLLM(paciente))
    estado = orq.nova_sessao(); orq.iniciar(estado)
    conversar(orq, estado)
    assert "febre_max" not in estado.perguntadas and "febre_continua" not in estado.perguntadas
    assert estado.dados.evolucao.febre_continua == Resposta.NAO_INFORMADO


def test_homem_nao_pergunta_gravidez():
    paciente = {**BASE, "identificacao.sexo": "M"}
    orq = Orquestrador(FakeLLM(paciente))
    estado = orq.nova_sessao(); orq.iniciar(estado)
    conversar(orq, estado)
    assert "gravidez" not in estado.perguntadas
    assert estado.dados.identificacao.gravidez == Resposta.NAO


def test_resposta_incompreendida_repete_uma_vez_e_registra_sem_resposta():
    orq = Orquestrador(FakeLLM(BASE, falhar_extracao=True))
    estado = orq.nova_sessao(); orq.iniciar(estado)
    m1 = orq.turno(estado, "hmm").mensagem
    assert "Não consegui identificar" in m1 and estado.atual == "idade"
    orq.turno(estado, "hmm")                                   # 2ª tentativa -> segue em frente
    assert "identificacao.idade" in estado.dados.campos_sem_resposta
    assert estado.atual == "sexo"
    assert estado.dados.identificacao.idade is None            # desconhecido, não inventado


def test_erro_do_llm_nao_derruba_conversa():
    class Quebrado(FakeLLM):
        def json(self, *a, **k):
            raise RuntimeError("API fora")
    orq = Orquestrador(Quebrado())
    estado = orq.nova_sessao(); orq.iniciar(estado)
    msg = orq.turno(estado, "30 anos").mensagem
    assert "problema técnico" in msg and not estado.concluida


def test_todos_os_campos_das_perguntas_existem():
    from dengue.schemas import DadosPaciente
    d = DadosPaciente()
    for p in PERGUNTAS:
        for c in p.campos + p.extras:
            d.obter(c)
