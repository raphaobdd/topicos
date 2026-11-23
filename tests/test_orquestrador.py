from dengue.agents.orientacoes import AgenteOrientacoes
from dengue.orchestrator import Orquestrador
from dengue.rag.retriever import Retriever
from dengue.rag.store import LocalStore
from dengue.schemas import Grupo
from tests.conftest import BASE, FakeLLM, embedding_hash


def retriever_com_docs(tmp_path, llm):
    store = LocalStore(tmp_path / "v.json")
    docs = [
        ("fluxograma_manejo_dengue_2025", 1, "[[capitulo:integral]] Sinais de alarme da dengue incluem dor abdominal intensa e contínua, vômitos persistentes e sangramento de mucosa. Procure atendimento."),
        ("manual_enfermagem_dengue_2026", 20, "[[capitulo:integral]] Grupo A: hidratação oral e retorno se surgirem sinais de alarme. Evitar anti-inflamatórios e ácido acetilsalicílico."),
    ]
    store.adicionar([{"fonte": f, "pagina": p, "conteudo": c, "embedding": embedding_hash(c)} for f, p, c in docs])
    return Retriever(llm, store, k=4)


def conversar(orq, estado):
    n = 0
    while not estado.concluida and n < 60:
        t = orq.turno(estado, "ok"); n += 1
    return t


def test_fluxo_normal_gera_resultado_e_relatorio_completo(tmp_path):
    llm = FakeLLM(BASE)
    orq = Orquestrador(llm, retriever=retriever_com_docs(tmp_path, llm))
    estado = orq.nova_sessao(); orq.iniciar(estado)
    t = conversar(orq, estado)
    assert t.etapa == "coleta_concluida"
    r = orq.executar(estado)
    assert not r.interrompido and r.analise.classificacao.grupo == Grupo.A
    assert r.orientacoes.orientacoes and r.orientacoes.referencias
    assert r.orientacoes.referencias[0].fonte in {"fluxograma_manejo_dengue_2025", "manual_enfermagem_dengue_2026"}
    for n in range(1, 13):
        assert f"## {n}. " in r.relatorio_md, f"seção {n} ausente"
    assert "Grupo A" in r.relatorio_md and "Não perguntado" not in r.relatorio_md.split("## 12.")[0]
    assert "| Sexo | Feminino |" in r.relatorio_md


def test_sinal_de_alerta_interrompe_fluxo_e_nao_chama_agente_de_orientacoes(tmp_path):
    paciente = {**BASE, "sinais_alerta.vomitos_persistentes": "sim"}
    llm = FakeLLM(paciente)
    orq = Orquestrador(llm, retriever=retriever_com_docs(tmp_path, llm))
    estado = orq.nova_sessao(); orq.iniciar(estado)
    t = conversar(orq, estado)
    assert t.etapa == "alerta" and "Procure atendimento médico agora" in t.mensagem
    assert estado.interrompida and estado.concluida
    assert "alerta_2" not in estado.perguntadas and "hist_1" not in estado.perguntadas  # parou cedo
    r = orq.executar(estado)
    assert r.interrompido and r.mensagem_urgente
    assert r.analise.classificacao.grupo in (Grupo.C, Grupo.D)
    assert not any("ORIENTAÇÕES" in s for s in llm.sistemas_chamados)   # sem LLM de orientações
    assert "FLUXO INTERROMPIDO" in r.relatorio_md and "Não perguntado" in r.relatorio_md


def test_dificuldade_respirar_orienta_samu():
    paciente = {**BASE, "sinais_alerta.dificuldade_respirar": "sim"}
    orq = Orquestrador(FakeLLM(paciente))
    estado = orq.nova_sessao(); orq.iniciar(estado)
    t = conversar(orq, estado)
    assert "192" in t.mensagem
    assert orq.executar(estado).analise.classificacao.grupo == Grupo.D


def test_rag_vazio_nao_inventa_orientacoes():
    llm = FakeLLM(BASE)
    orq = Orquestrador(llm, retriever=Retriever(llm, LocalStore("/tmp/inexistente_vazio.json")))
    estado = orq.nova_sessao(); orq.iniciar(estado); conversar(orq, estado)
    r = orq.executar(estado)
    assert r.orientacoes.contexto_insuficiente and r.orientacoes.aviso
    assert not r.orientacoes.referencias
    assert not any("ORIENTAÇÕES" in s for s in llm.sistemas_chamados)


def test_orientacao_sem_referencia_ou_com_dose_e_descartada(tmp_path):
    llm = FakeLLM(BASE, orientacao={
        "orientacoes": [
            {"texto": "Tome 500 mg de paracetamol a cada 6 horas.", "refs": [1]},     # dose -> fora
            {"texto": "Invenção sem suporte nenhum.", "refs": []},                      # sem ref -> fora
            {"texto": "Referência inexistente.", "refs": [99]},                         # ref inválida -> fora
            {"texto": "Procure atendimento se surgirem sinais de alarme.", "refs": [2, 1]},
        ],
        "quando_procurar_atendimento": [], "contexto_insuficiente": False})
    orq = Orquestrador(llm, retriever=retriever_com_docs(tmp_path, llm))
    estado = orq.nova_sessao(); orq.iniciar(estado); conversar(orq, estado)
    o = orq.executar(estado).orientacoes
    assert [i.texto for i in o.orientacoes] == ["Procure atendimento se surgirem sinais de alarme."]
    assert sorted(o.orientacoes[0].refs) == [1, 2] and len(o.referencias) == 2


def test_encerrar_antes_gera_relatorio_parcial():
    orq = Orquestrador(FakeLLM(BASE))
    estado = orq.nova_sessao(); orq.iniciar(estado)
    orq.turno(estado, "30 anos")
    orq.encerrar_agora(estado)
    r = orq.executar(estado)
    assert "Não perguntado" in r.relatorio_md and not r.interrompido
    assert r.analise.classificacao.grupo == Grupo.B       # dados incompletos -> nunca A


def test_montar_consultas_inclui_condicoes_especiais():
    from tests.conftest import dados_com
    from dengue.safety import classificar_manejo
    from dengue.schemas import AnaliseInicial
    d = dados_com(identificacao__gravidez="sim")
    a = AnaliseInicial(classificacao=classificar_manejo(d))
    consultas = AgenteOrientacoes(FakeLLM()).montar_consultas(d, a)
    assert any("gestantes" in c for c in consultas) and any("grupo B" in c for c in consultas)
