"""Teste de fumaça da interface Streamlit (sem rede): AppTest + orquestrador com LLM falso."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

import dengue.orchestrator as orch
from tests.conftest import BASE, FakeLLM


def _app(monkeypatch, paciente):
    import streamlit as st

    st.cache_resource.clear()  # o app cacheia o orquestrador entre execuções
    monkeypatch.setattr(orch, "criar_orquestrador", lambda: orch.Orquestrador(FakeLLM(paciente)))
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=30)
    at.run()
    return at


def test_conversa_completa_mostra_resultado_e_relatorio(monkeypatch):
    at = _app(monkeypatch, BASE)
    assert not at.exception and "Qual é a sua idade" in at.chat_message[0].markdown[0].value
    for _ in range(40):
        if at.session_state["estado"].concluida:
            break
        at.chat_input[0].set_value("ok").run()
        assert not at.exception
    assert at.session_state["estado"].concluida
    at.run()
    assert any("Avaliação automatizada experimental: Grupo A" in s.value for s in at.subheader)
    assert len(at.tabs) == 3


def test_sinal_de_alerta_mostra_banner_de_emergencia(monkeypatch):
    at = _app(monkeypatch, {**BASE, "sinais_alerta.dificuldade_respirar": "sim"})
    for _ in range(40):
        if at.session_state["estado"].concluida:
            break
        at.chat_input[0].set_value("ok").run()
    at.run()
    assert any("Fluxo interrompido" in e.value for e in at.error)
    assert any("192" in m.markdown[0].value for m in at.chat_message)
