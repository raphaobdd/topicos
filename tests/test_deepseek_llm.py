from types import SimpleNamespace

import pytest

from dengue.llm import DeepSeekLLM, LLMErro, LocalEmbedder


class Respostas:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = '{"campos": {}, "nao_sabe": [], "mensagem": ""}' if "text" in kwargs else "Resumo."
        return SimpleNamespace(status="completed", output_text=content)


class Encoder:
    def encode(self, textos, **kwargs):
        assert kwargs["normalize_embeddings"] is True
        return [[1.0, 0.0] for _ in textos]


def test_deepseek_texto_json_e_embedding_local():
    client = SimpleNamespace(responses=Respostas())
    llm = DeepSeekLLM(client=client, encoder=Encoder(), dim=2)
    assert llm.texto("sistema", "prompt") == "Resumo."
    assert llm.json("sistema", "prompt")["campos"] == {}
    assert client.responses.calls[1]["text"] == {"format": {"type": "json_object"}}
    assert all(c["reasoning"] == {"effort": "none"} for c in client.responses.calls)
    assert llm.embed(["um", "dois"]) == [[1.0, 0.0], [1.0, 0.0]]


def test_deepseek_exige_chave_sem_cliente(monkeypatch):
    monkeypatch.setattr("dengue.config.DEEPSEEK_API_KEY", "")
    with pytest.raises(LLMErro, match="DEEPSEEK_API_KEY"):
        DeepSeekLLM()
    assert LocalEmbedder(encoder=Encoder(), dim=2).embed(["dengue"]) == [[1.0, 0.0]]


def test_deepseek_saldo_insuficiente_tem_mensagem_util():
    class SemSaldo:
        def create(self, **kwargs):
            erro = RuntimeError("Insufficient Balance")
            erro.status_code = 402
            raise erro

    llm = DeepSeekLLM(client=SimpleNamespace(responses=SemSaldo()))
    with pytest.raises(LLMErro, match="Saldo insuficiente"):
        llm.json("JSON", "idade 22")
