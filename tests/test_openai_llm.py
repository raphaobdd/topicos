from types import SimpleNamespace

import pytest

from dengue.llm import LLMErro, LocalEmbedder, OpenAILLM


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


def test_openai_texto_json_e_embedding_local():
    client = SimpleNamespace(responses=Respostas())
    llm = OpenAILLM(client=client, encoder=Encoder(), dim=2)
    assert llm.texto("sistema", "prompt") == "Resumo."
    assert llm.json("sistema", "prompt")["campos"] == {}
    assert client.responses.calls[1]["text"] == {"format": {"type": "json_object"}}
    assert "JSON" in client.responses.calls[1]["input"]
    assert all(c["store"] is False for c in client.responses.calls)
    assert llm.embed(["um", "dois"]) == [[1.0, 0.0], [1.0, 0.0]]


def test_openai_exige_chave_sem_cliente(monkeypatch):
    monkeypatch.setattr("dengue.config.OPENAI_API_KEY", "")
    with pytest.raises(LLMErro, match="OPENAI_API_KEY"):
        OpenAILLM()
    assert LocalEmbedder(encoder=Encoder(), dim=2).embed(["dengue"]) == [[1.0, 0.0]]


def test_openai_cota_indisponivel_tem_mensagem_util():
    class SemCota:
        def create(self, **kwargs):
            erro = RuntimeError("Quota exceeded")
            erro.status_code = 429
            raise erro

    llm = OpenAILLM(client=SimpleNamespace(responses=SemCota()))
    with pytest.raises(LLMErro, match="limite ou cota"):
        llm.json("JSON", "idade 22")
