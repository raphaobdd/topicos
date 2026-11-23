"""Texto/JSON pela API DeepSeek; embeddings locais para o RAG."""
from __future__ import annotations

import json
import math
import re
from typing import Protocol

from . import config


class LLMErro(RuntimeError):
    pass


class LLM(Protocol):
    def texto(self, sistema: str, prompt: str, temperatura: float = 0.2) -> str: ...
    def json(self, sistema: str, prompt: str, temperatura: float = 0.0) -> dict: ...
    def embed(self, textos: list[str], consulta: bool = False) -> list[list[float]]: ...


def _normalizar(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def extrair_json(txt: str) -> dict:
    txt = (txt or "").strip()
    txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.I)
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", txt, flags=re.S)
        if not m:
            raise LLMErro(f"Resposta sem JSON válido: {txt[:200]!r}")
        return json.loads(m.group(0))


class LocalEmbedder:
    def __init__(self, modelo_embed: str | None = None, dim: int | None = None, encoder=None):
        self.modelo_embed = modelo_embed or config.EMBED_MODEL
        self.dim = dim or config.EMBED_DIM
        self._encoder = encoder

    def embed(self, textos: list[str], consulta: bool = False) -> list[list[float]]:
        if not textos:
            return []
        if any(not t.strip() for t in textos):
            raise ValueError("Texto vazio não pode ser convertido em embedding")
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.modelo_embed)
        vetores = self._encoder.encode(textos, batch_size=32, normalize_embeddings=True)
        saida = [list(map(float, v)) for v in vetores]
        if len(saida) != len(textos) or any(len(v) != self.dim for v in saida):
            raise LLMErro("Quantidade ou dimensão dos embeddings locais não corresponde à configuração")
        return saida


class DeepSeekLLM(LocalEmbedder):
    def __init__(self, api_key: str | None = None, modelo: str | None = None,
                 modelo_embed: str | None = None, dim: int | None = None, client=None, encoder=None):
        from openai import OpenAI

        self.api_key = api_key or config.DEEPSEEK_API_KEY
        if not self.api_key and client is None:
            raise LLMErro("DEEPSEEK_API_KEY não configurada. Copie .env.example para .env e preencha.")
        self.client = client if client is not None else OpenAI(
            api_key=self.api_key, base_url="https://api.deepseek.com", max_retries=2)
        self.modelo = modelo or config.DEEPSEEK_MODEL
        super().__init__(modelo_embed=modelo_embed, dim=dim, encoder=encoder)

    def _gerar(self, sistema: str, prompt: str, temperatura: float, json_mode: bool) -> str:
        kwargs = {"model": self.modelo, "instructions": sistema, "input": prompt,
                  "temperature": temperatura, "store": False,
                  "reasoning": {"effort": "none"}}
        if json_mode:
            kwargs["text"] = {"format": {"type": "json_object"}}
        try:
            resp = self.client.responses.create(**kwargs)
        except Exception as e:
            if getattr(e, "status_code", None) == 402:
                raise LLMErro("Saldo insuficiente na conta DeepSeek. Confira o faturamento no painel da DeepSeek.") from e
            if getattr(e, "status_code", None) == 401:
                raise LLMErro("Chave DeepSeek inválida ou sem acesso. Confira DEEPSEEK_API_KEY no .env.") from e
            raise LLMErro(f"Falha ao chamar a DeepSeek: {type(e).__name__}") from e
        if resp.status != "completed" or not resp.output_text:
            raise LLMErro("Resposta vazia ou incompleta da DeepSeek.")
        return resp.output_text.strip()

    def texto(self, sistema: str, prompt: str, temperatura: float = 0.2) -> str:
        return self._gerar(sistema, prompt, temperatura, json_mode=False)

    def json(self, sistema: str, prompt: str, temperatura: float = 0.0) -> dict:
        for tentativa in range(2):
            try:
                return extrair_json(self._gerar(sistema, prompt, temperatura, json_mode=True))
            except (json.JSONDecodeError, LLMErro):
                if tentativa:
                    raise
        raise AssertionError("inalcançável")

