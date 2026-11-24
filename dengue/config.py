"""Configuração central (lê variáveis do .env)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parent.parent
load_dotenv(RAIZ / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
RAG_STORE = os.getenv("RAG_STORE", "auto").lower()

DOCS_DIR = RAIZ / "data" / "docs"
LOCAL_STORE = RAIZ / "data" / "vetores_locais.json"
MODELO_ARVORE = RAIZ / "models" / "arvore_dengue.joblib"
MODELO_XGBOOST = RAIZ / "models" / "xgboost_dengue.joblib"

AVISO_CURTO = (
    "Protótipo acadêmico e experimental. Não substitui consulta médica, não faz "
    "diagnóstico e não toma decisões clínicas."
)
