from dengue.rag.ingest import dividir_texto, indexar
from dengue.rag.retriever import Retriever
from dengue.rag.store import LocalStore, criar_store
from tests.conftest import FakeLLM


def test_dividir_texto_respeita_tamanho_e_sobreposicao():
    texto = ". ".join(f"Frase número {i} sobre dengue e hidratação oral" for i in range(200))
    chunks = dividir_texto(texto, tamanho=400, sobreposicao=60)
    assert len(chunks) > 5 and all(len(c) <= 400 for c in chunks)
    assert dividir_texto("curto") == []


def test_indexar_e_recuperar_trecho_relevante(tmp_path):
    pasta = tmp_path / "docs"; pasta.mkdir()
    (pasta / "manual.txt").write_text(
        "Os sinais de alarme da dengue são dor abdominal intensa, vômitos persistentes e sangramento. " * 3
        + "\n\nA vacinação contra influenza ocorre anualmente em campanhas nacionais para grupos prioritários. " * 3,
        encoding="utf-8")
    llm = FakeLLM(); store = LocalStore(tmp_path / "v.json")
    assert indexar(llm, store, pasta, tamanho=250, sobreposicao=30) >= 2
    r = Retriever(llm, store, k=2)
    top = r.buscar("quais são os sinais de alarme da dengue dor abdominal")[0]
    assert "alarme" in top.conteudo and top.fonte == "manual"
    # persistência + reindexação idempotente
    n1 = LocalStore(tmp_path / "v.json").contar()
    indexar(llm, store, pasta, tamanho=250, sobreposicao=30)
    assert store.contar() == n1


def test_buscar_varias_remove_duplicados(tmp_path):
    llm = FakeLLM(); store = LocalStore(tmp_path / "v.json")
    from tests.conftest import embedding_hash
    c = "sinais de alarme dengue dor abdominal vômitos persistentes " * 2
    store.adicionar([{"fonte": "a", "pagina": 1, "conteudo": c, "embedding": embedding_hash(c)}])
    out = Retriever(llm, store, k=3).buscar_varias(["sinais de alarme", "dor abdominal dengue"])
    assert len(out) == 1


def test_indice_local_prevalece_no_modo_auto_mesmo_com_supabase_configurado(tmp_path, monkeypatch):
    from dengue import config
    p = tmp_path / "vetores.json"
    p.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(config, "LOCAL_STORE", p)
    monkeypatch.setattr(config, "RAG_STORE", "auto")
    monkeypatch.setattr(config, "SUPABASE_URL", "https://exemplo.supabase.co")
    monkeypatch.setattr(config, "SUPABASE_KEY", "chave")
    assert isinstance(criar_store(), LocalStore)
