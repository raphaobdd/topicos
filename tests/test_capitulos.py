from dengue.rag.capitulos import permitido


def test_fontes_e_capitulos_isolados_por_agente():
    clinico = "ministerio_saude_dengue_manejo_2024"
    marcado = lambda cap: f"[[capitulo:{cap}]] trecho"
    assert permitido(1, clinico, marcado("2.2"))
    assert not permitido(1, clinico, marcado("6.4"))
    assert permitido(2, clinico, marcado("6.4"))
    assert not permitido(2, clinico, marcado("6.9"))
    assert permitido(3, clinico, marcado("6.9"))
    assert permitido(3, "manual_enfermagem_dengue_2026", marcado("integral"))
    assert not permitido(3, "sinan_ficha_dengue_chik", marcado("integral"))
    assert permitido(4, clinico, marcado("10.4"))
    assert permitido(4, "sinan_instrucional_dengue_chik", marcado("integral"))
    assert not permitido(4, clinico, marcado("6.1"))
    assert not permitido(1, clinico, "trecho legado sem capítulo")
