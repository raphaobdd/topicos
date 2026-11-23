"""Recorte do manual clínico de 2024 por títulos verificados no PDF oficial.

Os números de página são páginas físicas do PDF (1-based). Mudanças de edição
devem falhar no validador em vez de atribuir trechos ao capítulo errado.
"""
from __future__ import annotations

import re
from collections.abc import Iterator

CLINICO = "ministerio_saude_dengue_manejo_2024"
FLUXOGRAMA = "fluxograma_manejo_dengue_2025"
ENFERMAGEM = "manual_enfermagem_dengue_2026"
SINAN_FICHA = "sinan_ficha_dengue_chik"
SINAN_INSTRUCOES = "sinan_instrucional_dengue_chik"

# (página física, capítulo, início do título). Inclui limites não usados pelos agentes.
MARCOS = [
    (12, "2", "Espectro"), (13, "2.1", "FASE FEBRIL"),
    (13, "2.2", "FASE CR"), (19, "2.3", "FASE DE RECUPERA"),
    (19, "2.4", "ASPECTOS CL"), (20, "2.5", "ASPECTOS CL"),
    (21, "3", "Diagn"), (24, "4", "Atendimento"),
    (24, "4.1", "ANAMNESE"), (25, "4.2", "EXAME F"),
    (26, "5", "Classifica"), (29, "6", "Estadiamento"),
    (29, "6.1", "GRUPO A"), (32, "6.2", "GRUPO B"),
    (33, "6.3", "GRUPO C"), (36, "6.4", "GRUPO D"),
    (40, "6.5", "CONSIDERA"), (42, "6.6", "CORRE"),
    (43, "6.7", "DIST"), (43, "6.8", "ÓBITO"),
    (44, "6.9", "INDICA"), (44, "6.10", "CRIT"),
    (45, "7", "Tratamento"), (47, "8", "Hidrata"),
    (53, "9", "Gesta"), (56, "10", "Classifica"),
    (56, "10.1", "CASO SUSPEITO DE DENGUE"),
    (56, "10.2", "CASO SUSPEITO DE DENGUE COM"),
    (57, "10.3", "CASO SUSPEITO DE DENGUE GRAVE"),
    (57, "10.4", "CASO CONFIRMADO"),
    (59, "11", "Uso de antiagregantes"),
]


def extrair_secoes_clinicas(leitor) -> Iterator[tuple[int, str, str]]:
    """Gera (página, capítulo, texto), separando títulos na mesma página."""
    encontrados: set[str] = set()
    atual: str | None = None
    for pagina in range(12, min(len(leitor.pages), 64) + 1):
        texto = leitor.pages[pagina - 1].extract_text() or ""
        posicoes = []
        for pg, cap, titulo in MARCOS:
            if pg != pagina:
                continue
            padrao = re.compile(r"(?<![\d.])" + re.escape(cap) + r"\s+" + re.escape(titulo), re.I)
            m = padrao.search(texto)
            if not m:
                raise ValueError(f"Capítulo {cap} não encontrado na página PDF {pagina}")
            posicoes.append((m.start(), cap))
            encontrados.add(cap)
        posicoes.sort()
        if atual and (not posicoes or posicoes[0][0] > 0):
            fim = posicoes[0][0] if posicoes else len(texto)
            if texto[:fim].strip():
                yield pagina, atual, texto[:fim]
        for i, (inicio, cap) in enumerate(posicoes):
            fim = posicoes[i + 1][0] if i + 1 < len(posicoes) else len(texto)
            if texto[inicio:fim].strip():
                yield pagina, cap, texto[inicio:fim]
            atual = cap
    esperados = {cap for _, cap, _ in MARCOS}
    if encontrados != esperados:
        raise ValueError(f"Capítulos não encontrados: {sorted(esperados - encontrados)}")


def marcar_capitulo(capitulo: str, conteudo: str) -> str:
    return f"[[capitulo:{capitulo}]] {conteudo}"


def capitulo_do_trecho(conteudo: str) -> str | None:
    m = re.match(r"\[\[capitulo:([^\]]+)\]\]", conteudo)
    return m.group(1) if m else None


POLITICA = {
    1: {CLINICO: {"2", "2.1", "2.2", "2.3", "2.4", "2.5", "3", "4.1"}},
    2: {CLINICO: {"2.2", "5", "6", "6.1", "6.2", "6.3", "6.4", "6.5", "10.1", "10.2", "10.3"}},
    3: {CLINICO: {"6.1", "6.2", "6.3", "6.4", "6.5", "6.9", "6.10", "7", "8", "9", "11"},
        FLUXOGRAMA: {"integral"}, ENFERMAGEM: {"integral"}},
    4: {CLINICO: {"10"}, SINAN_FICHA: {"integral"}, SINAN_INSTRUCOES: {"integral"}},
}


def permitido(agente: int, fonte: str, conteudo: str) -> bool:
    cap = capitulo_do_trecho(conteudo)
    permitidos = POLITICA[agente].get(fonte, set())
    if cap is None:
        return False
    # O capítulo 10 ou 11 inteiro inclui suas subseções. Nos demais, apenas os
    # subcapítulos explicitamente listados incluem descendentes.
    return any(cap == p or (cap.startswith(p + ".") and p not in {"2", "6"}) for p in permitidos)
