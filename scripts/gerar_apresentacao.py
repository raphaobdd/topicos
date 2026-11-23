"""Gera slides editaveis e PDF com os resultados medidos no projeto."""
from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import landscape, A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "apresentacao"
METRICAS = json.loads((RAIZ / "models/arvore_dengue.metricas.json").read_text(encoding="utf-8"))

SLIDES = [
    ("Sistema multiagente de IA para dengue", [
        "Protótipo acadêmico para organizar informações de casos suspeitos",
        "Coleta conversacional • regras de alerta • análise • RAG • relatório",
        "Não diagnostica e não substitui profissionais de saúde."]),
    ("Problema e objetivo", [
        "A dengue exige coleta organizada e reconhecimento rápido de sinais de alerta.",
        "O projeto apoia uma avaliação inicial experimental e gera um relatório estruturado.",
        "O atendimento presencial continua indispensável."]),
    ("Arquitetura implementada", [
        "1  Coleta: roteiro fixo; OpenAI interpreta respostas livres em JSON validado.",
        "2  Análise: grupo A/B/C/D por regras; árvore acrescenta estatística da base.",
        "3  Orientações: busca em manual oficial, com fonte e página.",
        "4  Relatório: Markdown montado dos dados estruturados.",
        "Orquestrador verifica alertas após cada turno e interrompe o fluxo quando necessário."]),
    ("Base real e avaliação da árvore", [
        "Fonte: SINAN/Dengue 2024, Portal de Dados Abertos do SUS.",
        "Arquivo: 6.564.924 registros; amostra aleatória de 500.000, semente 42.",
        f"Alvo: CLASSI_FIN 11/12 versus 10; prevalência na amostra elegível: {METRICAS['prevalencia']:.2%}.",
        f"Teste: AUC {METRICAS['metricas']['auc_teste']:.3f}; precisão média {METRICAS['metricas']['average_precision']:.3f}.",
        "A estimativa não é risco individual nem substitui a verificação de alertas."]),
    ("Estado da integração", [
        "Código: Streamlit, quatro agentes, regras, treino e relatórios implementados.",
        "Dados: CSV oficial baixado; manual de manejo do Ministério da Saúde disponível para indexação.",
        "Supabase: esquema e importador preparados; carga remota depende do projeto e conexão.",
        "OpenAI depende de chave API; o RAG usa embeddings locais indexados no Supabase.",
        "Testes automatizados: executar antes da demonstração."]),
    ("Limitações e próximos passos", [
        "Validar sinais e classificações com profissional de saúde e protocolo vigente.",
        "Configurar acessos; importar a base; indexar o manual; testar dois percursos reais.",
        "Revisar métricas em outros recortes e medir desempenho por subgrupos.",
        "Ferramenta acadêmica: não usar para diagnóstico ou decisão clínica."]),
]


def gerar_pptx():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    for numero, (titulo, linhas) in enumerate(SLIDES, 1):
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        fundo = slide.background.fill
        fundo.solid(); fundo.fore_color.rgb = RGBColor(11, 32, 51)
        box = slide.shapes.add_textbox(Inches(.7), Inches(.45), Inches(12), Inches(.9)).text_frame
        box.text = titulo
        p = box.paragraphs[0]; p.font.size = Pt(30); p.font.bold = True; p.font.color.rgb = RGBColor(255, 255, 255)
        corpo = slide.shapes.add_textbox(Inches(.85), Inches(1.65), Inches(11.7), Inches(5.2)).text_frame
        corpo.word_wrap = True
        for i, linha in enumerate(linhas):
            p = corpo.paragraphs[0] if i == 0 else corpo.add_paragraph()
            p.text = linha; p.space_after = Pt(22)
            p.font.size = Pt(21 if len(linha) < 100 else 18)
            p.font.color.rgb = RGBColor(224, 238, 243)
        rodape = slide.shapes.add_textbox(Inches(.8), Inches(7.08), Inches(11), Inches(.25)).text_frame
        rodape.text = f"Projeto acadêmico • 23/11/2026                                      {numero}/{len(SLIDES)}"
        rodape.paragraphs[0].font.size = Pt(10)
        rodape.paragraphs[0].font.color.rgb = RGBColor(150, 190, 202)
    prs.save(SAIDA / "sistema_dengue_atualizado.pptx")


def gerar_pdf():
    fonte = Path("C:/Windows/Fonts/arial.ttf")
    if fonte.exists():
        pdfmetrics.registerFont(TTFont("Arial", str(fonte)))
    nome_fonte = "Arial" if fonte.exists() else "Helvetica"
    c = canvas.Canvas(str(SAIDA / "sistema_dengue_atualizado.pdf"), pagesize=landscape(A4))
    largura, altura = landscape(A4)
    for numero, (titulo, linhas) in enumerate(SLIDES, 1):
        c.setFillColor(HexColor("#0b2033")); c.rect(0, 0, largura, altura, fill=1, stroke=0)
        c.setFillColor(HexColor("#ffffff")); c.setFont(nome_fonte, 25); c.drawString(42, altura-65, titulo)
        y = altura-135
        for linha in linhas:
            c.setFillColor(HexColor("#e0eef3")); c.setFont(nome_fonte, 15)
            # Quebra por largura para manter legibilidade no PDF.
            palavras, parte = linha.split(), ""
            for palavra in palavras:
                proximo = (parte + " " + palavra).strip()
                if c.stringWidth(proximo, nome_fonte, 15) > largura-90 and parte:
                    c.drawString(53, y, parte); y -= 23; parte = palavra
                else:
                    parte = proximo
            c.drawString(53, y, parte); y -= 56
        c.setFillColor(HexColor("#96beca")); c.setFont(nome_fonte, 9)
        c.drawString(42, 23, f"Projeto acadêmico • 23/11/2026                                           {numero}/{len(SLIDES)}")
        c.showPage()
    c.save()


if __name__ == "__main__":
    SAIDA.mkdir(exist_ok=True)
    gerar_pptx(); gerar_pdf()
    print(SAIDA)
