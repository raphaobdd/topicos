"""Interface Streamlit — rode com:  streamlit run app.py"""
from __future__ import annotations

import streamlit as st

from dengue import config
from dengue.orchestrator import criar_orquestrador
from dengue.schemas import DESCRICAO_GRUPOS

st.set_page_config(page_title="Triagem de dengue (protótipo)", page_icon="🩺", layout="centered")


@st.cache_resource(show_spinner="Iniciando componentes...")
def get_orquestrador():
    return criar_orquestrador()


try:
    orq = get_orquestrador()
except Exception as e:  # ex.: sem DEEPSEEK_API_KEY
    st.error(f"Não foi possível iniciar a aplicação: {e}")
    if isinstance(e, ModuleNotFoundError):
        st.info("Instale as dependências com `python -m pip install -r requirements.txt` e reinicie.")
    elif "DEEPSEEK_API_KEY" in str(e):
        st.info("Copie `.env.example` para `.env`, preencha `DEEPSEEK_API_KEY` e reinicie.")
    else:
        st.info("Confira as configurações do `.env` e o status dos serviços usados pela aplicação.")
    st.stop()


def nova_sessao() -> None:
    estado = orq.nova_sessao()
    orq.iniciar(estado)
    st.session_state.estado = estado
    st.session_state.resultado = None


if "estado" not in st.session_state:
    nova_sessao()

estado = st.session_state.estado

# ----------------------------- cabeçalho ---------------------------------- #
st.title("🩺 Avaliação inicial de suspeita de dengue")
st.warning(
    f"**{config.AVISO_CURTO}** Em caso de emergência, procure um serviço de saúde ou ligue **192 (SAMU)**. "
    "Não informe nome, CPF ou endereço."
)

with st.sidebar:
    st.header("Sobre")
    st.write(
        "Um agente conversa com você, outro classifica o caso com regras do protocolo do Ministério da Saúde, "
        "outro busca orientações em documentos oficiais (RAG) e outro monta o relatório."
    )
    if st.button("🔄 Nova conversa", use_container_width=True):
        nova_sessao()
        st.rerun()
    if not estado.concluida and st.button("⏹️ Encerrar e gerar relatório", use_container_width=True):
        orq.encerrar_agora(estado)
        st.rerun()
    with st.expander("Status dos componentes"):
        for nome, valor in orq.status().items():
            st.write(f"**{nome}:** {valor}")

# ------------------------------ conversa ---------------------------------- #
for m in estado.mensagens:
    with st.chat_message(m.role):
        st.markdown(m.content)

if not estado.concluida:
    texto = st.chat_input("Digite sua resposta...")
    if texto:
        with st.chat_message("user"):
            st.markdown(texto)
        with st.spinner("Analisando sua resposta..."):
            orq.turno(estado, texto)
        st.rerun()

# ------------------------------ resultado --------------------------------- #
if estado.concluida:
    if st.session_state.get("resultado") is None:
        with st.spinner("Analisando o caso, consultando documentos e montando o relatório..."):
            st.session_state.resultado = orq.executar(estado)
    r = st.session_state.resultado

    st.divider()
    if r.interrompido:
        st.error("Fluxo interrompido: foram relatados sinais de alerta. Procure atendimento médico agora (192 - SAMU em caso grave).")
    else:
        g = r.analise.classificacao.grupo.value
        st.subheader(f"Avaliação automatizada experimental: Grupo {g}")
        st.caption(DESCRICAO_GRUPOS[g] + " A classificação definitiva é feita por um profissional de saúde.")
        st.write(r.analise.resumo)

    aba_orient, aba_rel, aba_tec = st.tabs(["Orientações", "Relatório", "Detalhes técnicos"])

    with aba_orient:
        o = r.orientacoes
        if o.aviso:
            st.info(o.aviso)
        for it in o.orientacoes:
            ref = f" _(ref. {', '.join(map(str, it.refs))})_" if it.refs else ""
            st.markdown(f"- {it.texto}{ref}")
        if o.quando_procurar_atendimento:
            st.markdown("**Procure atendimento se houver:**")
            for it in o.quando_procurar_atendimento:
                ref = f" _(ref. {', '.join(map(str, it.refs))})_" if it.refs else ""
                st.markdown(f"- {it.texto}{ref}")
        if o.referencias:
            st.markdown("**Referências**")
            for ref in o.referencias:
                pg = f", p. {ref.pagina}" if ref.pagina else ""
                st.caption(f"{ref.numero}. {ref.fonte}{pg}")

    with aba_rel:
        st.download_button(
            "⬇️ Baixar relatório (.md)",
            data=r.relatorio_md,
            file_name=f"relatorio_dengue_{r.sessao_id[:8]}.md",
            mime="text/markdown",
        )
        st.markdown(r.relatorio_md)

    with aba_tec:
        st.markdown("**Etapas executadas**")
        for etapa in r.etapas:
            st.write(f"- {etapa}")
        with st.expander("Dados estruturados (JSON)"):
            st.json(estado.dados.model_dump(mode="json"))
        if r.analise.arvore:
            with st.expander("Árvore de decisão"):
                st.json(r.analise.arvore.model_dump(mode="json"))
