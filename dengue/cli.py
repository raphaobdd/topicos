"""Conversa pelo terminal (sem Streamlit) — útil para depurar com a API OpenAI.

    python -m dengue.cli
"""
from __future__ import annotations

from .orchestrator import criar_orquestrador


def main() -> None:
    orq = criar_orquestrador()
    estado = orq.nova_sessao()
    print("\nAssistente:", orq.iniciar(estado))
    while not estado.concluida:
        try:
            texto = input("\nVocê: ").strip()
        except (EOFError, KeyboardInterrupt):
            orq.encerrar_agora(estado)
            break
        if texto:
            print("\nAssistente:", orq.turno(estado, texto).mensagem)
    resultado = orq.executar(estado)
    print("\n" + "=" * 70 + "\n" + resultado.relatorio_md)


if __name__ == "__main__":
    main()
