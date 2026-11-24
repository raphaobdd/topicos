"""Executa casos fictícios revisados em JSONL pelo fluxo real da aplicação.

Formato por linha: {"id":"A001", "grupo":"A", "alerta":false,
"mensagens":["..."], "campos_esperados":{"evolucao.febre":"sim"}}.
Rótulos clínicos devem ser revisados por profissional antes de divulgação.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .orchestrator import criar_orquestrador


def executar(casos, orquestrador):
    linhas = []
    for caso in casos:
        inicio = time.perf_counter()
        estado = orquestrador.nova_sessao()
        orquestrador.iniciar(estado)
        for mensagem in caso["mensagens"]:
            if estado.concluida:
                break
            orquestrador.turno(estado, mensagem)
        if not estado.concluida:
            orquestrador.encerrar_agora(estado)
        resultado = orquestrador.executar(estado)
        esperado = caso.get("campos_esperados", {})
        campos = {campo: str(estado.dados.obter(campo).value if hasattr(estado.dados.obter(campo), "value")
                             else estado.dados.obter(campo)) == str(valor)
                  for campo, valor in esperado.items()}
        refs = {r.numero for r in resultado.orientacoes.referencias}
        itens = resultado.orientacoes.orientacoes + resultado.orientacoes.quando_procurar_atendimento
        refs_validas = all(set(item.refs) <= refs for item in itens)
        linhas.append({"id": caso["id"], "grupo_esperado": caso["grupo"],
                       "grupo_obtido": resultado.analise.classificacao.grupo.value,
                       "alerta_esperado": bool(caso["alerta"]), "alerta_obtido": resultado.interrompido,
                       "extracao_campos": campos, "referencias_validas": refs_validas,
                       "contexto_insuficiente": resultado.orientacoes.contexto_insuficiente,
                       "relatorio_consistente": bool(resultado.relatorio_md.strip()) and
                       resultado.analise.classificacao.grupo.value in resultado.relatorio_md,
                       "tempo_segundos": round(time.perf_counter() - inicio, 3)})
    y, pred = [r["grupo_esperado"] for r in linhas], [r["grupo_obtido"] for r in linhas]
    alertas = [r for r in linhas if r["alerta_esperado"]]
    campos = [v for r in linhas for v in r["extracao_campos"].values()]
    metricas = {"acuracia_grupos": accuracy_score(y, pred),
                "classificacao": classification_report(y, pred, labels=list("ABCD"), output_dict=True, zero_division=0),
                "matriz_confusao": confusion_matrix(y, pred, labels=list("ABCD")).tolist(),
                "recall_alertas": sum(r["alerta_obtido"] for r in alertas) / len(alertas) if alertas else None,
                "acuracia_extracao": sum(campos) / len(campos) if campos else None,
                "referencias_validas": sum(r["referencias_validas"] for r in linhas) / len(linhas),
                "consistencia_relatorios": sum(r["relatorio_consistente"] for r in linhas) / len(linhas),
                "tempo_medio_segundos": sum(r["tempo_segundos"] for r in linhas) / len(linhas)}
    return {"gerado_em": datetime.now(timezone.utc).isoformat(), "n_casos": len(linhas),
            "aviso": "Benchmark fictício; referências válidas verificam apenas existência, não fidelidade semântica. Não é validação clínica.",
            "metricas": metricas, "casos": linhas}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--casos", required=True, help="JSONL com casos rotulados e mensagens simuladas")
    ap.add_argument("--saida", default="models/benchmark.json")
    args = ap.parse_args()
    casos = [json.loads(s) for s in Path(args.casos).read_text(encoding="utf-8").splitlines() if s.strip()]
    if not casos:
        raise SystemExit("Arquivo de casos vazio")
    resultado = executar(casos, criar_orquestrador())
    destino = Path(args.saida)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Benchmark salvo em {destino}")


if __name__ == "__main__":
    main()
