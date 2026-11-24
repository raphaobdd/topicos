# Sistema multiagente de IA para avaliação inicial de casos suspeitos de dengue

Protótipo **acadêmico e experimental**: não substitui consulta médica, não diagnostica e não toma decisões clínicas.

```
Paciente → Agente 1 (coleta) → Orquestrador ⇄ regras de segurança
                                   ↓ (sem sinal de alerta)
           Agente 2 (regras A/B/C/D + árvore) → Agente 3 (RAG) → Agente 4 (relatório)
```
Sinal de alerta em qualquer turno → o orquestrador **interrompe** o fluxo, orienta procurar atendimento (192/SAMU se houver indício de gravidade) e gera relatório só com o que foi coletado.

## Como rodar

### Dados reais e Supabase

O recurso oficial DENGBR24 e o manual de manejo usados nesta versão estão descritos em [`data/FONTES.md`](data/FONTES.md). O CSV oficial de 2024 não inclui `NU_NOTIFIC`; a importação trata essa coluna como opcional.

Para guardar no Supabase apenas os campos usados pela análise, crie o projeto, execute `supabase_schema.sql` no SQL Editor e preencha `SUPABASE_DB_URL` no `.env` com a conexão PostgreSQL direta ou session pooler. Depois execute:

```bash
python -m dengue.ml.import_sinan --csv data/DENGBR24.csv.zip
python -m dengue.ml.train_tree --csv data/DENGBR24.csv.zip --amostra-aleatoria 500000
python -m dengue.rag.ingest
```

**Alternativa sem URL PostgreSQL:** com `SUPABASE_URL` e `SUPABASE_SERVICE_KEY`, execute `python -m dengue.ml.import_sinan_rest`. O importador usa a API HTTPS para carregar uma amostra reproduzível de 1,5% do CSV completo (semente 42), distribuída ao longo do arquivo. A carga é idempotente; reexecutar não duplica linhas. Em 23/11/2026, foram selecionadas e confirmadas **97.981 linhas** no Supabase. Isso é uma amostra para demonstração, não a base integral de 6.564.924 registros. O arquivo integral permanece em `data/DENGBR24.csv.zip` para análise e treino locais.

A carga é transacional e reconhece o hash do arquivo para não duplicar a mesma versão. A árvore treinada em amostra é uma avaliação estatística acadêmica; não representa risco individual. A chave OpenAI e a conexão Supabase nunca devem ser incluídas no repositório.

A apresentação atualizada está em `apresentacao/sistema_dengue_atualizado.pptx` e `.pdf`; regenere ambos com `python scripts/gerar_apresentacao.py` depois de refazer o treino. Os slides distinguem resultados medidos das integrações ainda pendentes. Antes de uma demonstração com serviços reais:

1. Criar projeto Supabase e chave da API OpenAI; preencher `OPENAI_API_KEY`, `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` e `SUPABASE_DB_URL` no `.env` local.
2. Conferir espaço disponível no projeto Supabase antes de carregar os 6,56 milhões de registros; a carga seleciona 23 colunas e mantém uma versão por hash de origem.
3. Executar o SQL, importar a base e indexar o PDF; verificar a contagem no banco e o status dos componentes na interface.
4. Percorrer um caso sintético sem alerta e outro com alerta no Streamlit; conferir orientações citadas, interrupção, relatório e sessão salva.
5. Revisar as regras e mensagens com profissional de saúde antes de qualquer apresentação fora do ambiente acadêmico.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # preencha OPENAI_API_KEY
pytest                                                 # 35 testes, sem rede

# (1) RAG: coloque PDFs oficiais em data/docs/ (ex.: "Dengue: diagnóstico e manejo clínico", Min. Saúde)
python -m dengue.rag.ingest --local                    # sem Supabase; sem --local usa o Supabase do .env

# (2) Árvore de decisão (opcional): análise da base + treino
python -m dengue.ml.train_tree --csv data/DENGBR24.csv.zip --explorar
python -m dengue.ml.train_tree --csv data/DENGBR24.csv.zip --max-linhas 500000   # teste rápido
python -m dengue.ml.train_tree --csv data/DENGBR24.csv.zip                       # base inteira

# (3) Interface
streamlit run app.py          # ou:  python -m dengue.cli  (terminal)
```

### Supabase (opcional)
1. Crie um projeto e rode `supabase_schema.sql` no SQL Editor (tabelas `sessoes` e `documentos`, índice HNSW e a função `match_documentos`).
2. Preencha `SUPABASE_URL` e `SUPABASE_SERVICE_KEY` no `.env` (chave **service_role**, só no backend; RLS ligado e sem políticas).
3. A dimensão `vector(384)` tem que bater com `EMBED_DIM`.
Sem Supabase: RAG usa `data/vetores_locais.json` e as sessões não são persistidas.

## Estrutura

| Arquivo | Papel |
|---|---|
| `dengue/schemas.py` | Modelos Pydantic; informação **tri-estado** (`sim` / `nao` / `nao_informado`) |
| `dengue/perguntas.py` | Roteiro das seções A–E (perguntas agrupadas, 1 campo por sinal) |
| `dengue/agents/coleta.py` | **Agente 1** — roteiro determinístico + OpenAI só interpreta a resposta livre (JSON validado) |
| `dengue/safety.py` | **Regras explícitas** de sinais de alerta e classificação A/B/C/D (sem LLM) |
| `dengue/orchestrator.py` | Orquestrador: fluxo, verificação a cada turno, interrupção, persistência |
| `dengue/agents/analise.py` | **Agente 2** — grupo vem das regras; árvore complementa; LLM só redige o resumo |
| `dengue/agents/orientacoes.py` | **Agente 3** — RAG; descarta item sem referência ou com dose |
| `dengue/agents/relatorio.py` | **Agente 4** — relatório em 12 seções, com origem de cada informação |
| `dengue/rag/` | `ingest` (PDF→chunks→embeddings), `store` (pgvector/local), `retriever` |
| `dengue/ml/` | `features` (treino = inferência), `train_tree` (EDA+treino), `predict`, `sintetico` |
| `app.py` | Interface Streamlit |

## Decisões importantes (revise com seu orientador)

- **Alvo da árvore ≠ grupos A/B/C/D.** O alvo é `CLASSI_FIN ∈ {11,12}` vs `10` entre casos confirmados: "chance de evoluir como dengue com sinais de alarme/grave dado o perfil na triagem". Fica **fora** tudo que vaza o alvo ou não existe na triagem: `ALRM_*`, `GRAV_*`, `HOSPITALIZ`, `EVOLUCAO`, `CRITERIO`, exames. Ignorado (código 9) vira `NaN`, não "não".
- **A árvore só entra na aplicação se for "viável"** (AUC no teste ≥ `--auc-minimo`, padrão 0.65) e não for sintética. Senão o relatório diz por que não foi usada. Probabilidades são comparadas com a prevalência da base, não apresentadas como risco individual.
- **Segurança não depende do LLM**: `ALERTAS` e `POSSIVEL_GRAVIDADE` em `dengue/safety.py` são listas pequenas e editáveis. Valide-as contra o protocolo do Ministério da Saúde. Escolhas conservadoras: dados faltantes sobre sinais de alerta/comorbidades → nunca Grupo A; manchas na pele → Grupo B (avaliar petéquias).
- **Perguntei sexo** (não está na especificação) porque a pergunta de gravidez é "quando aplicável" e o sexo entra na árvore.
- **Sem nome/CPF/endereço** em nenhum ponto. Use dados sintéticos/desidentificados em demonstrações.

## Pontos de atenção

- Modelo padrão: `OPENAI_MODEL=gpt-4.1-mini` para texto/JSON. O RAG usa `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` local para embeddings de 384 dimensões. O índice já criado no Supabase continua válido; a troca da API de texto não exige reindexação.
- `python -m dengue.rag.ingest --local` cria `data/vetores_locais.json` sem chave da API. Para Supabase, execute `python -m dengue.rag.ingest --supabase` após configurar `SUPABASE_URL` e `SUPABASE_SERVICE_KEY`. Se você já criou `documentos` com `vector(768)`, execute `supabase_recriar_rag_384.sql` e depois `supabase_schema.sql` antes de reindexar.
- `RAG_STORE=auto` usa o índice local quando `data/vetores_locais.json` existe, mesmo com Supabase configurado para salvar sessões. Defina `RAG_STORE=supabase` no `.env` para usar os vetores remotos, ou `RAG_STORE=local` para forçar o arquivo local.

### Fontes por agente

Todos os PDFs ficam na mesma tabela `documentos`. Cada trecho recebe um marcador de capítulo no conteúdo; a recuperação filtra por agente antes de fornecer contexto ao modelo. Reexecute `python -m dengue.rag.ingest --supabase` depois de atualizar os PDFs, pois trechos antigos sem marcador não entram nas buscas dos agentes.

| Agente | Fontes autorizadas |
|---|---|
| 1, coleta | Manual clínico 2024: 2, 2.1–2.5, 3, 4.1 |
| 2, análise | Manual clínico 2024: 2.2, 5, 6, 6.1–6.5, 10.1–10.3 |
| 3, orientações | Manual clínico 2024: 6.1–6.5, 6.9, 6.10, 7, 8, 9, 11; fluxograma completo; Manual de Enfermagem completo |
| 4, relatório | Ficha e instruções do SINAN; capítulo 10 do manual clínico 2024 |

O RAG serve de apoio documental. A classificação A/B/C/D continua determinada pelas regras e o relatório não constitui notificação oficial ao SINAN. As fontes e seus hashes estão em `data/FONTES.md`.
- Confira no dicionário do DENGBR24 os códigos usados em `dengue/ml/features.py` (`CLASSI_FIN`, `CS_GESTANT`, `NU_IDADE_N`, nomes de colunas) e o separador do CSV (`--sep`).
- Se mudar `EMBED_DIM`, recrie a coluna/índice `vector(...)` e reindexe os documentos.
# Avaliação experimental

O Agente 2 carrega `models/xgboost_dengue.joblib` quando o modelo atinge o
limiar mínimo de ROC-AUC no teste. Para reproduzir o treino usado na aplicação:

```bash
python -m dengue.ml.train_xgboost --csv data/DENGBR24.csv.zip --amostra-aleatoria 500000
```

O escore XGBoost ponderado não é uma probabilidade clínica calibrada. A
classificação A/B/C/D e a interrupção por sinais de alerta continuam seguindo
as regras explícitas do protocolo.

Compare modelos com `python -m dengue.ml.compare --csv data/DENGBR24.csv.zip`.
O script reserva 20% dos registros elegíveis para teste, ajusta hiperparâmetros
por validação cruzada no treino e escolhe o limiar em uma validação separada.
XGBoost, LightGBM e CatBoost entram na comparação quando instalados. O arquivo
`models/comparacao.json` alimenta a página **Avaliação e Desempenho** no Streamlit.
O alvo SINAN (CLASSI_FIN 11/12 versus 10) não equivale aos grupos A/B/C/D.

Para avaliar a aplicação, crie um JSONL com um caso por linha, por exemplo:

```json
{"id":"A001","grupo":"A","alerta":false,"mensagens":["Tenho febre há dois dias"],"campos_esperados":{"evolucao.febre":"sim"}}
```

Execute `python -m dengue.benchmark --casos data/casos_revisados.jsonl`.
Cada caso passa pela conversa, orquestrador, análise, orientações e relatório.
Os 200 casos propostos ainda precisam ser escritos e revisados por profissional
de saúde. A verificação automática de referências confirma apenas que os números
citados existem; fidelidade semântica e qualidade clínica exigem revisão humana.
Nenhum resultado experimental equivale a validação clínica.

