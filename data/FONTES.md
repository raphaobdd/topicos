# Fontes dos dados

## DENGBR24

- Origem: [Portal de Dados Abertos do SUS — Dengue 2024](https://dadosabertos.saude.gov.br/dataset/arboviroses-dengue/resource/4cf2f7b8-9920-41b5-817a-1ffc13e0690c)
- Arquivo: `DENGBR24.csv.zip`, baixado em 23/11/2026; 187.264.443 bytes compactados.
- SHA-256: `1b6b7bf7f6b1fdab4ed2314edc2f57919c33b8103988383f22b7d4776c4a931a`
- O ZIP contém um CSV de 121 colunas e 1.840.398.627 bytes descompactados.
- A coluna `NU_NOTIFIC` não está presente nesse recurso. A carga no Supabase usa hash do arquivo e número da linha como chave de importação; isso não identifica pacientes.
- O arquivo bruto é ignorado pelo controle de versão. O treino reproduzível usa amostra uniforme de 500 mil registros selecionada em todo o arquivo com semente 42.

## Manual de manejo

- Origem: [Ministério da Saúde, Dengue: diagnóstico e manejo clínico, 6ª edição (2024)](https://www.gov.br/saude/pt-br/centrais-de-conteudo/publicacoes/svsa/dengue/dengue-diagnostico-e-manejo-clinico-adulto-e-crianca)
- Arquivo: `docs/ministerio_saude_dengue_manejo_2024.pdf`, baixado em 23/11/2026.
- SHA-256: `2dcb72fb9ebb48efcb473d4b5fd14cf93461a009c628be0671dc064f93ea8998`
- A ingestão de embeddings usa um modelo multilíngue local; o documento sozinho ainda não ativa o RAG. A primeira execução baixa o modelo localmente.

## Demais documentos do RAG

- Fluxograma do Manejo Clínico da Dengue, Ministério da Saúde: [PDF oficial](https://www.gov.br/saude/pt-br/composicao/svsa/coes/arboviroses/publicacoes/fluxograma-do-manejo-clinico-da-dengue.pdf/view). Arquivo `docs/fluxograma_manejo_dengue_2025.pdf`; SHA-256 `b72ca74eda3310c96420267021feb0f8c4fd68444cc72cb1aa8f617a9b7fb220`.
- Dengue — Manual de Enfermagem, edição revisada de 2026, Ministério da Saúde: [PDF](https://mosquito.saude.es.gov.br/Media/dengue/manuais-do-ms/dengue_manual_enfermagem_3ed_rev.pdf). Arquivo `docs/manual_enfermagem_dengue_2026.pdf`; SHA-256 `0d4c377dc12ca91cc87aff602063f3f1574455f47ed3e25947003de395b56e45`.
- Ficha de Notificação/Investigação Dengue e Chikungunya, SINAN: [PDF oficial](https://portalsinan.saude.gov.br/images/documentos/Agravos/Dengue/Ficha_DENGCHIK_FINAL.pdf). Arquivo `docs/sinan_ficha_dengue_chik.pdf`; SHA-256 `aaf0972abf2003984af78798e8a70e2641a6673e1de5d330110d6dc09dbda637`.
- Instruções para preenchimento da ficha do SINAN: [PDF oficial](https://portalsinan.saude.gov.br/images/documentos/Agravos/Dengue/Instrucional_DENGUE_CHIK.pdf). Arquivo `docs/sinan_instrucional_dengue_chik.pdf`; SHA-256 `c4739c2c1be771b0d98f2d9a8709778a4743c838b5091175deeab1475c4b0edb`.
