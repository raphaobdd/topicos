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
