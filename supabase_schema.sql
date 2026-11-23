-- Execute no SQL Editor do Supabase (uma vez).
-- A dimensão do vetor deve ser igual a EMBED_DIM do .env (padrão: 384).

create extension if not exists vector;

-- Sessões de atendimento: dados estruturados, mensagens, análise, orientações e relatório
create table if not exists sessoes (
  id            uuid primary key,
  criada_em     timestamptz not null default now(),
  interrompido  boolean     not null default false,
  dados         jsonb       not null,
  mensagens     jsonb       not null default '[]'::jsonb,
  seguranca     jsonb,
  analise       jsonb,
  orientacoes   jsonb,
  relatorio_md  text
);

-- Base de conhecimento do RAG
create table if not exists documentos (
  id         bigserial primary key,
  fonte      text not null,
  pagina     int,
  conteudo   text not null,
  embedding  vector(384) not null,
  criado_em  timestamptz not null default now()
);

create index if not exists documentos_fonte_idx on documentos (fonte);
create index if not exists documentos_embedding_idx
  on documentos using hnsw (embedding vector_cosine_ops);

-- Busca por similaridade de cosseno
create or replace function match_documentos(
  query_embedding vector(384),
  match_count int default 4
)
returns table (id bigint, fonte text, pagina int, conteudo text, similaridade float)
language sql stable
as $$
  select d.id, d.fonte, d.pagina, d.conteudo,
         1 - (d.embedding <=> query_embedding) as similaridade
  from documentos d
  order by d.embedding <=> query_embedding
  limit match_count;
$$;

-- Segurança: dados de saúde. RLS ligado e SEM políticas = só a chave service_role
-- (usada apenas no backend/Streamlit server) consegue ler/escrever.
alter table sessoes    enable row level security;
alter table documentos enable row level security;

-- Importacao epidemiologica: somente campos usados na exploracao e no modelo.
-- Todos os valores de origem ficam como texto; a conversao ocorre no pipeline Python.
create table if not exists sinan_dengue_2024 (
  fonte_sha256 text not null,
  linha_arquivo bigint not null,
  "AUTO_IMUNE" text, "ARTRALGIA" text, "CEFALEIA" text,
  "CLASSI_FIN" text, "CS_GESTANT" text, "CS_SEXO" text,
  "DIABETES" text, "DOR_RETRO" text, "DT_NOTIFIC" text,
  "DT_SIN_PRI" text, "EXANTEMA" text, "FEBRE" text,
  "HEMATOLOG" text, "HEPATOPAT" text, "HIPERTENSA" text,
  "ID_MUNICIP" text, "MIALGIA" text, "NAUSEA" text,
  "NU_IDADE_N" text, "NU_NOTIFIC" text, "RENAL" text,
  "SG_UF_NOT" text, "VOMITO" text,
  primary key (fonte_sha256, linha_arquivo)
);
create index if not exists sinan_dengue_2024_classificacao_idx
  on sinan_dengue_2024 ("CLASSI_FIN");
alter table sinan_dengue_2024 enable row level security;
