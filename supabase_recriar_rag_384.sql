-- Use apenas se o projeto Supabase já tiver a tabela RAG antiga com vector(768).
-- Apaga somente os vetores/documentos antigos; reexecute supabase_schema.sql e a ingestão.
drop function if exists match_documentos(vector, integer);
drop table if exists documentos;
