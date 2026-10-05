import json
import re
from dataclasses import dataclass

import asyncpg
from pgvector.asyncpg import register_vector

from app.config import Settings
from app.services.chunking import Chunk


@dataclass
class ExperimentConfig:
    id: int
    chunking_strategy: str
    chunk_size: int
    chunk_overlap: int
    embedding_provider: str
    embedding_model: str
    embedding_dim: int


def _embedding_model_for_provider(settings: Settings, provider: str) -> str:
    field = {
        "openai": "openai_embedding_model",
        "ollama": "ollama_embedding_model",
        "local": "local_embedding_model",
    }[provider]
    return getattr(settings, field)


class Database:
    def __init__(self, dsn: str, readonly_dsn: str | None = None):
        self._dsn = dsn
        self._readonly_dsn = readonly_dsn
        self.pool: asyncpg.Pool | None = None
        self.readonly_pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        # Ensure the `vector` extension exists *before* creating the pool:
        # asyncpg's `init` callback registers the vector type codec on every
        # new connection at pool-creation time, which fails on a fresh
        # database where the type doesn't exist yet.
        conn = await asyncpg.connect(self._dsn)
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        finally:
            await conn.close()
        self.pool = await asyncpg.create_pool(self._dsn, init=register_vector)

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
        if self.readonly_pool is not None:
            await self.readonly_pool.close()

    async def list_tables(self) -> list[str]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT tablename FROM pg_catalog.pg_tables
                WHERE schemaname = 'public'
                ORDER BY tablename
                """
            )
            return [row["tablename"] for row in rows]

    async def migrate(self, settings: Settings) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_configs (
                    id                 SERIAL PRIMARY KEY,
                    chunking_strategy  TEXT NOT NULL,
                    chunk_size         INT NOT NULL,
                    chunk_overlap      INT NOT NULL,
                    embedding_provider TEXT NOT NULL,
                    embedding_model    TEXT NOT NULL,
                    embedding_dim      INT NOT NULL,
                    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    filename TEXT NOT NULL,
                    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_queries (
                    id           SERIAL PRIMARY KEY,
                    query_text   TEXT NOT NULL,
                    gold_answer  TEXT NOT NULL,
                    query_type   TEXT
                )
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_query_gold_pages (
                    query_id     INT NOT NULL REFERENCES eval_queries(id),
                    document_id  UUID NOT NULL REFERENCES documents(id),
                    page_number  INT NOT NULL,
                    PRIMARY KEY (query_id, document_id, page_number)
                )
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_runs (
                    id         SERIAL PRIMARY KEY,
                    config_id  INT NOT NULL REFERENCES experiment_configs(id),
                    results    JSONB NOT NULL,
                    ran_at     TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            # total_chunks/error_count summarize `results` (avoids re-parsing
            # the JSONB for a header/list view); input_size_bytes is the sum
            # of the ingested PDFs' byte sizes; vector_db_size_bytes is a
            # snapshot of the config's chunks_dim_N table's total on-disk
            # size (data + indexes) right after this run, via
            # pg_total_relation_size — not just what this run added.
            await conn.execute(
                "ALTER TABLE ingestion_runs ADD COLUMN IF NOT EXISTS total_chunks INT"
            )
            await conn.execute(
                "ALTER TABLE ingestion_runs ADD COLUMN IF NOT EXISTS total_time_ms INT"
            )
            await conn.execute(
                "ALTER TABLE ingestion_runs ADD COLUMN IF NOT EXISTS error_count INT"
            )
            await conn.execute(
                "ALTER TABLE ingestion_runs ADD COLUMN IF NOT EXISTS input_size_bytes BIGINT"
            )
            await conn.execute(
                "ALTER TABLE ingestion_runs ADD COLUMN IF NOT EXISTS total_pages INT"
            )
            await conn.execute(
                "ALTER TABLE ingestion_runs ADD COLUMN IF NOT EXISTS vector_db_size_bytes BIGINT"
            )
            # A cycle is one "Run Eval" invocation over the whole golden
            # dataset against one config. Individual per-question scores
            # still live in eval_runs (one row per query); eval_runs.cycle_id
            # groups them back into the run that produced them, which is
            # what the Eval History view lists and expands.
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_cycles (
                    id           SERIAL PRIMARY KEY,
                    config_id    INT NOT NULL REFERENCES experiment_configs(id),
                    status       TEXT NOT NULL DEFAULT 'running',
                    total        INT NOT NULL,
                    aggregates   JSONB,
                    started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
                    completed_at TIMESTAMPTZ
                )
                """
            )
            # Recorded per-cycle (not looked up live from settings) so
            # history stays accurate even if EVAL_JUDGE_PROVIDER/MODEL
            # changes later — it's what actually scored this run.
            await conn.execute(
                "ALTER TABLE eval_cycles ADD COLUMN IF NOT EXISTS judge_provider TEXT"
            )
            await conn.execute(
                "ALTER TABLE eval_cycles ADD COLUMN IF NOT EXISTS judge_model TEXT"
            )
            # Soft delete: set instead of removing the row, so a bad click in
            # the UI doesn't permanently destroy eval history the way the
            # manual SQL cleanup path does. list_eval_cycles() filters these out.
            await conn.execute(
                "ALTER TABLE eval_cycles ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ"
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_runs (
                    id                  SERIAL PRIMARY KEY,
                    config_id           INT NOT NULL REFERENCES experiment_configs(id),
                    query_id            INT NOT NULL REFERENCES eval_queries(id),
                    retrieved_chunk_ids UUID[],
                    generated_answer    TEXT,
                    faithfulness        FLOAT,
                    answer_relevance    FLOAT,
                    info_recall         FLOAT,
                    hit_rate            FLOAT,
                    mrr                 FLOAT,
                    eval_latency_ms     INT,
                    run_at              TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            # correctness (LLM judge vs. reference answer) and citation_accuracy
            # (ground-truth-page retrieval precision) alongside the RAGAS-style
            # judge dimensions declared on the table above (faithfulness,
            # answer_relevance, info_recall) plus citation_completeness (does
            # every claim in the answer cite one of the sources it was actually
            # given) below.
            await conn.execute(
                "ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS correctness FLOAT"
            )
            await conn.execute(
                "ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS citation_accuracy FLOAT"
            )
            # Precision/recall counterpart to citation_accuracy -- see
            # docs/eval-metrics.md for how this differs from citation_accuracy,
            # citation_completeness, and info_recall (all easy to conflate).
            await conn.execute(
                "ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS citation_recall FLOAT"
            )
            # Ranked-retrieval diagnostics keyed "<metric>@<k>", e.g.
            # {"recall@5": 0.8, "recall@10": 1.0, "ndcg@10": 0.7}. JSONB rather
            # than named columns so the recall k-set (settings.eval_recall_k_values)
            # and any future metric can change without a schema migration.
            # Started life as recall_at_k (recall only); renamed when nDCG@10
            # joined it. Idempotent: only renames while the old name exists.
            await conn.execute(
                """
                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'eval_runs' AND column_name = 'recall_at_k'
                    ) THEN
                        ALTER TABLE eval_runs RENAME COLUMN recall_at_k TO retrieval_metrics;
                    END IF;
                END
                $$
                """
            )
            await conn.execute(
                "ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS retrieval_metrics JSONB"
            )
            await conn.execute(
                "ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS citation_completeness FLOAT"
            )
            await conn.execute(
                "ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS cycle_id "
                "INT REFERENCES eval_cycles(id)"
            )
            # Cost/usage signals, split by which call spent them: generation_*
            # is the QnA call's own token cost, eval_* is the judge's cost of
            # scoring that answer (summed across all 5 judge calls). These
            # were one combined input_tokens/output_tokens pair until
            # 2026-09-13; the rename below is idempotent (only fires once,
            # when the old names are still present) so old rows' combined
            # totals become eval_* rather than being lost -- generation_*
            # stays NULL for any row recorded before this split existed.
            # llm_call_count is how many LLM calls the row's totals came from.
            async def _rename_if_needed(old: str, new: str) -> None:
                exists_old = await conn.fetchval(
                    "SELECT 1 FROM information_schema.columns "
                    "WHERE table_name = 'eval_runs' AND column_name = $1",
                    old,
                )
                exists_new = await conn.fetchval(
                    "SELECT 1 FROM information_schema.columns "
                    "WHERE table_name = 'eval_runs' AND column_name = $1",
                    new,
                )
                if exists_old and not exists_new:
                    await conn.execute(f"ALTER TABLE eval_runs RENAME COLUMN {old} TO {new}")

            await _rename_if_needed("input_tokens", "eval_input_tokens")
            await _rename_if_needed("output_tokens", "eval_output_tokens")

            await conn.execute(
                "ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS eval_input_tokens INT"
            )
            await conn.execute(
                "ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS eval_output_tokens INT"
            )
            await conn.execute(
                "ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS generation_input_tokens INT"
            )
            await conn.execute(
                "ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS generation_output_tokens INT"
            )
            await conn.execute(
                "ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS llm_call_count INT"
            )

            # Same split, same rename convention, for latency: eval_latency_ms
            # timed only the judge gather, generation_latency_ms only the
            # answer_question() call. The pre-2026-09-13 combined column was
            # named latency_ms (timed from before generation through the end
            # of judging) and renamed to eval_latency_ms for the same reason
            # as the token columns above -- old rows' combined timing becomes
            # eval_latency_ms rather than being lost, generation_latency_ms
            # stays NULL for rows recorded before this split existed.
            await _rename_if_needed("latency_ms", "eval_latency_ms")

            await conn.execute(
                "ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS eval_latency_ms INT"
            )
            await conn.execute(
                "ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS generation_latency_ms INT"
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_entities (
                    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    entity_type  TEXT NOT NULL,
                    entity_text  TEXT NOT NULL,
                    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE (entity_type, entity_text)
                )
                """
            )
            # Generic edge table for both edge types graph RAG phase 1 needs.
            # source_id/target_id are polymorphic by edge_type:
            #   MENTIONS:       source_id = chunk_id,   target_id = entity_id
            #   CO_OCCURS_WITH: source_id = entity_id,  target_id = entity_id
            #                   (unordered pair, canonicalized so
            #                   source_id < target_id -- see derive_cooccurrence_edges)
            # config_id is required for MENTIONS (chunk_id is only unique
            # within a config's own chunks_dim_N table). document_id is
            # required for CO_OCCURS_WITH since that's what it's scoped by.
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_edges (
                    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    edge_type    TEXT NOT NULL,
                    source_id    UUID NOT NULL,
                    target_id    UUID NOT NULL,
                    config_id    INT REFERENCES experiment_configs(id),
                    document_id  UUID REFERENCES documents(id),
                    char_start   INT,
                    char_end     INT,
                    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
                    CHECK (
                        (edge_type = 'MENTIONS' AND config_id IS NOT NULL AND document_id IS NULL)
                        OR
                        (edge_type = 'CO_OCCURS_WITH' AND document_id IS NOT NULL AND config_id IS NULL)
                    )
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS graph_edges_target_idx ON graph_edges (target_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS graph_edges_source_idx ON graph_edges (source_id)"
            )
            await conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS graph_edges_mentions_uniq
                    ON graph_edges (source_id, target_id, config_id)
                    WHERE edge_type = 'MENTIONS'
                """
            )
            await conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS graph_edges_cooccurs_uniq
                    ON graph_edges (source_id, target_id, document_id)
                    WHERE edge_type = 'CO_OCCURS_WITH'
                """
            )
            await conn.execute(
                "ALTER TABLE eval_cycles ADD COLUMN IF NOT EXISTS graph_augmented "
                "BOOLEAN NOT NULL DEFAULT FALSE"
            )
            await conn.execute(
                "ALTER TABLE eval_cycles ADD COLUMN IF NOT EXISTS rerank_model TEXT"
            )
            await conn.execute(
                "ALTER TABLE eval_cycles ADD COLUMN IF NOT EXISTS rerank_candidates INTEGER"
            )
            await conn.execute(
                "ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS rerank_latency_ms INTEGER"
            )

            # RCA agent: one session per (cycle_id, query_id) investigation,
            # persisted (not in-memory) so past investigations are browsable
            # from the RCA tab's sidebar. tool_trace is only ever set on
            # 'assistant' rows -- see rca_agent.run_turn.
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rca_sessions (
                    id          SERIAL PRIMARY KEY,
                    cycle_id    INT NOT NULL REFERENCES eval_cycles(id),
                    query_id    INT NOT NULL REFERENCES eval_queries(id),
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rca_messages (
                    id          SERIAL PRIMARY KEY,
                    session_id  INT NOT NULL REFERENCES rca_sessions(id),
                    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content     TEXT NOT NULL,
                    tool_trace  JSONB,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS rca_messages_session_idx "
                "ON rca_messages (session_id, created_at)"
            )
            # Per-call Anthropic usage for the turn that produced this assistant
            # row (summed across every provider.send() call run_turn made while
            # handling tool-calling round trips) -- NULL on user rows. Powers
            # per-thread token totals in the RCA tab's sidebar.
            await conn.execute(
                "ALTER TABLE rca_messages ADD COLUMN IF NOT EXISTS input_tokens INT"
            )
            await conn.execute(
                "ALTER TABLE rca_messages ADD COLUMN IF NOT EXISTS output_tokens INT"
            )
            # Soft delete, same pattern as eval_cycles.deleted_at: list_rca_sessions()
            # filters these out, but the row and its rca_messages stay in the DB.
            await conn.execute(
                "ALTER TABLE rca_sessions ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ"
            )
            # rca_readonly backs run_readonly_sql (the LLM-authored-SQL escape
            # hatch) -- SELECT-only grants, so even a bypassed app-level check
            # can't write. Password is a fixed dev-harness value (matching
            # this file's existing raguser/ragpass style), not a real secret.
            await conn.execute(
                f"""
                DO $$
                BEGIN
                    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'rca_readonly') THEN
                        CREATE ROLE rca_readonly LOGIN PASSWORD '{settings.rca_readonly_password}';
                    END IF;
                END
                $$
                """
            )
            await conn.execute("GRANT CONNECT ON DATABASE tagore TO rca_readonly")
            await conn.execute("GRANT USAGE ON SCHEMA public TO rca_readonly")
            await conn.execute("GRANT SELECT ON ALL TABLES IN SCHEMA public TO rca_readonly")
            await conn.execute(
                "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO rca_readonly"
            )
            if self._readonly_dsn and self.readonly_pool is None:
                self.readonly_pool = await asyncpg.create_pool(self._readonly_dsn)

            existing = await conn.fetchval("SELECT COUNT(*) FROM experiment_configs")
            if existing == 0:
                provider = settings.embedding_provider
                await conn.execute(
                    """
                    INSERT INTO experiment_configs
                        (chunking_strategy, chunk_size, chunk_overlap,
                         embedding_provider, embedding_model, embedding_dim)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    "sliding_window",
                    1000,
                    200,
                    provider,
                    _embedding_model_for_provider(settings, provider),
                    settings.embedding_dim,
                )

        await self.ensure_chunks_table(settings.embedding_dim)

    async def ensure_chunks_table(self, dim: int) -> None:
        dim = int(dim)
        if dim <= 0:
            raise ValueError(f"embedding dim must be positive, got {dim!r}")
        table = f"chunks_dim_{dim}"
        async with self.pool.acquire() as conn:
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    config_id    INT NOT NULL REFERENCES experiment_configs(id),
                    document_id  UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    page_number  INT NOT NULL,
                    chunk_text   TEXT NOT NULL,
                    embedding    VECTOR({dim}) NOT NULL
                )
                """
            )
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS {table}_config_id_idx ON {table} (config_id)"
            )
            # Set only for sentence_window chunks: the sentence padded with
            # surrounding context (see chunking.py's Chunk.context_text).
            # NULL for every other strategy -- chat.py falls back to
            # chunk_text when this is NULL.
            await conn.execute(
                f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS context_text TEXT"
            )

    async def create_experiment_config(
        self,
        chunking_strategy: str,
        chunk_size: int,
        chunk_overlap: int,
        embedding_provider: str,
        embedding_model: str,
        embedding_dim: int,
    ) -> ExperimentConfig:
        await self.ensure_chunks_table(embedding_dim)
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO experiment_configs
                    (chunking_strategy, chunk_size, chunk_overlap,
                     embedding_provider, embedding_model, embedding_dim)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id, chunking_strategy, chunk_size, chunk_overlap,
                          embedding_provider, embedding_model, embedding_dim
                """,
                chunking_strategy,
                chunk_size,
                chunk_overlap,
                embedding_provider,
                embedding_model,
                embedding_dim,
            )
            return ExperimentConfig(**dict(row))

    async def get_experiment_config(self, config_id: int) -> ExperimentConfig | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, chunking_strategy, chunk_size, chunk_overlap,
                       embedding_provider, embedding_model, embedding_dim
                FROM experiment_configs WHERE id = $1
                """,
                config_id,
            )
            return ExperimentConfig(**dict(row)) if row else None

    async def list_experiment_configs(self) -> list[ExperimentConfig]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, chunking_strategy, chunk_size, chunk_overlap,
                       embedding_provider, embedding_model, embedding_dim
                FROM experiment_configs ORDER BY id
                """
            )
            return [ExperimentConfig(**dict(row)) for row in rows]

    async def find_experiment_config(
        self,
        chunking_strategy: str,
        chunk_size: int,
        chunk_overlap: int,
        embedding_provider: str,
        embedding_model: str,
        embedding_dim: int,
    ) -> ExperimentConfig | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, chunking_strategy, chunk_size, chunk_overlap,
                       embedding_provider, embedding_model, embedding_dim
                FROM experiment_configs
                WHERE chunking_strategy = $1 AND chunk_size = $2 AND chunk_overlap = $3
                  AND embedding_provider = $4 AND embedding_model = $5 AND embedding_dim = $6
                """,
                chunking_strategy,
                chunk_size,
                chunk_overlap,
                embedding_provider,
                embedding_model,
                embedding_dim,
            )
            return ExperimentConfig(**dict(row)) if row else None

    async def delete_chunks_for_config(self, config_id: int, dim: int) -> None:
        table = f"chunks_dim_{int(dim)}"
        async with self.pool.acquire() as conn:
            await conn.execute(f"DELETE FROM {table} WHERE config_id = $1", config_id)

    async def record_ingestion_run(
        self,
        config_id: int,
        results: list[dict],
        total_chunks: int,
        total_time_ms: int,
        error_count: int,
        input_size_bytes: int,
        vector_db_size_bytes: int,
        total_pages: int = 0,
    ) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO ingestion_runs
                    (config_id, results, total_chunks, total_time_ms, error_count,
                     input_size_bytes, vector_db_size_bytes, total_pages)
                VALUES ($1, $2::jsonb, $3, $4, $5, $6, $7, $8)
                """,
                config_id,
                json.dumps(results),
                total_chunks,
                total_time_ms,
                error_count,
                input_size_bytes,
                vector_db_size_bytes,
                total_pages,
            )

    async def get_config_vector_storage_bytes(self, config_id: int, dim: int) -> int:
        """On-disk bytes of just this config's rows in chunks_dim_{dim} — not
        the whole table, which is shared across every config at that
        embedding dimension. Excludes index storage (the HNSW index isn't
        separable per config_id), so this is row/vector data only."""
        table = f"chunks_dim_{int(dim)}"
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                f"SELECT COALESCE(SUM(pg_column_size(t.*)), 0) FROM {table} t WHERE config_id = $1",
                config_id,
            )

    async def list_ingestion_runs(self) -> list[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT r.id, r.ran_at, r.results, r.total_chunks, r.total_time_ms,
                       r.error_count, r.input_size_bytes, r.vector_db_size_bytes, r.total_pages,
                       c.id AS config_id, c.chunking_strategy, c.chunk_size, c.chunk_overlap,
                       c.embedding_provider, c.embedding_model, c.embedding_dim
                FROM ingestion_runs r
                JOIN experiment_configs c ON c.id = r.config_id
                ORDER BY r.ran_at DESC
                """
            )
            return [
                {
                    "id": row["id"],
                    "ran_at": row["ran_at"],
                    "results": json.loads(row["results"]),
                    "total_chunks": row["total_chunks"],
                    "total_time_ms": row["total_time_ms"],
                    "error_count": row["error_count"],
                    "input_size_bytes": row["input_size_bytes"],
                    "vector_db_size_bytes": row["vector_db_size_bytes"],
                    "total_pages": row["total_pages"],
                    "config": {
                        "id": row["config_id"],
                        "chunking_strategy": row["chunking_strategy"],
                        "chunk_size": row["chunk_size"],
                        "chunk_overlap": row["chunk_overlap"],
                        "embedding_provider": row["embedding_provider"],
                        "embedding_model": row["embedding_model"],
                        "embedding_dim": row["embedding_dim"],
                    },
                }
                for row in rows
            ]

    async def get_default_config_id(self) -> int:
        async with self.pool.acquire() as conn:
            return await conn.fetchval("SELECT MIN(id) FROM experiment_configs")

    async def insert_document(self, filename: str) -> str:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO documents (filename) VALUES ($1) RETURNING id", filename
            )
            return str(row["id"])

    async def get_document_by_filename(self, filename: str) -> str | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id FROM documents WHERE filename = $1 ORDER BY uploaded_at DESC LIMIT 1",
                filename,
            )
            return str(row["id"]) if row else None

    async def count_chunks_for_document(self, config_id: int, document_id: str, dim: int) -> int:
        table = f"chunks_dim_{int(dim)}"
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                f"SELECT COUNT(*) FROM {table} WHERE config_id = $1 AND document_id = $2",
                config_id,
                document_id,
            )

    async def insert_chunks(
        self,
        document_id: str,
        config_id: int,
        dim: int,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        table = f"chunks_dim_{int(dim)}"
        async with self.pool.acquire() as conn:
            await conn.executemany(
                f"""
                INSERT INTO {table}
                    (config_id, document_id, page_number, chunk_text, context_text, embedding)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                [
                    (config_id, document_id, c.page_number, c.text, c.context_text, emb)
                    for c, emb in zip(chunks, embeddings)
                ],
            )

    async def search_similar_chunks(
        self, embedding: list[float], config_id: int, dim: int, top_k: int
    ):
        table = f"chunks_dim_{int(dim)}"
        async with self.pool.acquire() as conn:
            return await conn.fetch(
                f"""
                SELECT c.id, c.chunk_text, c.context_text, c.page_number, d.filename,
                       c.embedding <=> $2 AS distance
                FROM {table} c
                JOIN documents d ON d.id = c.document_id
                WHERE c.config_id = $1
                ORDER BY distance
                LIMIT $3
                """,
                config_id,
                embedding,
                top_k,
            )

    async def insert_entity(self, entity_type: str, normalized_text: str) -> str:
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                """
                INSERT INTO graph_entities (entity_type, entity_text)
                VALUES ($1, $2)
                ON CONFLICT (entity_type, entity_text)
                    DO UPDATE SET entity_type = EXCLUDED.entity_type
                RETURNING id
                """,
                entity_type,
                normalized_text,
            )

    async def insert_mention(
        self, config_id: int, chunk_id: str, entity_id: str,
        char_start: int | None, char_end: int | None,
    ) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO graph_edges
                    (edge_type, source_id, target_id, config_id, char_start, char_end)
                VALUES ('MENTIONS', $1, $2, $3, $4, $5)
                ON CONFLICT (source_id, target_id, config_id) WHERE edge_type = 'MENTIONS'
                    DO NOTHING
                """,
                chunk_id,
                entity_id,
                config_id,
                char_start,
                char_end,
            )

    async def derive_cooccurrence_edges(self, config_id: int, dim: int, document_id: str) -> int:
        """Links every pair of distinct entities that each have a MENTIONS
        edge to some chunk of this document, under this config. Call once per
        document after all its chunks' mentions have been inserted."""
        table = f"chunks_dim_{int(dim)}"
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                INSERT INTO graph_edges (edge_type, source_id, target_id, document_id)
                SELECT DISTINCT 'CO_OCCURS_WITH',
                       LEAST(e1.target_id, e2.target_id),
                       GREATEST(e1.target_id, e2.target_id),
                       $2::uuid
                FROM graph_edges e1
                JOIN {table} c1 ON c1.id = e1.source_id
                JOIN graph_edges e2 ON e2.config_id = e1.config_id
                JOIN {table} c2 ON c2.id = e2.source_id
                WHERE e1.edge_type = 'MENTIONS' AND e2.edge_type = 'MENTIONS'
                  AND e1.config_id = $1
                  AND c1.document_id = $2::uuid AND c2.document_id = $2::uuid
                  AND e1.target_id != e2.target_id
                ON CONFLICT (source_id, target_id, document_id) WHERE edge_type = 'CO_OCCURS_WITH'
                    DO NOTHING
                RETURNING id
                """,
                config_id,
                document_id,
            )
            return len(rows)

    async def find_entities(self, pairs: list[tuple[str, str]]) -> list[str]:
        if not pairs:
            return []
        types = [p[0] for p in pairs]
        texts = [p[1] for p in pairs]
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id FROM graph_entities
                WHERE (entity_type, entity_text) IN (
                    SELECT * FROM unnest($1::text[], $2::text[])
                )
                """,
                types,
                texts,
            )
            return [str(r["id"]) for r in rows]

    async def get_chunks_mentioning_entities(
        self, config_id: int, dim: int, entity_ids: list[str],
        embedding: list[float], limit: int,
    ) -> list[dict]:
        if not entity_ids:
            return []
        table = f"chunks_dim_{int(dim)}"
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT DISTINCT c.id, c.chunk_text, c.context_text, c.page_number, d.filename,
                       c.embedding <=> $4 AS distance
                FROM graph_edges e
                JOIN {table} c ON c.id = e.source_id
                JOIN documents d ON d.id = c.document_id
                WHERE e.edge_type = 'MENTIONS' AND e.config_id = $1
                  AND e.target_id = ANY($2::uuid[])
                ORDER BY distance
                LIMIT $3
                """,
                config_id,
                entity_ids,
                limit,
                embedding,
            )
            return [dict(row) for row in rows]

    async def list_graph_entities(
        self, config_id: int, dim: int, document_id: str, page: int, page_size: int,
    ) -> tuple[list[dict], int]:
        table = f"chunks_dim_{int(dim)}"
        async with self.pool.acquire() as conn:
            total = await conn.fetchval(
                f"""
                SELECT COUNT(*)
                FROM graph_edges e
                JOIN {table} c ON c.id = e.source_id
                WHERE e.edge_type = 'MENTIONS' AND e.config_id = $1 AND c.document_id = $2
                """,
                config_id,
                document_id,
            )
            rows = await conn.fetch(
                f"""
                SELECT ge.entity_type, ge.entity_text, c.id AS chunk_id, c.page_number,
                       d.filename
                FROM graph_edges e
                JOIN {table} c ON c.id = e.source_id
                JOIN graph_entities ge ON ge.id = e.target_id
                JOIN documents d ON d.id = c.document_id
                WHERE e.edge_type = 'MENTIONS' AND e.config_id = $1 AND c.document_id = $2
                ORDER BY ge.entity_type, ge.entity_text, c.page_number
                LIMIT $3 OFFSET $4
                """,
                config_id,
                document_id,
                page_size,
                (page - 1) * page_size,
            )
            return [dict(row) for row in rows], total

    async def list_cooccurrences(
        self, document_id: str, entity_type: str | None, page: int, page_size: int,
    ) -> tuple[list[dict], int]:
        conditions = ["e.edge_type = 'CO_OCCURS_WITH'", "e.document_id = $1"]
        params: list = [document_id]
        if entity_type:
            params.append(entity_type)
            conditions.append(f"(ea.entity_type = ${len(params)} OR eb.entity_type = ${len(params)})")
        where_clause = " AND ".join(conditions)

        async with self.pool.acquire() as conn:
            total = await conn.fetchval(
                f"""
                SELECT COUNT(*)
                FROM graph_edges e
                JOIN graph_entities ea ON ea.id = e.source_id
                JOIN graph_entities eb ON eb.id = e.target_id
                WHERE {where_clause}
                """,
                *params,
            )
            limit_params = [*params, page_size, (page - 1) * page_size]
            rows = await conn.fetch(
                f"""
                SELECT ea.entity_type AS entity_type_a, ea.entity_text AS entity_text_a,
                       eb.entity_type AS entity_type_b, eb.entity_text AS entity_text_b
                FROM graph_edges e
                JOIN graph_entities ea ON ea.id = e.source_id
                JOIN graph_entities eb ON eb.id = e.target_id
                WHERE {where_clause}
                ORDER BY ea.entity_type, ea.entity_text
                LIMIT ${len(limit_params) - 1} OFFSET ${len(limit_params)}
                """,
                *limit_params,
            )
            return [dict(row) for row in rows], total

    async def list_documents(self):
        async with self.pool.acquire() as conn:
            return await conn.fetch(
                "SELECT id, filename, uploaded_at FROM documents ORDER BY uploaded_at DESC"
            )

    async def find_or_create_eval_query(
        self, query_text: str, gold_answer: str, query_type: str | None
    ) -> int:
        async with self.pool.acquire() as conn:
            existing = await conn.fetchval(
                "SELECT id FROM eval_queries WHERE query_text = $1", query_text
            )
            if existing is not None:
                return existing
            return await conn.fetchval(
                """
                INSERT INTO eval_queries (query_text, gold_answer, query_type)
                VALUES ($1, $2, $3)
                RETURNING id
                """,
                query_text,
                gold_answer,
                query_type,
            )

    async def ensure_gold_pages(self, query_id: int, pages: list[tuple[str, int]]) -> None:
        """Insert (query_id, document_id, page_number) rows for gold pages whose
        document has already been ingested. A gold page naming a document
        that hasn't been ingested yet is silently skipped — it can't satisfy
        the eval_query_gold_pages foreign key until that document exists."""
        async with self.pool.acquire() as conn:
            for filename, page_number in pages:
                document_id = await conn.fetchval(
                    "SELECT id FROM documents WHERE filename = $1 "
                    "ORDER BY uploaded_at DESC LIMIT 1",
                    filename,
                )
                if document_id is None:
                    continue
                await conn.execute(
                    """
                    INSERT INTO eval_query_gold_pages (query_id, document_id, page_number)
                    VALUES ($1, $2, $3)
                    ON CONFLICT DO NOTHING
                    """,
                    query_id,
                    document_id,
                    page_number,
                )

    async def record_eval_run(
        self,
        config_id: int,
        query_id: int,
        generated_answer: str,
        hit_rate: float,
        mrr: float,
        citation_accuracy: float,
        correctness: float,
        generation_latency_ms: int,
        eval_latency_ms: int,
        cycle_id: int | None = None,
        citation_recall: float | None = None,
        retrieval_metrics: dict | None = None,
        faithfulness: float | None = None,
        answer_relevance: float | None = None,
        info_recall: float | None = None,
        citation_completeness: float | None = None,
        generation_input_tokens: int | None = None,
        generation_output_tokens: int | None = None,
        eval_input_tokens: int | None = None,
        eval_output_tokens: int | None = None,
        llm_call_count: int | None = None,
        retrieved_chunk_ids: list[str] | None = None,
        rerank_latency_ms: int | None = None,
    ) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO eval_runs
                    (config_id, query_id, generated_answer, hit_rate, mrr,
                     citation_accuracy, correctness, generation_latency_ms, eval_latency_ms,
                     cycle_id, citation_recall, retrieval_metrics, faithfulness, answer_relevance,
                     info_recall, citation_completeness,
                     generation_input_tokens, generation_output_tokens,
                     eval_input_tokens, eval_output_tokens, llm_call_count, retrieved_chunk_ids,
                     rerank_latency_ms)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12::jsonb, $13, $14, $15,
                        $16, $17, $18, $19, $20, $21, $22, $23)
                """,
                config_id,
                query_id,
                generated_answer,
                hit_rate,
                mrr,
                citation_accuracy,
                correctness,
                generation_latency_ms,
                eval_latency_ms,
                cycle_id,
                citation_recall,
                json.dumps(retrieval_metrics) if retrieval_metrics is not None else None,
                faithfulness,
                answer_relevance,
                info_recall,
                citation_completeness,
                generation_input_tokens,
                generation_output_tokens,
                eval_input_tokens,
                eval_output_tokens,
                llm_call_count,
                retrieved_chunk_ids,
                rerank_latency_ms,
            )

    async def create_eval_cycle(
        self, config_id: int, total: int, judge_provider: str, judge_model: str,
        graph_augmented: bool = False,
        rerank_model: str | None = None, rerank_candidates: int | None = None,
    ) -> int:
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                """
                INSERT INTO eval_cycles
                    (config_id, status, total, judge_provider, judge_model, graph_augmented,
                     rerank_model, rerank_candidates)
                VALUES ($1, 'running', $2, $3, $4, $5, $6, $7)
                RETURNING id
                """,
                config_id,
                total,
                judge_provider,
                judge_model,
                graph_augmented,
                rerank_model,
                rerank_candidates,
            )

    async def abort_running_eval_cycles(self) -> int:
        """Called on backend startup: any cycle still 'running' belonged to
        a process that no longer exists (it died with the previous restart,
        mid-run) — nothing will ever call complete_eval_cycle on it, so it
        would otherwise sit showing 'running' forever."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                UPDATE eval_cycles
                SET status = 'aborted', completed_at = now()
                WHERE status = 'running'
                RETURNING id
                """
            )
            return len(rows)

    async def complete_eval_cycle(self, cycle_id: int, status: str, aggregates: dict) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE eval_cycles
                SET status = $2, aggregates = $3::jsonb, completed_at = now()
                WHERE id = $1
                """,
                cycle_id,
                status,
                json.dumps(aggregates),
            )

    async def list_eval_cycles(self) -> list[dict]:
        async with self.pool.acquire() as conn:
            cycles = await conn.fetch(
                """
                SELECT cy.id, cy.status, cy.total, cy.aggregates, cy.started_at, cy.completed_at,
                       cy.judge_provider, cy.judge_model, cy.graph_augmented,
                       cy.rerank_model, cy.rerank_candidates,
                       c.id AS config_id, c.chunking_strategy, c.chunk_size, c.chunk_overlap,
                       c.embedding_provider, c.embedding_model, c.embedding_dim
                FROM eval_cycles cy
                JOIN experiment_configs c ON c.id = cy.config_id
                WHERE cy.deleted_at IS NULL
                ORDER BY cy.started_at DESC
                """
            )

            results = []
            for cycle in cycles:
                per_query = await conn.fetch(
                    """
                    SELECT r.id, q.query_text AS question, r.generated_answer,
                           r.hit_rate, r.mrr, r.citation_accuracy, r.citation_recall,
                           r.retrieval_metrics, r.correctness,
                           r.faithfulness, r.answer_relevance, r.info_recall,
                           r.citation_completeness,
                           r.generation_latency_ms, r.eval_latency_ms,
                           r.generation_input_tokens, r.generation_output_tokens,
                           r.eval_input_tokens, r.eval_output_tokens, r.llm_call_count,
                           r.rerank_latency_ms, r.run_at
                    FROM eval_runs r
                    JOIN eval_queries q ON q.id = r.query_id
                    WHERE r.cycle_id = $1
                    ORDER BY r.run_at
                    """,
                    cycle["id"],
                )
                results.append({
                    "id": cycle["id"],
                    "status": cycle["status"],
                    "total": cycle["total"],
                    "aggregates": json.loads(cycle["aggregates"]) if cycle["aggregates"] else None,
                    "started_at": cycle["started_at"],
                    "completed_at": cycle["completed_at"],
                    "judge_provider": cycle["judge_provider"],
                    "judge_model": cycle["judge_model"],
                    "graph_augmented": cycle["graph_augmented"],
                    "rerank_model": cycle["rerank_model"],
                    "rerank_candidates": cycle["rerank_candidates"],
                    "config": {
                        "id": cycle["config_id"],
                        "chunking_strategy": cycle["chunking_strategy"],
                        "chunk_size": cycle["chunk_size"],
                        "chunk_overlap": cycle["chunk_overlap"],
                        "embedding_provider": cycle["embedding_provider"],
                        "embedding_model": cycle["embedding_model"],
                        "embedding_dim": cycle["embedding_dim"],
                    },
                    "per_query": [
                        {
                            **dict(r),
                            "retrieval_metrics": (
                                json.loads(r["retrieval_metrics"]) if r["retrieval_metrics"] else None
                            ),
                        }
                        for r in per_query
                    ],
                })
            return results

    async def soft_delete_eval_cycle(self, cycle_id: int) -> bool:
        """Mark a cycle deleted instead of removing it -- list_eval_cycles()
        filters it out, but the row (and its eval_runs) stay in the DB.
        Returns False if the cycle doesn't exist or was already deleted."""
        async with self.pool.acquire() as conn:
            result = await conn.fetchval(
                """
                UPDATE eval_cycles SET deleted_at = now()
                WHERE id = $1 AND deleted_at IS NULL
                RETURNING id
                """,
                cycle_id,
            )
            return result is not None

    async def hard_delete_eval_cycle(self, cycle_id: int) -> bool:
        """Permanently remove a cycle and its eval_runs rows. Unlike
        soft_delete_eval_cycle, this cannot be undone -- eval_runs.cycle_id
        has no ON DELETE CASCADE, so eval_runs must be deleted first."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM eval_runs WHERE cycle_id = $1", cycle_id)
                result = await conn.fetchval(
                    "DELETE FROM eval_cycles WHERE id = $1 RETURNING id", cycle_id
                )
                return result is not None

    async def get_eval_cycle_summary(self, cycle_id: int) -> dict | None:
        """Just enough to start a reval: which config it ran against and how
        many questions -- not the full per_query detail list_eval_cycles()
        fetches for every cycle."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, config_id, total, status
                FROM eval_cycles
                WHERE id = $1 AND deleted_at IS NULL
                """,
                cycle_id,
            )
            return dict(row) if row else None

    async def get_cycle_reval_source(self, cycle_id: int) -> list[dict]:
        """Everything needed to re-judge a historical cycle's questions
        without re-running retrieval or generation: question/reference text,
        the generated answer, retrieval-level metrics to carry forward
        unchanged, and retrieved_chunk_ids to reconstruct context/source
        labels from (see resolve_chunks)."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT r.query_id, q.query_text AS question, q.gold_answer AS reference_answer,
                       r.generated_answer, r.hit_rate, r.mrr, r.citation_accuracy,
                       r.citation_recall, r.retrieval_metrics, r.retrieved_chunk_ids,
                       r.generation_latency_ms, r.generation_input_tokens,
                       r.generation_output_tokens, r.rerank_latency_ms
                FROM eval_runs r
                JOIN eval_queries q ON q.id = r.query_id
                WHERE r.cycle_id = $1
                ORDER BY r.run_at
                """,
                cycle_id,
            )
            return [
                {
                    **dict(row),
                    "retrieval_metrics": (
                        json.loads(row["retrieval_metrics"]) if row["retrieval_metrics"] else None
                    ),
                }
                for row in rows
            ]

    async def get_eval_run_detail(self, cycle_id: int, query_id: int) -> dict | None:
        """Everything RCA's get_eval_run tool needs in one call: cycle/config
        info (including embedding_dim, needed to resolve retrieved_chunk_ids
        later), the question + gold answer, and the eval_runs row if one
        exists for this (cycle_id, query_id) pair."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT cy.id AS cycle_id, cy.config_id, cy.graph_augmented, cy.status,
                       c.embedding_dim,
                       q.query_text AS question, q.gold_answer,
                       r.id AS run_id, r.generated_answer, r.retrieved_chunk_ids,
                       r.hit_rate, r.mrr, r.citation_accuracy, r.citation_recall,
                       r.correctness, r.faithfulness, r.answer_relevance, r.info_recall
                FROM eval_cycles cy
                JOIN experiment_configs c ON c.id = cy.config_id
                JOIN eval_queries q ON q.id = $2
                LEFT JOIN eval_runs r ON r.cycle_id = cy.id AND r.query_id = $2
                WHERE cy.id = $1 AND cy.deleted_at IS NULL
                """,
                cycle_id,
                query_id,
            )
            return dict(row) if row else None

    async def get_gold_pages(self, query_id: int) -> list[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT d.filename, g.page_number
                FROM eval_query_gold_pages g
                JOIN documents d ON d.id = g.document_id
                WHERE g.query_id = $1
                ORDER BY d.filename, g.page_number
                """,
                query_id,
            )
            return [dict(row) for row in rows]

    async def list_eval_queries(self) -> list[dict]:
        """Populates the RCA tab's Question dropdown."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, query_text, query_type FROM eval_queries ORDER BY id"
            )
            return [dict(row) for row in rows]

    async def create_rca_session(self, cycle_id: int, query_id: int) -> int:
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                "INSERT INTO rca_sessions (cycle_id, query_id) VALUES ($1, $2) RETURNING id",
                cycle_id,
                query_id,
            )

    async def get_rca_session(self, session_id: int) -> dict | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, cycle_id, query_id, created_at FROM rca_sessions "
                "WHERE id = $1 AND deleted_at IS NULL",
                session_id,
            )
            return dict(row) if row else None

    async def list_rca_sessions(self) -> list[dict]:
        """total_input_tokens/total_output_tokens sum every assistant message's
        usage for that session (NULL-safe via COALESCE(SUM(...), 0)) -- powers
        the RCA tab sidebar's per-thread token totals."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT s.id, s.cycle_id, s.query_id, s.created_at,
                       COALESCE(SUM(m.input_tokens), 0) AS total_input_tokens,
                       COALESCE(SUM(m.output_tokens), 0) AS total_output_tokens
                FROM rca_sessions s
                LEFT JOIN rca_messages m ON m.session_id = s.id
                WHERE s.deleted_at IS NULL
                GROUP BY s.id
                ORDER BY s.created_at DESC
                """
            )
            return [dict(row) for row in rows]

    async def soft_delete_rca_session(self, session_id: int) -> bool:
        """Mark a session deleted instead of removing it -- list_rca_sessions()
        filters it out, but the row (and its rca_messages) stay in the DB.
        Returns False if the session doesn't exist or was already deleted."""
        async with self.pool.acquire() as conn:
            result = await conn.fetchval(
                """
                UPDATE rca_sessions SET deleted_at = now()
                WHERE id = $1 AND deleted_at IS NULL
                RETURNING id
                """,
                session_id,
            )
            return result is not None

    async def insert_rca_message(
        self, session_id: int, role: str, content: str, tool_trace: list | None,
        input_tokens: int | None = None, output_tokens: int | None = None,
    ) -> int:
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                """
                INSERT INTO rca_messages
                    (session_id, role, content, tool_trace, input_tokens, output_tokens)
                VALUES ($1, $2, $3, $4::jsonb, $5, $6)
                RETURNING id
                """,
                session_id,
                role,
                content,
                json.dumps(tool_trace) if tool_trace is not None else None,
                input_tokens,
                output_tokens,
            )

    async def list_rca_messages(self, session_id: int) -> list[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, role, content, tool_trace, input_tokens, output_tokens, created_at
                FROM rca_messages
                WHERE session_id = $1
                ORDER BY created_at
                """,
                session_id,
            )
            return [
                {
                    **dict(row),
                    "tool_trace": json.loads(row["tool_trace"]) if row["tool_trace"] else None,
                }
                for row in rows
            ]

    async def execute_readonly(self, query: str) -> list[dict]:
        """Runs LLM-authored SQL as the rca_readonly role (SELECT-only Postgres
        grants). Belt-and-suspenders on top of the role restriction: rejects
        anything not starting with SELECT/WITH, wraps the query so a result
        cap applies regardless of what the query itself asked for, and bounds
        execution time so a runaway query can't hang the conversation."""
        stripped = query.strip().rstrip(";")
        if not re.match(r"(?is)^\s*(SELECT|WITH)\b", stripped):
            raise ValueError("Only SELECT/WITH statements are allowed")
        async with self.readonly_pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SET LOCAL statement_timeout = '5000'")
                rows = await conn.fetch(f"SELECT * FROM ({stripped}) AS _rca_query LIMIT 200")
                return [dict(row) for row in rows]

    async def resolve_chunks(self, chunk_ids: list, dim: int) -> dict:
        """Batch-resolve chunk ids back to their text/filename/page for
        reval. Any id no longer present in chunks_dim_N (its document was
        re-ingested since the original run, which deletes and regenerates
        chunk rows with new ids) is simply absent from the result -- callers
        detect staleness by comparing the requested count to what came back."""
        if not chunk_ids:
            return {}
        table = f"chunks_dim_{int(dim)}"
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT c.id, c.chunk_text, c.context_text, c.page_number, d.filename
                FROM {table} c
                JOIN documents d ON d.id = c.document_id
                WHERE c.id = ANY($1::uuid[])
                """,
                chunk_ids,
            )
            return {str(row["id"]): dict(row) for row in rows}

    async def list_chunk_documents(self, config_id: int, dim: int) -> list[dict]:
        """Documents actually ingested under this config -- populates the
        Chunks tab's document filter dropdown."""
        table = f"chunks_dim_{int(dim)}"
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT DISTINCT d.id AS document_id, d.filename
                FROM {table} c
                JOIN documents d ON d.id = c.document_id
                WHERE c.config_id = $1
                ORDER BY d.filename
                """,
                config_id,
            )
            return [dict(row) for row in rows]

    async def list_chunks(
        self,
        config_id: int,
        dim: int,
        document_id: str | None,
        search: str | None,
        page: int,
        page_size: int,
    ) -> tuple[list[dict], int]:
        """Paginated, filtered view of a config's stored chunks -- backs the
        Chunks tab, used for inspecting what a chunking strategy actually
        produced (e.g. verifying sentence_window's context_text, or
        checking a fixed_token boundary) rather than for the retrieval path."""
        table = f"chunks_dim_{int(dim)}"
        conditions = ["c.config_id = $1"]
        params: list = [config_id]
        if document_id:
            params.append(document_id)
            conditions.append(f"c.document_id = ${len(params)}::uuid")
        if search:
            params.append(f"%{search}%")
            conditions.append(f"c.chunk_text ILIKE ${len(params)}")
        where_clause = " AND ".join(conditions)

        async with self.pool.acquire() as conn:
            total = await conn.fetchval(
                f"SELECT COUNT(*) FROM {table} c WHERE {where_clause}", *params
            )
            limit_params = [*params, page_size, (page - 1) * page_size]
            rows = await conn.fetch(
                f"""
                SELECT c.id, c.page_number, c.chunk_text, c.context_text, d.filename
                FROM {table} c
                JOIN documents d ON d.id = c.document_id
                WHERE {where_clause}
                ORDER BY d.filename, c.page_number, c.id
                LIMIT ${len(limit_params) - 1} OFFSET ${len(limit_params)}
                """,
                *limit_params,
            )
            return [dict(row) for row in rows], total
