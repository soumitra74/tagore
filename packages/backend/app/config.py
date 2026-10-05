from functools import lru_cache

from pydantic import BaseModel, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TierSettings(BaseModel):
    """One model target for DynamicRouter (see app/routing/router.py Tier)."""

    name: str
    model: str
    base_url: str
    api_key: str | None = None
    temperature: float = 0.0
    cost_per_1k_in: float = 0.0
    cost_per_1k_out: float = 0.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://raguser:ragpass@localhost:15432/tagore"

    chat_provider: str = "anthropic"
    embedding_provider: str = "openai"
    embedding_dim: int = 1536

    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-5"

    # RCA agent: a dedicated read-only Postgres role backs the
    # run_readonly_sql escape hatch (see rca_db_tools.execute_readonly) so an
    # LLM-authored query can never write, even if the app-level SELECT/WITH
    # check were somehow bypassed. Same host/db as database_url, different
    # role -- created in db.migrate().
    rca_readonly_database_url: str = "postgresql://rca_readonly:rcareadonly@localhost:15432/tagore"
    rca_readonly_password: str = "rcareadonly"
    rca_supervisor_model: str = "claude-sonnet-5"
    # Warmer than the judge's near-zero setting: the RCA supervisor is
    # synthesizing a first-draft investigation narrative from tool results,
    # not cross-checking compliance -- some variation in phrasing is fine
    # here, unlike eval_judge_temperature below where it isn't.
    # Code-level default only (not set in .env); override with
    # RCA_SUPERVISOR_TEMPERATURE if needed.
    rca_supervisor_temperature: float = 0.7
    # Volume-mounted in docker-compose.yml (see main.py's file logging setup).
    rca_log_file: str = "/app/logs/backend.log"

    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    ollama_base_url: str = "http://ollama:11444"
    ollama_chat_model: str = "llama3.2"
    ollama_embedding_model: str = "nomic-embed-text"
    # Ollama's default context window (~2048 tokens for most models) is
    # smaller than a top_k=10 prompt can need, and /api/generate silently
    # truncates instead of erroring when a prompt exceeds it -- discovered
    # when raising eval top_k made generation_input_tokens plateau at the
    # same value for every question and correctness collapsed even though
    # the (untruncated) context judged fine on info_recall. Passed as
    # options.num_ctx on every Ollama chat/judge call.
    ollama_num_ctx: int = 8192

    local_embedding_model: str = "BAAI/bge-base-en-v1.5"

    retrieval_top_k: int = 5
    max_upload_mb: int = 25

    # Relative to the backend process's cwd. docker-compose mounts the
    # repo's corpus/ folder at /app/corpus (see docker-compose.yml); running
    # the backend outside docker requires the same relative layout or an
    # env override.
    corpus_dir: str = "corpus"

    # Judge LLM for scoring eval-cycle answers. Deliberately a different
    # model than the config's own chat_provider to avoid self-preference
    # bias. docker-compose mounts eval/golden_qa.json at eval_dataset_path
    # (see docker-compose.yml) so the eval harness's dataset stays the
    # single source of truth rather than being duplicated into the backend.
    eval_judge_provider: str = "openai"
    eval_judge_model: str = "gpt-4o-mini"
    eval_dataset_path: str = "eval/golden_qa.json"

    # Recall@k diagnostic, decoupled from retrieval_top_k -- computed against
    # a separately-fetched candidate pool of max(these values) chunks, not
    # whatever top_k the config actually uses for generation. Comma-separated
    # (not list[int]) to keep env-var overrides simple; parsed where used.
    # Configurable rather than a fixed set because different corpora/configs
    # may want a different k range to see where recall plateaus.
    eval_recall_k_values: str = "5,10,20,50"

    # How many eval cycles (POST /api/eval/run and /api/eval/reval both count
    # against this, sharing one counter) may run at once. Was previously a
    # hardcoded single-flight boolean lock (routes_eval.py's eval_in_progress)
    # -- this makes that same "1 at a time" behavior an explicit, visible
    # setting instead of an implicit code-level flag, without changing
    # today's behavior. Exists specifically to avoid GPU/LLM-backend
    # contention (see docs/pending-tasks.md's Reval entry) -- raise only once
    # the backend serving chat/judge calls can actually take concurrent load.
    # Code-level default only (not set in .env); override with
    # EVAL_CYCLE_CONCURRENCY if needed.
    eval_cycle_concurrency: int = 1

    # Near-deterministic: the judge is cross-checking/extracting a score
    # against fixed criteria, not drafting prose -- the same (question,
    # context, answer) triple should score the same way every time a cycle
    # is rerun or reval'd, not drift with sampling noise.
    # Code-level default only (not set in .env); override with
    # EVAL_JUDGE_TEMPERATURE if needed. Ignored when DynamicRouter is enabled
    # -- a routed judge call uses its tier's own `temperature` instead.
    eval_judge_temperature: float = 0.0

    # Caps how many of the per-question judge calls (correctness,
    # faithfulness, answer_relevance, info_recall, citation_completeness) run
    # concurrently. Default of 2 matches a typical 2-node setup for whatever
    # LLM backend is serving judge calls -- raise it if that backend can
    # actually take more concurrent load.
    eval_judge_concurrency: int = 2

    # Whether correctness/answer_relevance/citation_completeness are scored
    # with one combined judge call (score_reference_based) or three separate
    # calls (score_correctness/score_answer_relevance/score_citation_completeness).
    # Both implementations stay in judge.py -- flip this to A/B compare call-merge
    # efficiency (fewer, cheaper, faster calls) against separate-call score
    # stability (docs/pending-tasks.md: "Validate the combined judge call isn't
    # a quality regression").
    eval_judge_combine_calls: bool = True

    # DynamicRouter tiers. Empty (the default) means routing is off and
    # services keep using their own fixed providers. Set as JSON in .env, e.g.
    #   ROUTER_TIERS=[{"name":"small","model":"mistral:latest","base_url":"http://host:11444/v1"},
    #                 {"name":"large","model":"gpt-oss:20b","base_url":"http://host:11444/v1"}]
    #   ROUTER_TASK_TIERS={"extraction":"small","synthesis":"large"}
    #   ROUTER_FALLBACK_TIER=large
    # router_task_tiers maps a TaskType value to a tier name; unmapped tasks
    # use the fallback tier, which is also where escalation and errors go.
    router_tiers: list[TierSettings] = []
    router_task_tiers: dict[str, str] = {}
    router_fallback_tier: str | None = None

    @model_validator(mode="after")
    def _check_router_tiers(self):
        if not self.router_tiers:
            return self
        names = [t.name for t in self.router_tiers]
        if len(set(names)) != len(names):
            raise ValueError(f"router_tiers has duplicate names: {names}")
        if self.router_fallback_tier not in names:
            raise ValueError(
                f"router_fallback_tier {self.router_fallback_tier!r} must name one of {names}"
            )
        unknown = {t for t in self.router_task_tiers.values() if t not in names}
        if unknown:
            raise ValueError(f"router_task_tiers references unknown tiers: {sorted(unknown)}")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
