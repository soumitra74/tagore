import math

import pytest

from app.providers.reranker import (
    LocalCrossEncoderReranker,
    RERANKER_MODELS,
    get_reranker,
)
from app.services.chat import answer_question, rerank_rows
from app.services.eval_run import new_eval_job, run_eval_job
from tests.test_eval_run import (
    DATASET,
    FakeChatProvider,
    FakeEmbeddingProvider,
    FakeJudge,
    RankedFakeDatabase,
    TEST_CONFIG,
)


def _row(i, text=None, page=None):
    return {"id": f"{i:08d}-1111-1111-1111-111111111111", "chunk_text": text or f"chunk {i}",
            "context_text": None, "page_number": page if page is not None else i,
            "filename": "doc.pdf"}


class KeywordReranker:
    """Scores 1.0 for a passage equal to `target`, else 0.0."""

    def __init__(self, target):
        self.target = target
        self.calls: list[tuple[str, list[str]]] = []

    async def rerank(self, query, passages):
        self.calls.append((query, list(passages)))
        return [1.0 if p == self.target else 0.0 for p in passages]


@pytest.mark.asyncio
async def test_rerank_rows_reorders_head_and_keeps_tail_order():
    rows = [_row(i) for i in range(6)]
    out = await rerank_rows(KeywordReranker("chunk 2"), "q", rows, candidates=4)
    assert [r["page_number"] for r in out] == [2, 0, 1, 3, 4, 5]


@pytest.mark.asyncio
async def test_rerank_rows_scores_only_the_candidate_head():
    reranker = KeywordReranker("chunk 0")
    await rerank_rows(reranker, "q", [_row(i) for i in range(6)], candidates=3)
    [(query, passages)] = reranker.calls
    assert query == "q" and passages == ["chunk 0", "chunk 1", "chunk 2"]


@pytest.mark.asyncio
async def test_rerank_rows_with_no_rows_is_a_noop():
    assert await rerank_rows(KeywordReranker("x"), "q", [], candidates=5) == []


class PoolDatabase:
    def __init__(self, rows):
        self.rows = rows
        self.fetch_sizes: list[int] = []

    async def search_similar_chunks(self, embedding, config_id, dim, top_k):
        self.fetch_sizes.append(top_k)
        return self.rows[:top_k]


@pytest.mark.asyncio
async def test_answer_question_promotes_reranked_chunk_into_top_k():
    db = PoolDatabase([_row(i) for i in range(20)])
    result = await answer_question(
        "q", db, FakeEmbeddingProvider(), FakeChatProvider(), top_k=1, config=TEST_CONFIG,
        reranker=KeywordReranker("chunk 12"), rerank_candidates=15,
    )
    assert [s.page for s in result.sources] == [12]
    assert result.rerank_latency_ms is not None


@pytest.mark.asyncio
async def test_answer_question_widens_fetch_to_cover_rerank_candidates():
    db = PoolDatabase([_row(i) for i in range(50)])
    await answer_question(
        "q", db, FakeEmbeddingProvider(), FakeChatProvider(), top_k=5, config=TEST_CONFIG,
        reranker=KeywordReranker("x"), rerank_candidates=40,
    )
    assert db.fetch_sizes == [40]


@pytest.mark.asyncio
async def test_answer_question_without_reranker_is_unchanged():
    db = PoolDatabase([_row(i) for i in range(20)])
    result = await answer_question(
        "q", db, FakeEmbeddingProvider(), FakeChatProvider(), top_k=1, config=TEST_CONFIG,
    )
    assert [s.page for s in result.sources] == [0]
    assert result.rerank_latency_ms is None
    assert db.fetch_sizes == [3]


@pytest.mark.asyncio
async def test_run_eval_job_records_rerank_metrics_and_latency():
    job = new_eval_job(config_id=1, total=1)
    db = RankedFakeDatabase()
    reranker = KeywordReranker("chunk 7")

    await run_eval_job(
        job, db, TEST_CONFIG, FakeEmbeddingProvider(), FakeChatProvider(), FakeJudge(),
        DATASET, top_k=5, cycle_id=99, judge_concurrency=2,
        recall_k_values="5,10,20", reranker=reranker, rerank_candidates=10,
    )

    [result] = job.per_query
    metrics = result.retrieval_metrics
    # Raw ranking is untouched: gold page sits at candidate rank 8.
    assert metrics["recall@5"] == 0.0
    assert metrics["ndcg@10"] == pytest.approx(1 / math.log2(9))
    # After reranking the gold page is first.
    assert metrics["rerank_recall@5"] == 1.0
    assert metrics["rerank_ndcg@10"] == pytest.approx(1.0)
    assert result.rerank_latency_ms is not None

    [recorded] = db.recorded_calls
    assert recorded["rerank_latency_ms"] == result.rerank_latency_ms
    assert job.aggregates["mean_rerank_latency_ms"] == result.rerank_latency_ms
    assert job.aggregates["mean_retrieval_metrics"]["rerank_ndcg@10"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_run_eval_job_without_reranker_has_no_rerank_keys():
    job = new_eval_job(config_id=1, total=1)
    db = RankedFakeDatabase()
    await run_eval_job(
        job, db, TEST_CONFIG, FakeEmbeddingProvider(), FakeChatProvider(), FakeJudge(),
        DATASET, top_k=5, cycle_id=99, judge_concurrency=2, recall_k_values="5,10",
    )
    [result] = job.per_query
    assert not any(k.startswith("rerank_") for k in result.retrieval_metrics)
    assert result.rerank_latency_ms is None
    assert job.aggregates["mean_rerank_latency_ms"] is None


def test_unknown_reranker_model_rejected():
    with pytest.raises(ValueError, match="Unknown rerank model"):
        LocalCrossEncoderReranker("not/a-model")


def test_reranker_is_cached_per_model_and_loaded_lazily():
    cache: dict = {}
    model = next(iter(RERANKER_MODELS))
    first = get_reranker(cache, model)
    assert get_reranker(cache, model) is first
    assert first._model is None
