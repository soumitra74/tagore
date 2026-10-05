import asyncio
import math

import pytest

from app.db import ExperimentConfig
from app.providers.base import LLMResponse
from app.services.eval_run import (
    LLM_CALLS_PER_QUESTION_COMBINED,
    LLM_CALLS_PER_QUESTION_SEPARATE,
    new_eval_job,
    run_eval_job,
)
from app.services.judge import CombinedJudgeResult, JudgeResult

TEST_CONFIG = ExperimentConfig(
    id=1,
    chunking_strategy="sliding_window",
    chunk_size=1000,
    chunk_overlap=200,
    embedding_provider="local",
    embedding_model="BAAI/bge-base-en-v1.5",
    embedding_dim=768,
)

DATASET = [
    {
        "id": "q1",
        "question": "what happened?",
        "reference_answer": "the reference answer",
        "golden_pages": [{"document": "doc.pdf", "page": 3}],
    }
]


class FakeEmbeddingProvider:
    async def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2]


class FakeChatProvider:
    async def generate(self, question: str, context_chunks: list[str]) -> LLMResponse:
        return LLMResponse(text="generated answer", input_tokens=100, output_tokens=20)


class FakeDatabase:
    async def search_similar_chunks(self, embedding, config_id, dim, top_k):
        return [{"id": "11111111-1111-1111-1111-111111111111", "chunk_text": "chunk one",
                  "context_text": None, "page_number": 3, "filename": "doc.pdf"}]

    async def find_or_create_eval_query(self, question, reference_answer, query_type):
        return 42

    async def ensure_gold_pages(self, query_id, pages):
        pass

    def __init__(self):
        self.recorded_calls: list[dict] = []
        self.completed_cycle: tuple | None = None

    async def record_eval_run(self, config_id, query_id, generated_answer, hit_rate, mrr,
                               citation_accuracy, correctness, generation_latency_ms,
                               eval_latency_ms, **kwargs):
        self.recorded_calls.append({
            "config_id": config_id, "query_id": query_id, "generated_answer": generated_answer,
            "hit_rate": hit_rate, "mrr": mrr, "citation_accuracy": citation_accuracy,
            "correctness": correctness, "generation_latency_ms": generation_latency_ms,
            "eval_latency_ms": eval_latency_ms, **kwargs,
        })

    async def complete_eval_cycle(self, cycle_id, status, aggregates):
        self.completed_cycle = (cycle_id, status, aggregates)


class FakeJudge:
    """Exposes both the combined and separate-call interfaces so the same
    fake can back either combine_calls value."""

    async def score_reference_based(self, question, reference_answer, generated_answer,
                                     source_labels):
        return CombinedJudgeResult(
            correctness=0.9, correctness_reasoning="correct",
            answer_relevance=0.7, answer_relevance_reasoning="relevant",
            citation_completeness=0.5, citation_completeness_reasoning="one claim uncited",
            input_tokens=10, output_tokens=5,
        )

    async def score_correctness(self, question, reference_answer, generated_answer):
        return JudgeResult(score=0.9, reasoning="correct", input_tokens=3, output_tokens=2)

    async def score_answer_relevance(self, question, generated_answer):
        return JudgeResult(score=0.7, reasoning="relevant", input_tokens=3, output_tokens=2)

    async def score_citation_completeness(self, question, source_labels, generated_answer):
        return JudgeResult(
            score=0.5, reasoning="one claim uncited", input_tokens=4, output_tokens=1
        )

    async def score_faithfulness(self, question, context_chunks, generated_answer):
        return JudgeResult(score=0.8, reasoning="grounded", input_tokens=11, output_tokens=6)

    async def score_info_recall(self, reference_answer, context_chunks):
        return JudgeResult(score=0.6, reasoning="covers most", input_tokens=13, output_tokens=8)


@pytest.mark.asyncio
async def test_run_eval_job_records_all_ragas_metrics():
    job = new_eval_job(config_id=1, total=1)
    db = FakeDatabase()

    await run_eval_job(
        job, db, TEST_CONFIG, FakeEmbeddingProvider(), FakeChatProvider(), FakeJudge(),
        DATASET, top_k=5, cycle_id=99, judge_concurrency=2,
    )

    assert job.status == "completed"
    [result] = job.per_query
    assert result.correctness == 0.9
    assert result.citation_recall == 1.0
    assert result.faithfulness == 0.8
    assert result.faithfulness_reasoning == "grounded"
    assert result.answer_relevance == 0.7
    assert result.answer_relevance_reasoning == "relevant"
    assert result.info_recall == 0.6
    assert result.info_recall_reasoning == "covers most"
    assert result.citation_completeness == 0.5
    assert result.citation_completeness_reasoning == "one claim uncited"

    # Generation (100/20) and the 3 judge calls -- combined correctness/
    # answer_relevance/citation_completeness (10/5) + faithfulness (11/6) +
    # info_recall (13/8) -- are tracked separately: generation is the QnA
    # call's own cost, eval is the judge's cost of scoring that answer.
    assert result.generation_input_tokens == 100
    assert result.generation_output_tokens == 20
    assert result.eval_input_tokens == 10 + 11 + 13
    assert result.eval_output_tokens == 5 + 6 + 8
    assert result.llm_call_count == LLM_CALLS_PER_QUESTION_COMBINED == 4

    # Split the same way as tokens: generation_latency_ms times only the
    # answer_question() call, eval_latency_ms times only the judge gather.
    assert isinstance(result.generation_latency_ms, int)
    assert result.generation_latency_ms >= 0
    assert isinstance(result.eval_latency_ms, int)
    assert result.eval_latency_ms >= 0

    [recorded] = db.recorded_calls
    assert recorded["citation_recall"] == 1.0
    assert recorded["faithfulness"] == 0.8
    assert recorded["answer_relevance"] == 0.7
    assert recorded["info_recall"] == 0.6
    assert recorded["citation_completeness"] == 0.5
    assert recorded["retrieved_chunk_ids"] == ["11111111-1111-1111-1111-111111111111"]
    assert recorded["generation_input_tokens"] == 100
    assert recorded["generation_output_tokens"] == 20
    assert recorded["eval_input_tokens"] == 10 + 11 + 13
    assert recorded["eval_output_tokens"] == 5 + 6 + 8
    assert recorded["generation_latency_ms"] == result.generation_latency_ms
    assert recorded["eval_latency_ms"] == result.eval_latency_ms

    assert job.aggregates["mean_citation_recall"] == 1.0
    assert job.aggregates["mean_faithfulness"] == 0.8
    assert job.aggregates["mean_answer_relevance"] == 0.7
    assert job.aggregates["mean_info_recall"] == 0.6
    assert job.aggregates["mean_citation_completeness"] == 0.5
    assert job.aggregates["total_generation_input_tokens"] == 100
    assert job.aggregates["total_generation_output_tokens"] == 20
    assert job.aggregates["total_eval_input_tokens"] == 10 + 11 + 13
    assert job.aggregates["total_eval_output_tokens"] == 5 + 6 + 8
    assert job.aggregates["mean_generation_latency_ms"] == result.generation_latency_ms
    assert job.aggregates["mean_eval_latency_ms"] == result.eval_latency_ms


@pytest.mark.asyncio
async def test_run_eval_job_with_combine_calls_false_scores_separately():
    job = new_eval_job(config_id=1, total=1)
    db = FakeDatabase()

    await run_eval_job(
        job, db, TEST_CONFIG, FakeEmbeddingProvider(), FakeChatProvider(), FakeJudge(),
        DATASET, top_k=5, cycle_id=99, judge_concurrency=2, combine_calls=False,
    )

    assert job.status == "completed"
    [result] = job.per_query
    assert result.correctness == 0.9
    assert result.answer_relevance == 0.7
    assert result.citation_completeness == 0.5
    assert result.faithfulness == 0.8
    assert result.info_recall == 0.6

    # 5 separate judge calls (3+4+3+13+4 in, 2+2+1+8+6 out) instead of the
    # combined call's 3 -- llm_call_count reflects the SEPARATE constant.
    assert result.eval_input_tokens == 3 + 3 + 4 + 11 + 13
    assert result.eval_output_tokens == 2 + 2 + 1 + 6 + 8
    assert result.llm_call_count == LLM_CALLS_PER_QUESTION_SEPARATE == 6

    [recorded] = db.recorded_calls
    assert recorded["correctness"] == 0.9
    assert recorded["answer_relevance"] == 0.7
    assert recorded["citation_completeness"] == 0.5
    assert recorded["llm_call_count"] == 6


class ConcurrencyTrackingJudge:
    """Records the peak number of judge calls in flight at once, to prove
    concurrency is bounded to the configured worker count -- not unbounded
    and not fully serial. 3 judge calls now (combined + faithfulness +
    info_recall), still > the 2-worker cap used below so the test still
    proves genuine concurrency rather than accidentally passing serially."""

    CALL_DELAY = 0.05

    def __init__(self):
        self.in_flight = 0
        self.peak_in_flight = 0

    async def _enter(self) -> None:
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        await asyncio.sleep(self.CALL_DELAY)
        self.in_flight -= 1

    async def _tick(self, score: float, reasoning: str) -> JudgeResult:
        await self._enter()
        return JudgeResult(score=score, reasoning=reasoning)

    async def score_reference_based(self, question, reference_answer, generated_answer,
                                     source_labels):
        await self._enter()
        return CombinedJudgeResult(
            correctness=0.9, correctness_reasoning="correct",
            answer_relevance=0.7, answer_relevance_reasoning="relevant",
            citation_completeness=0.5, citation_completeness_reasoning="one claim uncited",
        )

    async def score_correctness(self, question, reference_answer, generated_answer):
        return await self._tick(0.9, "correct")

    async def score_answer_relevance(self, question, generated_answer):
        return await self._tick(0.7, "relevant")

    async def score_citation_completeness(self, question, source_labels, generated_answer):
        return await self._tick(0.5, "one claim uncited")

    async def score_faithfulness(self, question, context_chunks, generated_answer):
        return await self._tick(0.8, "grounded")

    async def score_info_recall(self, reference_answer, context_chunks):
        return await self._tick(0.6, "covers most")


@pytest.mark.asyncio
async def test_run_eval_job_bounds_judge_concurrency_to_worker_count():
    job = new_eval_job(config_id=1, total=1)
    db = FakeDatabase()
    judge = ConcurrencyTrackingJudge()

    await run_eval_job(
        job, db, TEST_CONFIG, FakeEmbeddingProvider(), FakeChatProvider(), judge,
        DATASET, top_k=5, cycle_id=99, judge_concurrency=2,
    )

    # 3 judge calls (combined + faithfulness + info_recall), capped at 2
    # workers -- never more than 2 in flight, but more than fully-serial (1)
    # so it's genuinely running concurrently.
    assert judge.peak_in_flight == 2


@pytest.mark.asyncio
async def test_run_eval_job_respects_a_different_worker_count():
    job = new_eval_job(config_id=1, total=1)
    db = FakeDatabase()
    judge = ConcurrencyTrackingJudge()

    await run_eval_job(
        job, db, TEST_CONFIG, FakeEmbeddingProvider(), FakeChatProvider(), judge,
        DATASET, top_k=5, cycle_id=99, judge_concurrency=1,
    )

    # Proves the cap is the parameter's value, not a hardcoded 2.
    assert judge.peak_in_flight == 1


@pytest.mark.asyncio
async def test_run_eval_job_bounds_judge_concurrency_with_combine_calls_false():
    job = new_eval_job(config_id=1, total=1)
    db = FakeDatabase()
    judge = ConcurrencyTrackingJudge()

    await run_eval_job(
        job, db, TEST_CONFIG, FakeEmbeddingProvider(), FakeChatProvider(), judge,
        DATASET, top_k=5, cycle_id=99, judge_concurrency=2, combine_calls=False,
    )

    # 5 separate judge calls, capped at 2 workers.
    assert judge.peak_in_flight == 2


class FakeNER:
    async def extract(self, text):
        return []


@pytest.mark.asyncio
async def test_run_eval_job_passes_graph_augmented_through():
    db = FakeDatabase()
    received = {}

    async def fake_answer_question(question, db_, embedding_provider, chat_provider, top_k,
                                    config, graph_augmented=False, ner=None,
                                    reranker=None, rerank_candidates=None):
        received["graph_augmented"] = graph_augmented
        received["ner"] = ner
        return await answer_question_original(
            question, db_, embedding_provider, chat_provider, top_k, config,
            graph_augmented=graph_augmented, ner=ner,
            reranker=reranker, rerank_candidates=rerank_candidates,
        )

    import app.services.eval_run as eval_run_module
    answer_question_original = eval_run_module.answer_question
    eval_run_module.answer_question = fake_answer_question
    try:
        job = new_eval_job(TEST_CONFIG.id, len(DATASET))
        ner = FakeNER()
        await run_eval_job(
            job, db, TEST_CONFIG, FakeEmbeddingProvider(), FakeChatProvider(), FakeJudge(),
            DATASET, top_k=5, cycle_id=1, judge_concurrency=2,
            graph_augmented=True, ner=ner,
        )
    finally:
        eval_run_module.answer_question = answer_question_original

    assert received["graph_augmented"] is True
    assert received["ner"] is ner


class RankedFakeDatabase(FakeDatabase):
    """search_similar_chunks returns a ranked 20-item pool where the golden
    page only appears at candidate position 8 -- found at k=10 but not k=5,
    from the SAME underlying candidate list (proving _compute_retrieval_metrics
    fetches once and slices, not once per k)."""

    def __init__(self):
        super().__init__()
        self.search_calls: list[tuple] = []

    async def search_similar_chunks(self, embedding, config_id, dim, top_k):
        self.search_calls.append((config_id, dim, top_k))
        rows = [
            {"id": f"1111111{i}-1111-1111-1111-111111111111", "chunk_text": f"chunk {i}",
             "context_text": None, "page_number": i, "filename": "other.pdf"}
            for i in range(20)
        ]
        rows[7] = {"id": "77777777-7777-7777-7777-777777777777", "chunk_text": "chunk 7",
                   "context_text": None, "page_number": 3, "filename": "doc.pdf"}
        return rows[:top_k]


@pytest.mark.asyncio
async def test_run_eval_job_computes_recall_at_configured_k_values():
    job = new_eval_job(config_id=1, total=1)
    db = RankedFakeDatabase()

    await run_eval_job(
        job, db, TEST_CONFIG, FakeEmbeddingProvider(), FakeChatProvider(), FakeJudge(),
        DATASET, top_k=5, cycle_id=99, judge_concurrency=2,
        recall_k_values="5,10,20",
    )

    # Gold page sits at candidate rank 8: missed at recall@5, found from @10 on.
    # nDCG@10 discounts it: 1/log2(8+1) with a single gold page (ideal DCG = 1).
    expected = {
        "recall@5": 0.0, "recall@10": 1.0, "recall@20": 1.0,
        "ndcg@10": pytest.approx(1 / math.log2(9)),
    }
    [result] = job.per_query
    assert result.retrieval_metrics == expected

    [recorded] = db.recorded_calls
    assert recorded["retrieval_metrics"] == expected

    assert job.aggregates["mean_retrieval_metrics"] == expected

    # One fetch, sized to max(k_values)=20, not one fetch per metric.
    recall_diagnostic_calls = [c for c in db.search_calls if c[2] == 20]
    assert len(recall_diagnostic_calls) == 1


@pytest.mark.asyncio
async def test_run_eval_job_default_recall_k_values():
    job = new_eval_job(config_id=1, total=1)
    db = RankedFakeDatabase()

    await run_eval_job(
        job, db, TEST_CONFIG, FakeEmbeddingProvider(), FakeChatProvider(), FakeJudge(),
        DATASET, top_k=5, cycle_id=99, judge_concurrency=2,
    )

    [result] = job.per_query
    assert set(result.retrieval_metrics.keys()) == {
        "recall@5", "recall@10", "recall@20", "recall@50", "ndcg@10",
    }


@pytest.mark.asyncio
async def test_run_eval_job_ndcg_pool_covers_ndcg_k_even_when_recall_ks_are_smaller():
    # Recall k-set tops out at 5, but the gold page is at candidate rank 8 --
    # the pool must still reach NDCG_K=10 for nDCG@10 to see it.
    job = new_eval_job(config_id=1, total=1)
    db = RankedFakeDatabase()

    await run_eval_job(
        job, db, TEST_CONFIG, FakeEmbeddingProvider(), FakeChatProvider(), FakeJudge(),
        DATASET, top_k=5, cycle_id=99, judge_concurrency=2,
        recall_k_values="3,5",
    )

    [result] = job.per_query
    assert result.retrieval_metrics["recall@5"] == 0.0
    assert result.retrieval_metrics["ndcg@10"] == pytest.approx(1 / math.log2(9))
    assert any(c[2] == 10 for c in db.search_calls)
