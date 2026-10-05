import asyncio

import pytest

from app.db import ExperimentConfig
from app.services.judge import CombinedJudgeResult, JudgeResult
from app.services.reval_run import (
    LLM_CALLS_PER_QUESTION_COMBINED,
    LLM_CALLS_PER_QUESTION_SEPARATE,
    run_reval_job,
)
from app.services.eval_run import new_eval_job

TEST_CONFIG = ExperimentConfig(
    id=1,
    chunking_strategy="sliding_window",
    chunk_size=1000,
    chunk_overlap=200,
    embedding_provider="local",
    embedding_model="BAAI/bge-base-en-v1.5",
    embedding_dim=768,
)

CHUNK_ID = "11111111-1111-1111-1111-111111111111"
MISSING_CHUNK_ID = "22222222-2222-2222-2222-222222222222"


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


class FakeDatabase:
    def __init__(self, source_rows, chunk_map):
        self._source_rows = source_rows
        self._chunk_map = chunk_map
        self.recorded_calls: list[dict] = []
        self.completed_cycle: tuple | None = None

    async def get_cycle_reval_source(self, cycle_id):
        return self._source_rows

    async def resolve_chunks(self, chunk_ids, dim):
        return {cid: self._chunk_map[cid] for cid in chunk_ids if cid in self._chunk_map}

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


@pytest.mark.asyncio
async def test_run_reval_job_rejudges_with_reconstructed_context():
    source_rows = [{
        "query_id": 42, "question": "what happened?", "reference_answer": "the reference answer",
        "generated_answer": "the original generated answer", "hit_rate": True, "mrr": 1.0,
        "citation_accuracy": 1.0, "citation_recall": 1.0, "retrieval_metrics": {"recall@5": 1.0, "ndcg@10": 1.0},
        "retrieved_chunk_ids": [CHUNK_ID],
        "generation_latency_ms": 5000, "generation_input_tokens": 100,
        "generation_output_tokens": 20,
    }]
    chunk_map = {CHUNK_ID: {"chunk_text": "chunk one", "context_text": None, "page_number": 3, "filename": "doc.pdf"}}
    db = FakeDatabase(source_rows, chunk_map)
    job = new_eval_job(config_id=1, total=1)

    await run_reval_job(
        job, db, TEST_CONFIG, FakeJudge(), source_cycle_id=99, cycle_id=100,
        judge_concurrency=2,
    )

    assert job.status == "completed"
    [result] = job.per_query
    assert result.error is None
    # Retrieval metrics carried forward unchanged from the source cycle.
    assert result.hit is True
    assert result.mrr == 1.0
    assert result.citation_accuracy == 1.0
    assert result.citation_recall == 1.0
    assert result.retrieval_metrics == {"recall@5": 1.0, "ndcg@10": 1.0}
    assert result.generated_answer == "the original generated answer"
    assert result.generation_latency_ms == 5000
    assert result.generation_input_tokens == 100
    assert result.generation_output_tokens == 20
    # Freshly judged.
    assert result.correctness == 0.9
    assert result.faithfulness == 0.8
    assert result.answer_relevance == 0.7
    assert result.info_recall == 0.6
    assert result.citation_completeness == 0.5

    assert result.llm_call_count == LLM_CALLS_PER_QUESTION_COMBINED == 3

    [recorded] = db.recorded_calls
    assert recorded["config_id"] == 1
    assert recorded["query_id"] == 42
    assert recorded["cycle_id"] == 100
    assert recorded["faithfulness"] == 0.8
    assert recorded["retrieval_metrics"] == {"recall@5": 1.0, "ndcg@10": 1.0}
    assert recorded["retrieved_chunk_ids"] == [CHUNK_ID]
    assert recorded["llm_call_count"] == 3

    assert job.aggregates["mean_correctness"] == 0.9
    assert db.completed_cycle == (100, "completed", job.aggregates)


@pytest.mark.asyncio
async def test_run_reval_job_with_combine_calls_false_scores_separately():
    source_rows = [{
        "query_id": 42, "question": "what happened?", "reference_answer": "the reference answer",
        "generated_answer": "the original generated answer", "hit_rate": True, "mrr": 1.0,
        "citation_accuracy": 1.0, "citation_recall": 1.0, "retrieval_metrics": {"recall@5": 1.0, "ndcg@10": 1.0},
        "retrieved_chunk_ids": [CHUNK_ID],
        "generation_latency_ms": 5000, "generation_input_tokens": 100,
        "generation_output_tokens": 20,
    }]
    chunk_map = {CHUNK_ID: {"chunk_text": "chunk one", "context_text": None, "page_number": 3, "filename": "doc.pdf"}}
    db = FakeDatabase(source_rows, chunk_map)
    job = new_eval_job(config_id=1, total=1)

    await run_reval_job(
        job, db, TEST_CONFIG, FakeJudge(), source_cycle_id=99, cycle_id=100,
        judge_concurrency=2, combine_calls=False,
    )

    assert job.status == "completed"
    [result] = job.per_query
    assert result.error is None
    assert result.correctness == 0.9
    assert result.answer_relevance == 0.7
    assert result.citation_completeness == 0.5
    assert result.faithfulness == 0.8
    assert result.info_recall == 0.6
    assert result.llm_call_count == LLM_CALLS_PER_QUESTION_SEPARATE == 5

    [recorded] = db.recorded_calls
    assert recorded["correctness"] == 0.9
    assert recorded["answer_relevance"] == 0.7
    assert recorded["citation_completeness"] == 0.5
    assert recorded["llm_call_count"] == 5


@pytest.mark.asyncio
async def test_run_reval_job_skips_questions_with_stale_chunk_ids():
    source_rows = [{
        "query_id": 43, "question": "what happened?", "reference_answer": "the reference answer",
        "generated_answer": "an answer", "hit_rate": True, "mrr": 1.0,
        "citation_accuracy": 1.0, "citation_recall": 1.0, "retrieval_metrics": {"recall@5": 1.0, "ndcg@10": 1.0},
        # One resolvable, one gone (e.g. the document was re-ingested).
        "retrieved_chunk_ids": [CHUNK_ID, MISSING_CHUNK_ID],
        "generation_latency_ms": 5000, "generation_input_tokens": 100,
        "generation_output_tokens": 20,
    }]
    chunk_map = {CHUNK_ID: {"chunk_text": "chunk one", "context_text": None, "page_number": 3, "filename": "doc.pdf"}}
    db = FakeDatabase(source_rows, chunk_map)
    job = new_eval_job(config_id=1, total=1)

    await run_reval_job(
        job, db, TEST_CONFIG, FakeJudge(), source_cycle_id=99, cycle_id=100,
        judge_concurrency=2,
    )

    assert job.status == "completed"
    [result] = job.per_query
    assert result.error is not None
    assert "1/2" in result.error
    assert db.recorded_calls == []  # never reached record_eval_run for this question


@pytest.mark.asyncio
async def test_run_reval_job_bounds_judge_concurrency_to_worker_count():
    class ConcurrencyTrackingJudge:
        CALL_DELAY = 0.05

        def __init__(self):
            self.in_flight = 0
            self.peak_in_flight = 0

        async def _tick(self, score, reasoning):
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
            await asyncio.sleep(self.CALL_DELAY)
            self.in_flight -= 1
            return JudgeResult(score=score, reasoning=reasoning)

        async def score_reference_based(self, question, reference_answer, generated_answer,
                                         source_labels):
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
            await asyncio.sleep(self.CALL_DELAY)
            self.in_flight -= 1
            return CombinedJudgeResult(
                correctness=0.9, correctness_reasoning="correct",
                answer_relevance=0.7, answer_relevance_reasoning="relevant",
                citation_completeness=0.5, citation_completeness_reasoning="one claim uncited",
            )

        async def score_faithfulness(self, question, context_chunks, generated_answer):
            return await self._tick(0.8, "grounded")

        async def score_info_recall(self, reference_answer, context_chunks):
            return await self._tick(0.6, "covers most")

    source_rows = [{
        "query_id": 44, "question": "what happened?", "reference_answer": "reference",
        "generated_answer": "answer", "hit_rate": True, "mrr": 1.0,
        "citation_accuracy": 1.0, "citation_recall": 1.0, "retrieval_metrics": {"recall@5": 1.0, "ndcg@10": 1.0},
        "retrieved_chunk_ids": [CHUNK_ID],
        "generation_latency_ms": 1000, "generation_input_tokens": 10,
        "generation_output_tokens": 2,
    }]
    chunk_map = {CHUNK_ID: {"chunk_text": "chunk one", "context_text": None, "page_number": 3, "filename": "doc.pdf"}}
    db = FakeDatabase(source_rows, chunk_map)
    job = new_eval_job(config_id=1, total=1)
    judge = ConcurrencyTrackingJudge()

    await run_reval_job(
        job, db, TEST_CONFIG, judge, source_cycle_id=99, cycle_id=100, judge_concurrency=2,
    )

    assert judge.peak_in_flight == 2
