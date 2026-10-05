import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app.services.chat import (
    MAX_FETCH,
    OVERFETCH_MULTIPLIER,
    answer_question,
    generate_from_candidates,
    rerank_rows,
    retrieve_candidates,
)
from app.services.eval_metrics import (
    compute_citation_accuracy,
    compute_citation_recall,
    compute_hit,
    compute_mrr,
    compute_ndcg_at_k,
    compute_recall_at_k,
)
from app.services.judge import Judge

logger = logging.getLogger(__name__)

# One generate() call for the answer, plus judge calls that depend on
# combine_calls: True merges correctness/answer_relevance/citation_completeness
# into one score_reference_based call (+ faithfulness + info_recall = 4 total);
# False scores those three separately (+ faithfulness + info_recall = 6 total).
# No retries exist yet, so this is always exactly one of these two on success.
LLM_CALLS_PER_QUESTION_COMBINED = 4
LLM_CALLS_PER_QUESTION_SEPARATE = 6

# nDCG's rank cutoff. Fixed (unlike the configurable recall k-set) -- the
# stored key is "ndcg@10", so changing this changes what that key means.
NDCG_K = 10


@dataclass
class EvalQueryResult:
    id: str
    question: str
    hit: bool
    mrr: float
    citation_accuracy: float
    citation_recall: float
    correctness: float
    retrieval_metrics: dict[str, float]
    correctness_reasoning: str
    generated_answer: str
    generation_latency_ms: int
    eval_latency_ms: int
    faithfulness: float = 0.0
    faithfulness_reasoning: str = ""
    answer_relevance: float = 0.0
    answer_relevance_reasoning: str = ""
    info_recall: float = 0.0
    info_recall_reasoning: str = ""
    citation_completeness: float = 0.0
    citation_completeness_reasoning: str = ""
    generation_input_tokens: int | None = None
    generation_output_tokens: int | None = None
    eval_input_tokens: int | None = None
    eval_output_tokens: int | None = None
    llm_call_count: int = 0
    rerank_latency_ms: int | None = None
    error: str | None = None


@dataclass
class EvalJob:
    id: str
    config_id: int
    total: int
    status: str = "running"  # running | completed | failed
    per_query: list[EvalQueryResult] = field(default_factory=list)
    aggregates: dict | None = None
    error: str | None = None


def load_golden_dataset(path: str) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def new_eval_job(config_id: int, total: int) -> EvalJob:
    return EvalJob(id=str(uuid.uuid4()), config_id=config_id, total=total)


def _mean_retrieval_metrics(per_query: list[EvalQueryResult]) -> dict[str, float]:
    """Averages each retrieval_metrics key ("recall@5", "ndcg@10", ...) across
    questions. Keyed by whatever is actually present, so a new metric added
    to the dict needs no change here (a question that failed carries {}, so
    it's simply absent from that key's average rather than counted as 0)."""
    totals: dict[str, list[float]] = {}
    for r in per_query:
        for key, value in r.retrieval_metrics.items():
            totals.setdefault(key, []).append(value)
    return {key: sum(values) / len(values) for key, values in totals.items()}


def _mean_rerank_latency(per_query: list[EvalQueryResult]) -> float | None:
    """None when reranking was off (no question carries a latency)."""
    values = [r.rerank_latency_ms for r in per_query if r.rerank_latency_ms is not None]
    return sum(values) / len(values) if values else None


def _summarize(per_query: list[EvalQueryResult]) -> dict:
    n = len(per_query)
    if n == 0:
        return {"num_queries": 0, "hit_rate": 0.0, "mean_mrr": 0.0,
                "mean_citation_accuracy": 0.0, "mean_citation_recall": 0.0,
                "mean_retrieval_metrics": {},
                "mean_correctness": 0.0,
                "mean_faithfulness": 0.0, "mean_answer_relevance": 0.0,
                "mean_info_recall": 0.0, "mean_citation_completeness": 0.0,
                "mean_generation_latency_ms": 0.0, "mean_eval_latency_ms": 0.0,
                "total_generation_input_tokens": 0, "total_generation_output_tokens": 0,
                "total_eval_input_tokens": 0, "total_eval_output_tokens": 0,
                "total_llm_calls": 0, "mean_rerank_latency_ms": None}
    return {
        "num_queries": n,
        "hit_rate": sum(1 for r in per_query if r.hit) / n,
        "mean_mrr": sum(r.mrr for r in per_query) / n,
        "mean_citation_accuracy": sum(r.citation_accuracy for r in per_query) / n,
        "mean_citation_recall": sum(r.citation_recall for r in per_query) / n,
        "mean_retrieval_metrics": _mean_retrieval_metrics(per_query),
        "mean_correctness": sum(r.correctness for r in per_query) / n,
        "mean_faithfulness": sum(r.faithfulness for r in per_query) / n,
        "mean_answer_relevance": sum(r.answer_relevance for r in per_query) / n,
        "mean_info_recall": sum(r.info_recall for r in per_query) / n,
        "mean_citation_completeness": sum(r.citation_completeness for r in per_query) / n,
        "mean_generation_latency_ms": sum(r.generation_latency_ms or 0 for r in per_query) / n,
        "mean_eval_latency_ms": sum(r.eval_latency_ms or 0 for r in per_query) / n,
        "total_generation_input_tokens": sum(r.generation_input_tokens or 0 for r in per_query),
        "total_generation_output_tokens": sum(r.generation_output_tokens or 0 for r in per_query),
        "total_eval_input_tokens": sum(r.eval_input_tokens or 0 for r in per_query),
        "total_eval_output_tokens": sum(r.eval_output_tokens or 0 for r in per_query),
        "total_llm_calls": sum(r.llm_call_count for r in per_query),
        "mean_rerank_latency_ms": _mean_rerank_latency(per_query),
    }


async def _compute_retrieval_metrics(
    db, embedding_provider, config, question: str, golden_pages: set, k_values: list[int],
) -> dict[str, float]:
    """Ranked-retrieval diagnostics (recall@k for each configured k, plus
    ndcg@10), decoupled from the actual generation pipeline: re-embeds the
    question and fetches one candidate pool directly (not via
    answer_question(), which retrieves only top_k and applies
    dedup/graph-augmentation) so ranking can be measured deeper than
    whatever the config uses for generation. The pool is sized to cover both
    the largest recall k and NDCG_K, and every metric is computed from that
    same ranked list -- one fetch, no per-metric retrieval. The extra
    embedding call is cheap relative to the generation/judge LLM calls
    already made for this question. Keys are "<metric>@<k>" so a new metric
    is just another key -- storage, aggregation, and display all iterate
    whatever's present.

    Used only when no reranker is configured -- see
    _retrieve_rerank_and_score for the reranking case, which needs its own
    (shared, not duplicated) retrieval because reranking, unlike this
    function's plain embed+search, is too expensive to pay for twice."""
    question_embedding = await embedding_provider.embed_query(question)
    pool_size = max([*k_values, NDCG_K])
    candidates = await db.search_similar_chunks(
        question_embedding, config.id, config.embedding_dim, pool_size
    )
    candidate_pages = [(row["filename"], row["page_number"]) for row in candidates]
    metrics = {
        f"recall@{k}": compute_recall_at_k(candidate_pages, golden_pages, k)
        for k in k_values
    }
    metrics[f"ndcg@{NDCG_K}"] = compute_ndcg_at_k(candidate_pages, golden_pages, NDCG_K)
    return metrics


async def _retrieve_rerank_and_score(
    db, embedding_provider, config, question: str, golden_pages: set, k_values: list[int],
    top_k: int, graph_augmented: bool, ner, reranker, rerank_candidates: int,
) -> tuple[dict[str, float], list, int]:
    """The reranking counterpart of _compute_retrieval_metrics, but doing a
    single retrieval + single rerank pass that's reused for BOTH the ranking
    diagnostics below and the generation context the caller builds from the
    returned rows -- reranking is a cross-encoder forward pass per
    candidate, expensive enough that doing it twice (once for generation via
    answer_question, once more here) roughly doubled every reranked
    question's latency for no benefit. Returns (metrics, reranked_rows,
    rerank_latency_ms); the raw (pre-rerank) recall@k/ndcg@10 keys are
    computed from the same pool before reordering, so they still measure
    unreranked retrieval depth exactly as _compute_retrieval_metrics does."""
    generation_fetch_k = min(top_k * OVERFETCH_MULTIPLIER, MAX_FETCH)
    pool_size = max(generation_fetch_k, *k_values, NDCG_K, rerank_candidates)
    candidate_rows = await retrieve_candidates(
        question, db, embedding_provider, config, pool_size, graph_augmented, ner,
    )
    candidate_pages = [(row["filename"], row["page_number"]) for row in candidate_rows]
    metrics = {
        f"recall@{k}": compute_recall_at_k(candidate_pages, golden_pages, k)
        for k in k_values
    }
    metrics[f"ndcg@{NDCG_K}"] = compute_ndcg_at_k(candidate_pages, golden_pages, NDCG_K)

    rerank_start = time.monotonic()
    reranked_rows = await rerank_rows(reranker, question, candidate_rows, rerank_candidates)
    rerank_latency_ms = int((time.monotonic() - rerank_start) * 1000)
    reranked_pages = [(row["filename"], row["page_number"]) for row in reranked_rows]
    for k in k_values:
        metrics[f"rerank_recall@{k}"] = compute_recall_at_k(reranked_pages, golden_pages, k)
    metrics[f"rerank_ndcg@{NDCG_K}"] = compute_ndcg_at_k(reranked_pages, golden_pages, NDCG_K)

    return metrics, reranked_rows, rerank_latency_ms


async def run_eval_job(
    job: EvalJob, db, config, embedding_provider, chat_provider, judge: Judge,
    dataset: list[dict], top_k: int, cycle_id: int, judge_concurrency: int,
    combine_calls: bool = True, graph_augmented: bool = False, ner=None,
    recall_k_values: str = "5,10,20,50",
    reranker=None, rerank_candidates: int | None = None,
) -> None:
    llm_calls_per_question = (
        LLM_CALLS_PER_QUESTION_COMBINED if combine_calls else LLM_CALLS_PER_QUESTION_SEPARATE
    )
    k_values = [int(k) for k in recall_k_values.split(",") if k.strip()]
    semaphore = asyncio.Semaphore(judge_concurrency)

    async def _bounded(coro):
        async with semaphore:
            return await coro

    for item in dataset:
        golden_pages = {(p["document"], p["page"]) for p in item["golden_pages"]}
        start = time.monotonic()
        try:
            if reranker is not None and rerank_candidates:
                # One retrieval + rerank pass serves both generation and the
                # ranking diagnostics below (see _retrieve_rerank_and_score's
                # docstring) -- this is the branch that used to rerank twice.
                retrieval_metrics, ranked_rows, rerank_latency_ms = await _retrieve_rerank_and_score(
                    db, embedding_provider, config, item["question"], golden_pages, k_values,
                    top_k, graph_augmented, ner, reranker, rerank_candidates,
                )
                result = await generate_from_candidates(
                    item["question"], ranked_rows, chat_provider, top_k,
                )
                result.rerank_latency_ms = rerank_latency_ms
            else:
                result = await answer_question(
                    item["question"], db, embedding_provider, chat_provider, top_k, config,
                    graph_augmented=graph_augmented, ner=ner,
                )
                retrieval_metrics = await _compute_retrieval_metrics(
                    db, embedding_provider, config, item["question"], golden_pages, k_values,
                )
            generation_latency_ms = int((time.monotonic() - start) * 1000)
            retrieved_pages = [(s.document, s.page) for s in result.sources]
            hit = compute_hit(retrieved_pages, golden_pages)
            mrr = compute_mrr(retrieved_pages, golden_pages)
            citation_accuracy = compute_citation_accuracy(retrieved_pages, golden_pages)
            citation_recall = compute_citation_recall(retrieved_pages, golden_pages)
            source_labels = [
                f"Source {i}: {s.document}, page {s.page}"
                for i, s in enumerate(result.sources, start=1)
            ]
            retrieved_chunk_ids = [s.chunk_id for s in result.sources]
            # Independent judgments over the same (question, context, answer)
            # triple -- none depends on another's output, so run them
            # concurrently (bounded by judge_concurrency) rather than paying
            # for sequential LLM round trips. When combine_calls, correctness/
            # answer_relevance/citation_completeness are combined into one call
            # (score_reference_based) since none of them need retrieved
            # context; otherwise each is scored separately. faithfulness and
            # info_recall always stay separate calls because they do need
            # context (see docs/pending-tasks.md).
            eval_start = time.monotonic()
            if combine_calls:
                combined_result, faithfulness_result, recall_result = await asyncio.gather(
                    _bounded(judge.score_reference_based(
                        item["question"], item["reference_answer"], result.answer, source_labels
                    )),
                    _bounded(judge.score_faithfulness(
                        item["question"], result.context_chunks, result.answer
                    )),
                    _bounded(judge.score_info_recall(item["reference_answer"], result.context_chunks)),
                )
                correctness = combined_result.correctness
                correctness_reasoning = combined_result.correctness_reasoning
                answer_relevance = combined_result.answer_relevance
                answer_relevance_reasoning = combined_result.answer_relevance_reasoning
                citation_completeness = combined_result.citation_completeness
                citation_completeness_reasoning = combined_result.citation_completeness_reasoning
                judge_calls = [combined_result, faithfulness_result, recall_result]
            else:
                (
                    correctness_result, faithfulness_result, relevance_result,
                    recall_result, citation_result,
                ) = await asyncio.gather(
                    _bounded(judge.score_correctness(
                        item["question"], item["reference_answer"], result.answer
                    )),
                    _bounded(judge.score_faithfulness(
                        item["question"], result.context_chunks, result.answer
                    )),
                    _bounded(judge.score_answer_relevance(item["question"], result.answer)),
                    _bounded(judge.score_info_recall(item["reference_answer"], result.context_chunks)),
                    _bounded(judge.score_citation_completeness(
                        item["question"], source_labels, result.answer
                    )),
                )
                correctness = correctness_result.score
                correctness_reasoning = correctness_result.reasoning
                answer_relevance = relevance_result.score
                answer_relevance_reasoning = relevance_result.reasoning
                citation_completeness = citation_result.score
                citation_completeness_reasoning = citation_result.reasoning
                judge_calls = [
                    correctness_result, faithfulness_result, relevance_result,
                    recall_result, citation_result,
                ]
            eval_latency_ms = int((time.monotonic() - eval_start) * 1000)
            # Kept separate rather than summed: generation is the QnA call's
            # own token cost, eval is the judge's cost of scoring that
            # answer -- conflating them hides which one is driving spend.
            eval_input_tokens = sum(jr.input_tokens or 0 for jr in judge_calls)
            eval_output_tokens = sum(jr.output_tokens or 0 for jr in judge_calls)

            query_id = await db.find_or_create_eval_query(
                item["question"], item["reference_answer"], item.get("query_type")
            )
            await db.ensure_gold_pages(
                query_id, [(p["document"], p["page"]) for p in item["golden_pages"]]
            )
            await db.record_eval_run(
                config.id, query_id, result.answer, float(hit), mrr,
                citation_accuracy, correctness, generation_latency_ms,
                eval_latency_ms,
                cycle_id=cycle_id,
                citation_recall=citation_recall,
                retrieval_metrics=retrieval_metrics,
                faithfulness=faithfulness_result.score,
                answer_relevance=answer_relevance,
                info_recall=recall_result.score,
                citation_completeness=citation_completeness,
                generation_input_tokens=result.generation_input_tokens,
                generation_output_tokens=result.generation_output_tokens,
                eval_input_tokens=eval_input_tokens, eval_output_tokens=eval_output_tokens,
                llm_call_count=llm_calls_per_question,
                retrieved_chunk_ids=retrieved_chunk_ids,
                rerank_latency_ms=result.rerank_latency_ms,
            )

            job.per_query.append(EvalQueryResult(
                id=item["id"], question=item["question"], hit=hit, mrr=mrr,
                citation_accuracy=citation_accuracy, citation_recall=citation_recall,
                retrieval_metrics=retrieval_metrics,
                correctness=correctness,
                correctness_reasoning=correctness_reasoning,
                generated_answer=result.answer,
                generation_latency_ms=generation_latency_ms, eval_latency_ms=eval_latency_ms,
                faithfulness=faithfulness_result.score,
                faithfulness_reasoning=faithfulness_result.reasoning,
                answer_relevance=answer_relevance,
                answer_relevance_reasoning=answer_relevance_reasoning,
                info_recall=recall_result.score,
                info_recall_reasoning=recall_result.reasoning,
                citation_completeness=citation_completeness,
                citation_completeness_reasoning=citation_completeness_reasoning,
                generation_input_tokens=result.generation_input_tokens,
                generation_output_tokens=result.generation_output_tokens,
                eval_input_tokens=eval_input_tokens, eval_output_tokens=eval_output_tokens,
                llm_call_count=llm_calls_per_question,
                rerank_latency_ms=result.rerank_latency_ms,
            ))
        except Exception as exc:
            # Whatever elapsed before the failure is attributed to
            # generation, since we don't know which stage (answer_question()
            # or the judge gather) actually raised.
            elapsed_ms = int((time.monotonic() - start) * 1000)
            # str(exc) alone can be empty (e.g. a bare TimeoutError()), which
            # makes failures invisible in both the stored error field and the
            # logs -- log the full traceback and prefix the exception's type
            # name so there's always something to go on.
            logger.exception(
                "eval question %r failed after %dms", item["id"], elapsed_ms
            )
            job.per_query.append(EvalQueryResult(
                id=item["id"], question=item["question"], hit=False, mrr=0.0,
                citation_accuracy=0.0, citation_recall=0.0, retrieval_metrics={},
                correctness=0.0, correctness_reasoning="",
                generated_answer="", generation_latency_ms=elapsed_ms, eval_latency_ms=0,
                error=f"{type(exc).__name__}: {exc}",
            ))

    job.aggregates = _summarize(job.per_query)
    job.status = "completed"
    await db.complete_eval_cycle(cycle_id, job.status, job.aggregates)
