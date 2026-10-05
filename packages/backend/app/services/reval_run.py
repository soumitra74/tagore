import asyncio
import logging
import time

from app.services.eval_run import EvalJob, EvalQueryResult, _summarize
from app.services.judge import Judge

logger = logging.getLogger(__name__)

# No generation call -- reval reuses the original run's generated_answer
# rather than calling answer_question() again. combine_calls=True merges
# correctness/answer_relevance/citation_completeness into one
# score_reference_based call (+ faithfulness + info_recall = 3 total);
# False scores those three separately (+ faithfulness + info_recall = 5
# total). Mirrors eval_run.py's LLM_CALLS_PER_QUESTION_COMBINED/_SEPARATE.
LLM_CALLS_PER_QUESTION_COMBINED = 3
LLM_CALLS_PER_QUESTION_SEPARATE = 5


async def run_reval_job(
    job: EvalJob, db, config, judge: Judge, source_cycle_id: int, cycle_id: int,
    judge_concurrency: int, combine_calls: bool = True,
) -> None:
    llm_calls_per_question = (
        LLM_CALLS_PER_QUESTION_COMBINED if combine_calls else LLM_CALLS_PER_QUESTION_SEPARATE
    )
    semaphore = asyncio.Semaphore(judge_concurrency)

    async def _bounded(coro):
        async with semaphore:
            return await coro

    source_rows = await db.get_cycle_reval_source(source_cycle_id)
    all_chunk_ids = {
        str(cid) for row in source_rows for cid in (row["retrieved_chunk_ids"] or [])
    }
    chunk_map = await db.resolve_chunks(list(all_chunk_ids), config.embedding_dim)

    for row in source_rows:
        chunk_ids = [str(cid) for cid in (row["retrieved_chunk_ids"] or [])]
        resolved = [chunk_map[cid] for cid in chunk_ids if cid in chunk_map]

        if len(resolved) < len(chunk_ids):
            # The source document(s) were likely re-ingested since the
            # original run -- that deletes and regenerates chunks_dim_N rows
            # with new ids, orphaning the ones this question was retrieved
            # against. Can't faithfully reconstruct context or source labels,
            # so skip judging rather than score against wrong/missing context.
            missing = len(chunk_ids) - len(resolved)
            error = (
                f"StaleContext: {missing}/{len(chunk_ids)} retrieved chunks no longer "
                "exist (documents likely re-ingested since the original run)"
            )
            logger.warning("reval question %r: %s", row["query_id"], error)
            job.per_query.append(EvalQueryResult(
                id=str(row["query_id"]), question=row["question"], hit=bool(row["hit_rate"]),
                mrr=row["mrr"], citation_accuracy=row["citation_accuracy"],
                citation_recall=row["citation_recall"] or 0.0,
                retrieval_metrics=row["retrieval_metrics"] or {},
                correctness=0.0, correctness_reasoning="",
                generated_answer=row["generated_answer"],
                generation_latency_ms=row["generation_latency_ms"] or 0, eval_latency_ms=0,
                error=error,
            ))
            continue

        # Same preference as chat.py's _display_text: sentence_window chunks
        # carry their padded context separately from the embedded sentence.
        context_chunks = [c["context_text"] or c["chunk_text"] for c in resolved]
        source_labels = [
            f"Source {i}: {c['filename']}, page {c['page_number']}"
            for i, c in enumerate(resolved, start=1)
        ]

        eval_start = time.monotonic()
        if combine_calls:
            combined_result, faithfulness_result, recall_result = await asyncio.gather(
                _bounded(judge.score_reference_based(
                    row["question"], row["reference_answer"], row["generated_answer"],
                    source_labels,
                )),
                _bounded(judge.score_faithfulness(
                    row["question"], context_chunks, row["generated_answer"]
                )),
                _bounded(judge.score_info_recall(row["reference_answer"], context_chunks)),
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
                    row["question"], row["reference_answer"], row["generated_answer"]
                )),
                _bounded(judge.score_faithfulness(
                    row["question"], context_chunks, row["generated_answer"]
                )),
                _bounded(judge.score_answer_relevance(row["question"], row["generated_answer"])),
                _bounded(judge.score_info_recall(row["reference_answer"], context_chunks)),
                _bounded(judge.score_citation_completeness(
                    row["question"], source_labels, row["generated_answer"]
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
        eval_input_tokens = sum(jr.input_tokens or 0 for jr in judge_calls)
        eval_output_tokens = sum(jr.output_tokens or 0 for jr in judge_calls)

        await db.record_eval_run(
            config.id, row["query_id"], row["generated_answer"], row["hit_rate"], row["mrr"],
            row["citation_accuracy"], correctness,
            row["generation_latency_ms"] or 0, eval_latency_ms,
            cycle_id=cycle_id,
            citation_recall=row["citation_recall"],
            retrieval_metrics=row["retrieval_metrics"],
            rerank_latency_ms=row.get("rerank_latency_ms"),
            faithfulness=faithfulness_result.score,
            answer_relevance=answer_relevance,
            info_recall=recall_result.score,
            citation_completeness=citation_completeness,
            generation_input_tokens=row["generation_input_tokens"],
            generation_output_tokens=row["generation_output_tokens"],
            eval_input_tokens=eval_input_tokens, eval_output_tokens=eval_output_tokens,
            llm_call_count=llm_calls_per_question,
            retrieved_chunk_ids=chunk_ids,
        )

        job.per_query.append(EvalQueryResult(
            id=str(row["query_id"]), question=row["question"], hit=bool(row["hit_rate"]),
            mrr=row["mrr"], citation_accuracy=row["citation_accuracy"],
            citation_recall=row["citation_recall"] or 0.0,
            retrieval_metrics=row["retrieval_metrics"] or {},
            rerank_latency_ms=row.get("rerank_latency_ms"),
            correctness=correctness,
            correctness_reasoning=correctness_reasoning,
            generated_answer=row["generated_answer"],
            generation_latency_ms=row["generation_latency_ms"] or 0,
            eval_latency_ms=eval_latency_ms,
            faithfulness=faithfulness_result.score,
            faithfulness_reasoning=faithfulness_result.reasoning,
            answer_relevance=answer_relevance,
            answer_relevance_reasoning=answer_relevance_reasoning,
            info_recall=recall_result.score,
            info_recall_reasoning=recall_result.reasoning,
            citation_completeness=citation_completeness,
            citation_completeness_reasoning=citation_completeness_reasoning,
            generation_input_tokens=row["generation_input_tokens"],
            generation_output_tokens=row["generation_output_tokens"],
            eval_input_tokens=eval_input_tokens, eval_output_tokens=eval_output_tokens,
            llm_call_count=llm_calls_per_question,
        ))

    job.aggregates = _summarize(job.per_query)
    job.status = "completed"
    await db.complete_eval_cycle(cycle_id, job.status, job.aggregates)
