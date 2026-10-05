import asyncio

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.providers.factory import get_embedding_provider_for_config
from app.services.eval_run import EvalQueryResult, load_golden_dataset, new_eval_job, run_eval_job
from app.providers.reranker import (
    DEFAULT_RERANK_CANDIDATES,
    MAX_RERANK_CANDIDATES,
    RERANKER_MODELS,
    get_reranker,
)
from app.routing.from_settings import judge_identity
from app.services.judge import create_judge
from app.services.reval_run import run_reval_job

router = APIRouter()


class StartEvalRunRequest(BaseModel):
    config_id: int
    top_k: int | None = None
    graph_augmented: bool = False
    rerank_model: str | None = None  # None = reranking off
    rerank_candidates: int | None = None


class StartRevalRequest(BaseModel):
    source_cycle_id: int


def _resolve_rerank(body: StartEvalRunRequest) -> tuple[str | None, int | None]:
    if body.rerank_model is None:
        return None, None
    if body.rerank_model not in RERANKER_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown rerank_model {body.rerank_model!r}; known: {sorted(RERANKER_MODELS)}",
        )
    candidates = body.rerank_candidates or DEFAULT_RERANK_CANDIDATES
    if not 1 <= candidates <= MAX_RERANK_CANDIDATES:
        raise HTTPException(
            status_code=400,
            detail=f"rerank_candidates must be between 1 and {MAX_RERANK_CANDIDATES}",
        )
    return body.rerank_model, candidates


async def _run_eval_job_and_release(app_state, *args, **kwargs) -> None:
    try:
        await run_eval_job(*args, **kwargs)
    finally:
        app_state.eval_running_count -= 1


async def _run_reval_job_and_release(app_state, *args, **kwargs) -> None:
    try:
        await run_reval_job(*args, **kwargs)
    finally:
        app_state.eval_running_count -= 1


def _query_result_to_dict(r: EvalQueryResult) -> dict:
    return {
        "id": r.id,
        "question": r.question,
        "hit": r.hit,
        "mrr": r.mrr,
        "citation_accuracy": r.citation_accuracy,
        "citation_recall": r.citation_recall,
        "correctness": r.correctness,
        "correctness_reasoning": r.correctness_reasoning,
        "faithfulness": r.faithfulness,
        "faithfulness_reasoning": r.faithfulness_reasoning,
        "answer_relevance": r.answer_relevance,
        "answer_relevance_reasoning": r.answer_relevance_reasoning,
        "info_recall": r.info_recall,
        "info_recall_reasoning": r.info_recall_reasoning,
        "citation_completeness": r.citation_completeness,
        "citation_completeness_reasoning": r.citation_completeness_reasoning,
        "generated_answer": r.generated_answer,
        "generation_latency_ms": r.generation_latency_ms,
        "eval_latency_ms": r.eval_latency_ms,
        "generation_input_tokens": r.generation_input_tokens,
        "generation_output_tokens": r.generation_output_tokens,
        "eval_input_tokens": r.eval_input_tokens,
        "eval_output_tokens": r.eval_output_tokens,
        "llm_call_count": r.llm_call_count,
        "error": r.error,
    }


@router.get("/api/eval/judge")
async def get_eval_judge(request: Request):
    settings = request.app.state.settings
    provider, model = judge_identity(settings)
    return {
        "provider": provider,
        "model": model,
        "combine_calls": settings.eval_judge_combine_calls,
    }


@router.post("/api/eval/run")
async def start_eval_run(request: Request, body: StartEvalRunRequest):
    settings = request.app.state.settings
    if request.app.state.eval_running_count >= settings.eval_cycle_concurrency:
        raise HTTPException(status_code=409, detail="An eval run is already in progress")
    request.app.state.eval_running_count += 1

    try:
        db = request.app.state.db

        config = await db.get_experiment_config(body.config_id)
        if config is None:
            raise HTTPException(status_code=400, detail=f"Unknown config_id: {body.config_id}")

        try:
            dataset = load_golden_dataset(settings.eval_dataset_path)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=500, detail=f"Golden dataset not found: {settings.eval_dataset_path}"
            ) from exc

        cache = request.app.state.embedding_provider_cache
        if config.id not in cache:
            cache[config.id] = get_embedding_provider_for_config(settings, config)

        rerank_model, rerank_candidates = _resolve_rerank(body)
        reranker = (
            get_reranker(request.app.state.reranker_cache, rerank_model) if rerank_model else None
        )

        if request.app.state.judge is None:
            request.app.state.judge = create_judge(settings, request.app.state.router)

        top_k = body.top_k if body.top_k is not None else settings.retrieval_top_k
        job = new_eval_job(config.id, len(dataset))
        request.app.state.eval_jobs[job.id] = job
        cycle_id = await db.create_eval_cycle(
            config.id, len(dataset), *judge_identity(settings), body.graph_augmented,
            rerank_model, rerank_candidates,
        )
    except Exception:
        request.app.state.eval_running_count -= 1
        raise

    asyncio.create_task(
        _run_eval_job_and_release(
            request.app.state, job, db, config, cache[config.id], request.app.state.chat_provider,
            request.app.state.judge, dataset, top_k, cycle_id, settings.eval_judge_concurrency,
            settings.eval_judge_combine_calls, body.graph_augmented, request.app.state.ner,
            settings.eval_recall_k_values,
            reranker=reranker, rerank_candidates=rerank_candidates,
        )
    )

    return {"job_id": job.id, "config_id": config.id, "total": job.total, "cycle_id": cycle_id}


@router.post("/api/eval/reval")
async def start_reval_run(request: Request, body: StartRevalRequest):
    """Re-judge a completed cycle's questions with the currently configured
    judge, reusing its retrieval/generation results (no new answer_question()
    calls). Shares eval_running_count with /api/eval/run against the same
    settings.eval_cycle_concurrency limit, so a reval and a normal eval run
    count against the same cap."""
    settings = request.app.state.settings
    if request.app.state.eval_running_count >= settings.eval_cycle_concurrency:
        raise HTTPException(status_code=409, detail="An eval run is already in progress")
    request.app.state.eval_running_count += 1

    try:
        db = request.app.state.db

        source_cycle = await db.get_eval_cycle_summary(body.source_cycle_id)
        if source_cycle is None:
            raise HTTPException(
                status_code=400, detail=f"Unknown eval cycle: {body.source_cycle_id}"
            )

        config = await db.get_experiment_config(source_cycle["config_id"])
        if config is None:
            raise HTTPException(
                status_code=400,
                detail=f"Source cycle's config {source_cycle['config_id']} no longer exists",
            )

        if request.app.state.judge is None:
            request.app.state.judge = create_judge(settings, request.app.state.router)

        job = new_eval_job(config.id, source_cycle["total"])
        request.app.state.eval_jobs[job.id] = job
        cycle_id = await db.create_eval_cycle(
            config.id, source_cycle["total"], *judge_identity(settings),
        )
    except Exception:
        request.app.state.eval_running_count -= 1
        raise

    asyncio.create_task(
        _run_reval_job_and_release(
            request.app.state, job, db, config, request.app.state.judge,
            body.source_cycle_id, cycle_id, settings.eval_judge_concurrency,
            settings.eval_judge_combine_calls,
        )
    )

    return {"job_id": job.id, "config_id": config.id, "total": job.total, "cycle_id": cycle_id}


@router.get("/api/eval/run/{job_id}")
async def get_eval_run(request: Request, job_id: str):
    job = request.app.state.eval_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown eval job: {job_id}")

    return {
        "job_id": job.id,
        "config_id": job.config_id,
        "total": job.total,
        "status": job.status,
        "per_query": [_query_result_to_dict(r) for r in job.per_query],
        "aggregates": job.aggregates,
        "error": job.error,
    }


@router.get("/api/eval/history")
async def get_eval_history(request: Request):
    cycles = await request.app.state.db.list_eval_cycles()
    return [
        {
            "id": cycle["id"],
            "status": cycle["status"],
            "total": cycle["total"],
            "aggregates": cycle["aggregates"],
            "started_at": cycle["started_at"].isoformat(),
            "completed_at": cycle["completed_at"].isoformat() if cycle["completed_at"] else None,
            "judge_provider": cycle["judge_provider"],
            "judge_model": cycle["judge_model"],
            "graph_augmented": cycle["graph_augmented"],
            "rerank_model": cycle["rerank_model"],
            "rerank_candidates": cycle["rerank_candidates"],
            "config": cycle["config"],
            "per_query": [
                {
                    "id": r["id"],
                    "question": r["question"],
                    "generated_answer": r["generated_answer"],
                    "hit_rate": r["hit_rate"],
                    "mrr": r["mrr"],
                    "citation_accuracy": r["citation_accuracy"],
                    "citation_recall": r["citation_recall"],
                    "correctness": r["correctness"],
                    "faithfulness": r["faithfulness"],
                    "answer_relevance": r["answer_relevance"],
                    "info_recall": r["info_recall"],
                    "citation_completeness": r["citation_completeness"],
                    "generation_latency_ms": r["generation_latency_ms"],
                    "eval_latency_ms": r["eval_latency_ms"],
                    "generation_input_tokens": r["generation_input_tokens"],
                    "generation_output_tokens": r["generation_output_tokens"],
                    "eval_input_tokens": r["eval_input_tokens"],
                    "eval_output_tokens": r["eval_output_tokens"],
                    "llm_call_count": r["llm_call_count"],
                    "rerank_latency_ms": r["rerank_latency_ms"],
                    "run_at": r["run_at"].isoformat(),
                }
                for r in cycle["per_query"]
            ],
        }
        for cycle in cycles
    ]


@router.delete("/api/eval/history/{cycle_id}")
async def delete_eval_cycle(request: Request, cycle_id: int):
    deleted = await request.app.state.db.soft_delete_eval_cycle(cycle_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Unknown eval cycle: {cycle_id}")
    return {"id": cycle_id, "deleted": True}


@router.delete("/api/eval/history/{cycle_id}/purge")
async def purge_eval_cycle(request: Request, cycle_id: int):
    deleted = await request.app.state.db.hard_delete_eval_cycle(cycle_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Unknown eval cycle: {cycle_id}")
    return {"id": cycle_id, "deleted": True, "permanent": True}


@router.get("/api/eval/queries")
async def list_eval_queries(request: Request):
    db = request.app.state.db
    return await db.list_eval_queries()
