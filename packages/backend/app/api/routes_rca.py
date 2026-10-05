from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.services.rca_agent import run_turn, to_anthropic_messages

router = APIRouter()


class CreateSessionRequest(BaseModel):
    cycle_id: int
    query_id: int


class SendMessageRequest(BaseModel):
    message: str


@router.get("/api/rca/sessions")
async def list_sessions(request: Request):
    db = request.app.state.db
    sessions = await db.list_rca_sessions()
    return [
        {
            "id": s["id"], "cycle_id": s["cycle_id"], "query_id": s["query_id"],
            "total_input_tokens": s["total_input_tokens"],
            "total_output_tokens": s["total_output_tokens"],
        }
        for s in sessions
    ]


@router.delete("/api/rca/sessions/{session_id}")
async def delete_session(session_id: int, request: Request):
    db = request.app.state.db
    deleted = await db.soft_delete_rca_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")
    return {"status": "deleted"}


@router.post("/api/rca/sessions")
async def create_session(request: Request, body: CreateSessionRequest):
    db = request.app.state.db
    cycle = await db.get_eval_cycle_summary(body.cycle_id)
    if cycle is None:
        raise HTTPException(status_code=400, detail=f"Unknown cycle_id: {body.cycle_id}")
    config = await db.get_experiment_config(cycle["config_id"])

    session_id = await db.create_rca_session(body.cycle_id, body.query_id)
    seed_message = f"Investigate cycle {body.cycle_id}, question {body.query_id}."
    await db.insert_rca_message(session_id, "user", seed_message, None)

    text, tool_trace, input_tokens, output_tokens = await run_turn(
        request.app.state.rca_provider, db, config.embedding_dim,
        request.app.state.settings.rca_log_file,
        [{"role": "user", "content": seed_message}],
    )
    await db.insert_rca_message(
        session_id, "assistant", text, tool_trace, input_tokens, output_tokens,
    )

    return {
        "session_id": session_id,
        "message": {"role": "assistant", "content": text, "tool_trace": tool_trace},
    }


@router.get("/api/rca/sessions/{session_id}/messages")
async def get_messages(session_id: int, request: Request):
    db = request.app.state.db
    return await db.list_rca_messages(session_id)


@router.post("/api/rca/sessions/{session_id}/messages")
async def send_message(session_id: int, request: Request, body: SendMessageRequest):
    db = request.app.state.db
    session = await db.get_rca_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")
    cycle = await db.get_eval_cycle_summary(session["cycle_id"])
    config = await db.get_experiment_config(cycle["config_id"])

    history = await db.list_rca_messages(session_id)
    messages = to_anthropic_messages(history) + [{"role": "user", "content": body.message}]

    text, tool_trace, input_tokens, output_tokens = await run_turn(
        request.app.state.rca_provider, db, config.embedding_dim,
        request.app.state.settings.rca_log_file, messages,
    )
    await db.insert_rca_message(session_id, "user", body.message, None)
    await db.insert_rca_message(
        session_id, "assistant", text, tool_trace, input_tokens, output_tokens,
    )

    return {"role": "assistant", "content": text, "tool_trace": tool_trace}
