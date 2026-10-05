import json

from app.services import rca_db_tools, rca_log_tools

MAX_TOOL_CALLS = 8

SYSTEM_PROMPT = """You are a root-cause analysis assistant for a RAG (retrieval-augmented generation) evaluation harness. Your job is to investigate why a specific eval run scored the way it did, using the tools available to you.

Domain knowledge specific to this corpus:
- The corpus contains near-duplicate NASA telemetry reports (PFR1 through PFR4) that share large amounts of near-identical boilerplate text. Retrieval can pull the "wrong" but content-identical duplicate document instead of the actual gold document for a question -- always check get_gold_pages against the documents actually present in retrieved_chunk_ids before concluding retrieval succeeded or failed.
- Citations in generated answers are written as [Source N], where N numbers the Nth retrieved CHUNK for that run, not a unique document. Multiple sources can point at the same document -- don't assume N implies N distinct documents.

For a new investigation, always start by calling get_eval_run with the given cycle_id and query_id. Use resolve_chunks to see what was actually retrieved, get_gold_pages to see what should have been retrieved, and check_text_contains to verify whether a specific citation is actually grounded in its chunk's text rather than assuming it from a preview. Use get_backend_logs if you suspect a crash, timeout, or infrastructure issue rather than a retrieval-quality one -- but note backend logging is currently sparse, so it may not have much to show. Use run_readonly_sql only for questions the other tools can't answer directly.

Explain your findings in plain language, citing the concrete evidence (chunk ids, page numbers, exact text) you gathered -- not just a metric name."""

TOOL_DEFINITIONS = [
    {
        "name": "get_eval_run",
        "description": (
            "Fetch everything known about one eval run: the cycle's config and "
            "graph_augmented flag, the question text and gold answer, the "
            "generated answer, all retrieval/judge metrics, and the list of "
            "retrieved_chunk_ids. Always call this first for a new investigation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "cycle_id": {"type": "integer"},
                "query_id": {"type": "integer"},
            },
            "required": ["cycle_id", "query_id"],
        },
    },
    {
        "name": "resolve_chunks",
        "description": "Resolve chunk ids to their document filename, page number, and full chunk text.",
        "input_schema": {
            "type": "object",
            "properties": {"chunk_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["chunk_ids"],
        },
    },
    {
        "name": "get_gold_pages",
        "description": "Get the gold (document, page) pairs a question's answer should have come from.",
        "input_schema": {
            "type": "object",
            "properties": {"query_id": {"type": "integer"}},
            "required": ["query_id"],
        },
    },
    {
        "name": "check_text_contains",
        "description": "Check whether a specific chunk's full text contains a claimed fact or phrase.",
        "input_schema": {
            "type": "object",
            "properties": {
                "chunk_id": {"type": "string"},
                "needle": {"type": "string"},
            },
            "required": ["chunk_id", "needle"],
        },
    },
    {
        "name": "get_backend_logs",
        "description": "Read backend log lines within a time range, optionally filtered by a substring.",
        "input_schema": {
            "type": "object",
            "properties": {
                "since": {"type": "string", "description": "e.g. '2026-09-17 10:00:00'"},
                "until": {"type": "string", "description": "e.g. '2026-09-17 10:10:00'"},
                "grep": {"type": "string"},
            },
            "required": ["since", "until"],
        },
    },
    {
        "name": "run_readonly_sql",
        "description": (
            "Escape hatch: run a read-only SQL SELECT/WITH query against the eval "
            "database for anything the other tools don't cover. Results are capped "
            "to 200 rows."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]


def to_anthropic_messages(history: list[dict]) -> list[dict]:
    """Reconstructs a plain user/assistant message list from persisted
    rca_messages rows for a follow-up turn. Only each past turn's *final*
    text is replayed, not its intermediate tool_use/tool_result blocks
    (those aren't persisted -- tool_trace is for display only) -- the
    current turn builds its own fresh tool_use/tool_result pairs in run_turn,
    which is all the Anthropic API needs to be valid."""
    return [{"role": row["role"], "content": row["content"]} for row in history]


async def _dispatch(name: str, args: dict, db, dim: int, log_file: str):
    if name == "get_eval_run":
        return await rca_db_tools.get_eval_run(db, args["cycle_id"], args["query_id"])
    if name == "resolve_chunks":
        return await rca_db_tools.resolve_chunks(db, dim, args["chunk_ids"])
    if name == "get_gold_pages":
        return await rca_db_tools.get_gold_pages(db, args["query_id"])
    if name == "check_text_contains":
        return await rca_db_tools.check_text_contains(db, dim, args["chunk_id"], args["needle"])
    if name == "get_backend_logs":
        return rca_log_tools.get_backend_logs(
            log_file, args["since"], args["until"], args.get("grep"),
        )
    if name == "run_readonly_sql":
        return await rca_db_tools.run_readonly_sql(db, args["query"])
    raise ValueError(f"Unknown tool: {name}")


def _summarize(result) -> str:
    text = json.dumps(result, default=str)
    return text if len(text) <= 300 else text[:300] + "..."


async def run_turn(
    provider, db, dim: int, log_file: str, messages: list[dict],
) -> tuple[str, list[dict], int, int]:
    """Runs the tool-calling loop for one chat turn: sends messages+tools to
    the supervisor, executes any requested tools, feeds results back, and
    repeats until the model returns a text-only response or MAX_TOOL_CALLS is
    hit. Returns (final_text, tool_trace, input_tokens, output_tokens) --
    tool_trace is what's persisted to rca_messages.tool_trace for the
    sidebar's replay view; the token counts are summed across every
    provider.send() call this turn made (a turn can span several rounds of
    tool-calling, each its own Anthropic API call with its own usage)."""
    tool_trace: list[dict] = []
    working_messages = list(messages)
    input_tokens = 0
    output_tokens = 0

    for _ in range(MAX_TOOL_CALLS):
        response = await provider.send(working_messages, TOOL_DEFINITIONS, SYSTEM_PROMPT)
        working_messages.append({"role": "assistant", "content": response.content})
        input_tokens += response.usage.input_tokens
        output_tokens += response.usage.output_tokens

        if response.stop_reason != "tool_use":
            text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            return text, tool_trace, input_tokens, output_tokens

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            try:
                result = await _dispatch(block.name, block.input, db, dim, log_file)
                tool_trace.append({
                    "tool": block.name, "args": block.input,
                    "result_summary": _summarize(result),
                })
                tool_results.append({
                    "type": "tool_result", "tool_use_id": block.id,
                    "content": json.dumps(result, default=str),
                })
            except Exception as exc:
                tool_trace.append({
                    "tool": block.name, "args": block.input,
                    "result_summary": str(exc),
                })
                tool_results.append({
                    "type": "tool_result", "tool_use_id": block.id,
                    "content": str(exc), "is_error": True,
                })
        working_messages.append({"role": "user", "content": tool_results})

    return (
        "Reached the tool-call limit for this turn without a final answer.",
        tool_trace, input_tokens, output_tokens,
    )
