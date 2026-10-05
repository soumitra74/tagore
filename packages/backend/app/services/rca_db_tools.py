async def get_eval_run(db, cycle_id: int, query_id: int) -> dict:
    result = await db.get_eval_run_detail(cycle_id, query_id)
    if result is None:
        raise ValueError(f"No eval_cycles row for cycle_id={cycle_id}")
    return result


async def resolve_chunks(db, dim: int, chunk_ids: list[str]) -> dict:
    return await db.resolve_chunks(chunk_ids, dim)


async def get_gold_pages(db, query_id: int) -> list[dict]:
    return await db.get_gold_pages(query_id)


async def check_text_contains(db, dim: int, chunk_id: str, needle: str) -> dict:
    resolved = await db.resolve_chunks([chunk_id], dim)
    chunk = resolved.get(chunk_id)
    if chunk is None:
        return {"found": False, "error": f"chunk_id {chunk_id} not found"}
    text = chunk.get("context_text") or chunk["chunk_text"]
    found = needle.lower() in text.lower()
    return {"found": found, "chunk_text_preview": text[:500]}


async def run_readonly_sql(db, query: str) -> list[dict]:
    return await db.execute_readonly(query)
