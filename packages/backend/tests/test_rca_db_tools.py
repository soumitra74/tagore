import pytest

from app.services.rca_db_tools import (
    check_text_contains,
    get_eval_run,
    get_gold_pages,
    resolve_chunks,
    run_readonly_sql,
)


class FakeDatabase:
    def __init__(self):
        self.readonly_calls = []

    async def get_eval_run_detail(self, cycle_id, query_id):
        if cycle_id == 62 and query_id == 1:
            return {"cycle_id": 62, "config_id": 13, "question": "serial number?"}
        return None

    async def resolve_chunks(self, chunk_ids, dim):
        assert dim == 768
        return {cid: {"chunk_text": f"text for {cid}", "page_number": 10} for cid in chunk_ids}

    async def get_gold_pages(self, query_id):
        return [{"filename": "PFR1.pdf", "page_number": 10}]

    async def execute_readonly(self, query):
        self.readonly_calls.append(query)
        return [{"count": 5}]


@pytest.mark.asyncio
async def test_get_eval_run_returns_detail():
    result = await get_eval_run(FakeDatabase(), 62, 1)
    assert result["question"] == "serial number?"


@pytest.mark.asyncio
async def test_get_eval_run_raises_for_unknown_run():
    with pytest.raises(ValueError, match="cycle_id=999"):
        await get_eval_run(FakeDatabase(), 999, 1)


@pytest.mark.asyncio
async def test_resolve_chunks_passes_dim_through():
    result = await resolve_chunks(FakeDatabase(), 768, ["chunk-1"])
    assert result["chunk-1"]["page_number"] == 10


@pytest.mark.asyncio
async def test_get_gold_pages_returns_list():
    result = await get_gold_pages(FakeDatabase(), 1)
    assert result == [{"filename": "PFR1.pdf", "page_number": 10}]


@pytest.mark.asyncio
async def test_check_text_contains_found():
    result = await check_text_contains(FakeDatabase(), 768, "chunk-1", "text for chunk-1")
    assert result["found"] is True


@pytest.mark.asyncio
async def test_check_text_contains_not_found():
    result = await check_text_contains(FakeDatabase(), 768, "chunk-1", "nonexistent phrase")
    assert result["found"] is False


@pytest.mark.asyncio
async def test_check_text_contains_unknown_chunk():
    class EmptyDatabase(FakeDatabase):
        async def resolve_chunks(self, chunk_ids, dim):
            return {}

    result = await check_text_contains(EmptyDatabase(), 768, "missing-chunk", "anything")
    assert result["found"] is False
    assert "not found" in result["error"]


@pytest.mark.asyncio
async def test_run_readonly_sql_delegates_to_db():
    db = FakeDatabase()
    result = await run_readonly_sql(db, "SELECT COUNT(*) FROM eval_runs")
    assert result == [{"count": 5}]
    assert db.readonly_calls == ["SELECT COUNT(*) FROM eval_runs"]
