from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/api/health")
async def health(request: Request) -> dict:
    try:
        tables = await request.app.state.db.list_tables()
        return {"status": "ok", "tables": tables}
    except Exception as exc:
        return {"status": "error", "tables": [], "error": str(exc)}
