import logging
import os
import re
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import (
    routes_chat,
    routes_chunks,
    routes_configs,
    routes_corpus,
    routes_documents,
    routes_eval,
    routes_graph,
    routes_health,
    routes_rca,
)
from app.config import get_settings
from app.db import Database
from app.providers.factory import get_chat_provider
from app.routing.adapters import RoutedChatProvider
from app.routing.from_settings import create_router
from app.providers.rca_supervisor_provider import RcaSupervisorProvider
from app.services.ner import NERService

# EvalPanel.tsx/RevalPanel.tsx poll GET /api/eval/run/{job_id} every 2s
# (POLL_INTERVAL_MS) while a job is running -- a 30-minute eval run floods
# the access log with ~900 identical 200 OK lines. Hide only those
# successful polls; failures and every other route still log normally.
# Mirrors ollama_load_balancer.py's _SkipSuccessfulPolls filter.
_SUCCESSFUL_EVAL_POLL_LOG = re.compile(r'"GET /api/eval/run/[^"\s?]+(?:\?[^"]*)? HTTP/[^"]+" 200\b')


class _SkipEvalPollLogs(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return _SUCCESSFUL_EVAL_POLL_LOG.search(record.getMessage()) is None


logging.getLogger("uvicorn.access").addFilter(_SkipEvalPollLogs())


class _TimestampedFormatter(logging.Formatter):
    """Prepends a timestamp to whatever a handler's existing formatter
    produces, instead of replacing it -- keeps uvicorn's own level/color
    formatting (levelprefix, client_addr, request_line, status_code) intact,
    so this doesn't need to duplicate or track uvicorn's internal fmt
    strings. Without this, uvicorn's default access/error logs carry no
    timestamp at all, making it impossible to tell when a log line happened
    relative to other events (e.g. an eval cycle's started_at/completed_at)."""

    def __init__(self, inner: logging.Formatter | None):
        super().__init__()
        self._inner = inner or logging.Formatter("%(message)s")

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        return f"{timestamp} {self._inner.format(record)}"


for _logger_name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
    for _handler in logging.getLogger(_logger_name).handlers:
        _handler.setFormatter(_TimestampedFormatter(_handler.formatter))


# File-based logging for the RCA agent's get_backend_logs tool (see
# rca_log_tools.py) -- deliberately not Docker-socket access, which would
# grant the backend container daemon control-plane rights it doesn't
# otherwise need. Attached to the two uvicorn loggers directly (uvicorn.access
# and uvicorn.error don't propagate to root by default) and to root (so any
# logging.getLogger(__name__) call in application code, e.g. reval_run.py's
# logger.warning, is captured too via normal propagation). Bare "uvicorn" is
# deliberately excluded: uvicorn.error has propagate=True by default, so its
# records already reach "uvicorn" via propagation -- attaching the handler
# there too would double-fire on every uvicorn.error record.
_rca_log_path = get_settings().rca_log_file
os.makedirs(os.path.dirname(_rca_log_path), exist_ok=True)
_rca_file_handler = RotatingFileHandler(_rca_log_path, maxBytes=10_000_000, backupCount=3)
_rca_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
for _logger_name in ("uvicorn.access", "uvicorn.error", ""):
    logging.getLogger(_logger_name).addHandler(_rca_file_handler)


def create_app() -> FastAPI:
    app = FastAPI(title="Tagore RAG API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(routes_health.router)
    app.include_router(routes_documents.router)
    app.include_router(routes_chat.router)
    app.include_router(routes_configs.router)
    app.include_router(routes_corpus.router)
    app.include_router(routes_eval.router)
    app.include_router(routes_chunks.router)
    app.include_router(routes_graph.router)
    app.include_router(routes_rca.router)

    @app.on_event("startup")
    async def startup() -> None:
        settings = get_settings()
        db = Database(settings.database_url, settings.rca_readonly_database_url)
        await db.connect()
        await db.migrate(settings)
        await db.abort_running_eval_cycles()
        app.state.settings = settings
        app.state.db = db
        app.state.router = create_router(settings)
        if app.state.router is not None:
            app.state.chat_provider = RoutedChatProvider(app.state.router)
        else:
            app.state.chat_provider = get_chat_provider(settings)
        app.state.embedding_provider_cache = {}
        app.state.reranker_cache = {}
        app.state.eval_jobs = {}
        app.state.judge = None
        app.state.ner = NERService()
        await app.state.ner.warm_up()
        app.state.rca_provider = RcaSupervisorProvider(settings)
        app.state.ingestion_in_progress = False
        # Counted (not boolean) so it can be compared against the
        # configurable settings.eval_cycle_concurrency limit -- see
        # routes_eval.py's start_eval_run/start_reval_run guards.
        app.state.eval_running_count = 0

    @app.on_event("shutdown")
    async def shutdown() -> None:
        await app.state.db.close()

    return app


app = create_app()
