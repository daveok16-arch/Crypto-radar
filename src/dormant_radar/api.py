"""HTTP service exposing collected wake-ups.

Read endpoints are served from SQLite, so the API stays responsive even
while a scan is running. `POST /scan` triggers a synchronous scan for
manual use; the background worker is the normal way data arrives.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from .chain import ChainError
from .config import Settings
from .scanner import scan_once
from .store import Store

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.store = Store(settings.db_path)
        try:
            yield
        finally:
            app.state.store.close()

    app = FastAPI(title="Dormant Radar", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/events")
    def events(
        limit: int = Query(50, ge=1, le=500),
        min_value_sats: int = Query(0, ge=0),
        since_height: int | None = Query(None, ge=0),
        hypothesis: str | None = Query(None, pattern="^(lost|holding|structural)$"),
    ) -> dict:
        items = app.state.store.list_wakeups(
            limit=limit,
            min_value_sats=min_value_sats,
            since_height=since_height,
            hypothesis=hypothesis,
        )
        return {"count": len(items), "events": items}

    @app.get("/stats")
    def stats() -> dict:
        return app.state.store.stats()

    @app.post("/scan")
    def scan() -> dict:
        try:
            result = scan_once(app.state.settings, store=app.state.store)
        except ChainError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {
            "scanned_from": result.scanned_from,
            "scanned_to": result.scanned_to,
            "blocks_examined": result.blocks_examined,
            "transactions_examined": result.transactions_examined,
            "wakeups_found": len(result.wakeups),
            "new_wakeups": result.new_wakeups,
            "errors": result.errors[:20],
        }

    return app


app = create_app()