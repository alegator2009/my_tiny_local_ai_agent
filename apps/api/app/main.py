from __future__ import annotations

import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import ensure_data_dirs, load_app_config, settings
from .services.telemetry import default_registry as _telemetry_registry
from .services.config_reloader import default_reloader as _config_reloader
from .db import init_db
from .routes.chat import router as chat_router
from .routes.evolution import router as evolution_router
from .routes.message_prefix_templates import router as message_prefix_templates_router
from .routes.memory import router as memory_router
from .routes.skill_state import router as skill_state_router
from .routes.providers import router as providers_router
from .routes.runs import router as runs_router
from .routes.sessions import router as sessions_router
from .routes.settings import router as settings_router
from .routes.workspace import router as workspace_router
from .services.runs import run_worker_loop


def create_app() -> FastAPI:
    ensure_data_dirs()
    init_db()
    _ = load_app_config()

    app = FastAPI(title="AI Infinite Session API", version="0.1.0")

    origins = [o.strip() for o in settings.cors_allow_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics() -> dict[str, object]:
        """Return the in-process telemetry snapshot as JSON.

        Intended for liveness checks and ad-hoc dashboards.  The shape is
        intentionally stable:

        ``{"uptime_seconds": float, "dropped_series": int,
        "counters": {name: [{"value": int, "labels": dict}, ...]},
        "histograms": {name: [{"count": int, "sum": float, ...}, ...]},
        "declared": {...}}``

        Series that have never been observed are omitted to keep the
        payload small.
        """
        return _telemetry_registry.snapshot()

    @app.post("/admin/reload")
    def admin_reload() -> dict[str, object]:
        """Force-reload the AppConfig from disk and return the result.

        Useful for picking up edits to ``data/config.json`` without
        restarting the API.  The response includes the new mtime, a
        ``reloaded`` flag, and (on success) the new model configuration
        so the caller can verify the change took effect.
        """
        result = _config_reloader.reload_now()
        return {
            "reloaded": result.reloaded,
            "reason": result.reason,
            "mtime": result.mtime,
            "error": result.error,
            "config": (
                result.config.model_dump() if result.config is not None else None
            ),
        }

    @app.get("/admin/config")
    def admin_get_config() -> dict[str, object]:
        """Return the currently-active AppConfig plus reloader stats."""
        cfg = _config_reloader.get_config()
        return {
            "config": cfg.model_dump(),
            "stats": _config_reloader.stats(),
        }

    @app.on_event("startup")
    async def startup_background_worker() -> None:
        if not settings.background_worker_enabled:
            return
        stop_event = asyncio.Event()
        app.state.run_worker_stop_event = stop_event
        app.state.run_worker_task = asyncio.create_task(run_worker_loop(stop_event))

    @app.on_event("shutdown")
    async def shutdown_background_worker() -> None:
        task = getattr(app.state, "run_worker_task", None)
        stop_event = getattr(app.state, "run_worker_stop_event", None)
        if stop_event is not None:
            stop_event.set()
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    app.include_router(sessions_router)
    app.include_router(chat_router)
    app.include_router(runs_router)
    app.include_router(evolution_router)
    app.include_router(message_prefix_templates_router)
    app.include_router(memory_router)
    app.include_router(skill_state_router)
    app.include_router(settings_router)
    app.include_router(providers_router)
    app.include_router(workspace_router)

    return app


app = create_app()
