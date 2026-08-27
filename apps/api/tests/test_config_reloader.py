"""Tests for the hot-reloadable config reloader."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from app.config import AppConfig, save_app_config
from app.services.config_reloader import ConfigReloader, ReloadResult
from app.services.prompt_cache import PromptCache
from app.services.schema_cache import SchemaCache


# --- helpers ----------------------------------------------------------------


@pytest.fixture
def tmp_config_file(tmp_path: Path) -> Path:
    path = tmp_path / "config.json"
    cfg = AppConfig()
    save_app_config_to_path(cfg, path)
    return path


def save_app_config_to_path(cfg: AppConfig, path: Path) -> None:
    path.write_text(cfg.model_dump_json(indent=2), encoding="utf-8")


def write_raw_config(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# --- basic load -------------------------------------------------------------


def test_ensure_loaded_returns_config(tmp_config_file: Path) -> None:
    reloader = ConfigReloader(path=tmp_config_file)
    cfg = reloader.ensure_loaded()
    assert isinstance(cfg, AppConfig)


def test_ensure_loaded_is_idempotent(tmp_config_file: Path) -> None:
    reloader = ConfigReloader(path=tmp_config_file)
    a = reloader.ensure_loaded()
    b = reloader.ensure_loaded()
    # Same instance -- second call does not re-read the file.
    assert a is b


def test_get_config_triggers_initial_load(tmp_config_file: Path) -> None:
    reloader = ConfigReloader(path=tmp_config_file)
    cfg = reloader.get_config()
    assert isinstance(cfg, AppConfig)


# --- reload detection -------------------------------------------------------


def test_maybe_reload_is_noop_when_unchanged(tmp_config_file: Path) -> None:
    reloader = ConfigReloader(path=tmp_config_file)
    reloader.ensure_loaded()
    result = reloader.maybe_reload()
    assert result.reloaded is False
    assert result.reason == "unchanged"


def test_maybe_reload_detects_mtime_change(tmp_config_file: Path) -> None:
    reloader = ConfigReloader(path=tmp_config_file)
    reloader.ensure_loaded()
    # Modify the file with a new mtime.
    time.sleep(0.01)  # ensure mtime changes on filesystems with low resolution
    cfg = AppConfig(system_prompt="updated")
    save_app_config_to_path(cfg, tmp_config_file)
    result = reloader.maybe_reload()
    assert result.reloaded is True
    assert result.reason == "mtime_changed"
    assert result.config is not None
    assert result.config.system_prompt == "updated"


def test_reload_now_ignores_mtime(tmp_config_file: Path) -> None:
    reloader = ConfigReloader(path=tmp_config_file)
    reloader.ensure_loaded()
    cfg = AppConfig(system_prompt="forced")
    save_app_config_to_path(cfg, tmp_config_file)
    result = reloader.reload_now()
    assert result.reloaded is True
    assert result.reason == "forced"


# --- error handling ---------------------------------------------------------


def test_reload_returns_error_on_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text("{ this is not json", encoding="utf-8")
    reloader = ConfigReloader(path=path)
    result = reloader.maybe_reload()
    assert result.reloaded is False
    assert result.reason == "json_error"
    assert result.error is not None


def test_reload_returns_error_on_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "nope.json"
    reloader = ConfigReloader(path=path)
    result = reloader.maybe_reload()
    assert result.reloaded is False
    assert result.reason == "file_not_found"


def test_reload_returns_error_on_pydantic_validation(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    # Invalid: pre_rollover_threshold must be a float, not a string.
    write_raw_config(path, {"rollover_config": {"pre_rollover_threshold": "oops"}})
    reloader = ConfigReloader(path=path)
    result = reloader.maybe_reload()
    assert result.reloaded is False
    assert result.reason == "validation_error"
    assert result.error is not None


# --- hooks ------------------------------------------------------------------


def test_on_reload_hook_is_invoked(tmp_config_file: Path) -> None:
    reloader = ConfigReloader(path=tmp_config_file)
    reloader.ensure_loaded()
    seen: list[AppConfig] = []
    reloader.add_on_reload_hook(lambda cfg: seen.append(cfg))
    cfg = AppConfig(system_prompt="hooked")
    save_app_config_to_path(cfg, tmp_config_file)
    reloader.maybe_reload()
    assert len(seen) == 1
    assert seen[0].system_prompt == "hooked"


def test_default_invalidate_caches_drops_prompt_and_schema_cache(tmp_path: Path) -> None:
    """The default hook should clear both the prompt and schema caches so
    stale state derived from the old config is not reused."""
    from app.services.prompt_cache import default_cache as _prompt_cache_default
    from app.services.schema_cache import default_cache as _schema_cache_default

    # Seed the caches via module singletons.
    _prompt_cache_default.put(("k", "m", "t"), "v")
    _schema_cache_default.get_or_compute(
        server_slug="s", tool_name="t", raw_schema={"type": "object"}
    )
    assert _prompt_cache_default.stats()["size"] >= 1
    assert _schema_cache_default.stats()["size"] >= 1

    path = tmp_path / "config.json"
    cfg = AppConfig()
    save_app_config_to_path(cfg, path)
    reloader = ConfigReloader(path=path)
    reloader.reload_now()

    # Default hook should have invalidated both module singletons.
    assert _prompt_cache_default.stats()["size"] == 0
    assert _schema_cache_default.stats()["size"] == 0


def test_hook_exception_does_not_break_reload(tmp_config_file: Path) -> None:
    reloader = ConfigReloader(path=tmp_config_file)
    reloader.ensure_loaded()

    def bad_hook(_cfg: AppConfig) -> None:
        raise RuntimeError("boom")

    reloader.add_on_reload_hook(bad_hook)

    cfg = AppConfig(system_prompt="after-hook-failure")
    save_app_config_to_path(cfg, tmp_config_file)
    result = reloader.maybe_reload()
    assert result.reloaded is True
    assert result.config.system_prompt == "after-hook-failure"


# --- stats ------------------------------------------------------------------


def test_stats_track_reload_counts(tmp_config_file: Path) -> None:
    reloader = ConfigReloader(path=tmp_config_file)
    reloader.ensure_loaded()
    cfg = AppConfig(system_prompt="x")
    save_app_config_to_path(cfg, tmp_config_file)
    reloader.maybe_reload()
    reloader.maybe_reload()  # no-op
    stats = reloader.stats()
    assert stats["reload_count"] == 2  # initial + one real reload
    assert stats["last_reload_ok"] is True


def test_stats_track_failures(tmp_path: Path) -> None:
    path = tmp_path / "nope.json"
    reloader = ConfigReloader(path=path)
    reloader.maybe_reload()  # file not found
    stats = reloader.stats()
    assert stats["failure_count"] >= 1
    assert stats["last_reload_ok"] is False
    assert stats["last_error"] is not None


# --- thread safety ---------------------------------------------------------


def test_reload_is_thread_safe(tmp_config_file: Path) -> None:
    reloader = ConfigReloader(path=tmp_config_file)
    reloader.ensure_loaded()

    stop = threading.Event()

    def writer() -> None:
        i = 0
        while not stop.is_set():
            cfg = AppConfig(system_prompt=f"v{i}")
            save_app_config_to_path(cfg, tmp_config_file)
            i += 1
            time.sleep(0.001)

    def reader() -> None:
        for _ in range(50):
            cfg = reloader.get_config()
            assert isinstance(cfg, AppConfig)
            time.sleep(0.001)

    threads = [
        threading.Thread(target=writer),
        threading.Thread(target=reader),
        threading.Thread(target=reader),
    ]
    for t in threads:
        t.start()
    time.sleep(0.2)
    stop.set()
    for t in threads:
        t.join()


# --- /admin/reload endpoint ------------------------------------------------


def test_admin_reload_endpoint_returns_reloaded_config(tmp_path: Path) -> None:
    """End-to-end test against the FastAPI app."""
    from fastapi.testclient import TestClient
    # Redirect the default config path to a temp file so the test does not
    # touch the real on-disk config.
    from app import config as app_config_module
    from app.services import config_reloader as reloader_module
    from app.main import create_app

    target = tmp_path / "config.json"
    save_app_config_to_path(AppConfig(), target)
    # Patch both the settings path and the reloader path.
    original_settings_path = app_config_module.settings.app_config_path
    original_reloader_path = reloader_module.default_reloader._path
    app_config_module.settings.app_config_path = str(target)
    reloader_module.default_reloader = ConfigReloader(path=target)
    try:
        app = create_app()
        with TestClient(app) as client:
            # First reload -- file is already there but no reloader has been
            # loaded yet (create_app calls load_app_config).
            response = client.post("/admin/reload")
            assert response.status_code == 200
            body = response.json()
            assert body["reloaded"] is True
            assert body["config"] is not None
            assert "model_dump" not in body  # sanity: we returned a dict

            response2 = client.post("/admin/reload")
            assert response2.status_code == 200
            # Second reload: file unchanged -- should be a no-op.
            body2 = response2.json()
            assert body2["reloaded"] is True  # reload_now forces
    finally:
        app_config_module.settings.app_config_path = original_settings_path
        reloader_module.default_reloader = ConfigReloader(path=original_reloader_path)


def test_admin_get_config_endpoint_returns_stats() -> None:
    from fastapi.testclient import TestClient
    from app.main import create_app
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/admin/config")
    assert response.status_code == 200
    body = response.json()
    assert "config" in body
    assert "stats" in body
    assert "reload_count" in body["stats"]
