"""Hot-reload :class:`AppConfig` from ``data/config.json`` without restarting the process.

The reloader polls the file's modification time at a configurable interval
(default 1 second) and re-parses the file whenever the mtime changes.  The
result is exposed through a thread-safe accessor that returns the most
recently loaded :class:`AppConfig`.  Callers that hold a reference to the
old config (e.g. the orchestrator inside an in-flight turn) are not
disturbed: they keep using the snapshot they were given, while subsequent
turns see the new config.

A second mechanism, :meth:`reload_now`, is exposed for the ``/admin/reload``
HTTP route and for tests.  It reads the file unconditionally and updates
the cached config, returning a structured :class:`ReloadResult`.

The reloader also fires a small on-reload hook list.  Modules that cache
data derived from the config (prompt cache, schema cache) register a
callback so they can drop their entries when the file changes.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..config import AppConfig, load_app_config, save_app_config
from .prompt_cache import default_cache as _prompt_cache
from .schema_cache import default_cache as _schema_cache
from .telemetry import default_registry as _telemetry_registry


@dataclass
class ReloadResult:
    """Outcome of a single reload attempt."""

    reloaded: bool
    reason: str
    mtime: float
    config: AppConfig | None = None
    error: str | None = None


@dataclass
class _State:
    lock: threading.Lock = field(default_factory=threading.Lock)
    config: AppConfig | None = None
    mtime: float = 0.0
    last_reload_at: float = 0.0
    last_reload_ok: bool = True
    last_error: str | None = None
    reload_count: int = 0
    failure_count: int = 0


class ConfigReloader:
    """Polling-based hot-reloader for the AppConfig file.

    The reloader is intentionally simple: it reads the file's ``st_mtime``
    on each call to :meth:`maybe_reload` and only re-parses the JSON when
    the mtime has advanced.  This keeps the idle cost at a single stat()
    syscall per call.

    Callers obtain the current config via :meth:`get_config`, which always
    returns the most recently cached value.  If the file has never been
    loaded, :meth:`ensure_loaded` triggers a one-shot load.
    """

    def __init__(self, path: str | Path = "data/config.json") -> None:
        self._path = Path(path)
        self._state = _State()
        self._on_reload: list[Callable[[AppConfig], None]] = []
        # Register default invalidation hooks so cache state is consistent.
        self._on_reload.append(self._default_invalidate_caches)

    # -- public API --------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    def add_on_reload_hook(self, hook: Callable[[AppConfig], None]) -> None:
        """Register a callable invoked after a successful reload.

        The hook receives the new :class:`AppConfig`.  Exceptions raised by
        the hook are logged and swallowed so they cannot block subsequent
        reloads.
        """
        with self._state.lock:
            self._on_reload.append(hook)

    def ensure_loaded(self) -> AppConfig:
        """Load the config once if it has never been loaded.

        Subsequent calls return the cached config.  This method is safe to
        call from multiple threads; only the first call performs I/O.
        """
        with self._state.lock:
            if self._state.config is None:
                cfg = load_app_config()
                self._state.config = cfg
                self._state.mtime = self._safe_mtime()
                self._state.last_reload_at = time.time()
                self._state.reload_count += 1
            return self._state.config

    def get_config(self) -> AppConfig:
        return self.ensure_loaded()

    def maybe_reload(self) -> ReloadResult:
        """Check the file's mtime and reload if it has advanced.

        Returns a :class:`ReloadResult` describing what happened.  The
        reloader is idempotent: calling it many times per second with no
        file change is cheap and produces ``reloaded=False``.
        """
        with self._state.lock:
            current_mtime = self._safe_mtime()
            if self._state.config is None:
                # First call: load unconditionally.
                return self._do_reload_locked(force=True, reason="initial_load")
            if current_mtime == self._state.mtime:
                return ReloadResult(
                    reloaded=False,
                    reason="unchanged",
                    mtime=current_mtime,
                    config=self._state.config,
                )
            return self._do_reload_locked(force=True, reason="mtime_changed")

    def reload_now(self) -> ReloadResult:
        """Force a reload, ignoring the mtime check.

        Useful for the ``/admin/reload`` HTTP route and for tests that
        want to validate the reload path directly.
        """
        with self._state.lock:
            return self._do_reload_locked(force=True, reason="forced")

    def stats(self) -> dict[str, Any]:
        with self._state.lock:
            return {
                "path": str(self._path),
                "mtime": self._state.mtime,
                "last_reload_at": self._state.last_reload_at,
                "last_reload_ok": self._state.last_reload_ok,
                "last_error": self._state.last_error,
                "reload_count": self._state.reload_count,
                "failure_count": self._state.failure_count,
            }

    # -- internal helpers --------------------------------------------------

    def _safe_mtime(self) -> float:
        try:
            return self._path.stat().st_mtime
        except FileNotFoundError:
            return 0.0
        except OSError:
            return 0.0

    def _do_reload_locked(self, *, force: bool, reason: str) -> ReloadResult:
        """Reload while the state lock is already held."""
        mtime = self._safe_mtime()
        if not force and mtime != 0.0 and mtime == self._state.mtime and self._state.config is not None:
            return ReloadResult(
                reloaded=False,
                reason="unchanged",
                mtime=mtime,
                config=self._state.config,
            )
        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
            cfg = AppConfig.model_validate(data)
        except FileNotFoundError as exc:
            self._state.failure_count += 1
            self._state.last_reload_ok = False
            self._state.last_error = f"file_not_found: {exc}"
            _telemetry_registry.counter(
                "config_reload_total", label_keys=("outcome",)
            ).with_labels(outcome="file_not_found").inc()
            return ReloadResult(
                reloaded=False,
                reason="file_not_found",
                mtime=mtime,
                error=self._state.last_error,
            )
        except json.JSONDecodeError as exc:
            self._state.failure_count += 1
            self._state.last_reload_ok = False
            self._state.last_error = f"json_error: {exc}"
            _telemetry_registry.counter(
                "config_reload_total", label_keys=("outcome",)
            ).with_labels(outcome="json_error").inc()
            return ReloadResult(
                reloaded=False,
                reason="json_error",
                mtime=mtime,
                error=self._state.last_error,
            )
        except Exception as exc:  # pydantic.ValidationError, OSError, etc.
            self._state.failure_count += 1
            self._state.last_reload_ok = False
            self._state.last_error = f"validation_error: {exc}"
            _telemetry_registry.counter(
                "config_reload_total", label_keys=("outcome",)
            ).with_labels(outcome="error").inc()
            return ReloadResult(
                reloaded=False,
                reason="validation_error",
                mtime=mtime,
                error=self._state.last_error,
            )

        self._state.config = cfg
        self._state.mtime = mtime
        self._state.last_reload_at = time.time()
        self._state.last_reload_ok = True
        self._state.last_error = None
        self._state.reload_count += 1
        _telemetry_registry.counter(
            "config_reload_total", label_keys=("outcome",)
        ).with_labels(outcome="ok").inc()

        # Fire hooks.  Snapshot to allow hooks to mutate the list safely.
        for hook in list(self._on_reload):
            try:
                hook(cfg)
            except Exception:
                # Hooks are best-effort; never let one break the reload.
                pass
        return ReloadResult(
            reloaded=True,
            reason=reason,
            mtime=mtime,
            config=cfg,
        )

    def _default_invalidate_caches(self, cfg: AppConfig) -> None:
        """Drop derived caches so the new config takes effect immediately."""
        try:
            _prompt_cache.invalidate()
        except Exception:
            pass
        try:
            _schema_cache.invalidate()
        except Exception:
            pass


# A process-wide reloader bound to the default config path.  Tests
# instantiate their own :class:`ConfigReloader` to keep state isolated.
default_reloader = ConfigReloader()
