from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# --- Backwards compatibility ------------------------------------------------
# Older imports referenced ``ModelConfig``.  We keep the name alive so
# existing tests and downstream code don't break; new code should use
# ``ProviderConfig`` + ``ModelEntry``.
class ModelConfig(BaseModel):
    provider_name: str = "openai-compatible"
    base_url: str = ""
    endpoint: str = "/chat/completions"
    api_key: str = ""
    model_name: str = "gpt-4o-mini"
    context_window_size: int = 128000
    max_output_tokens: int = 2048
    request_timeout_sec: int = 240
    temperature: float = 0.2
    top_p: float = 1.0
    extra_params_json: dict[str, Any] = Field(default_factory=dict)

    @field_validator("request_timeout_sec", mode="before")
    @classmethod
    def _normalize_timeout(cls, value: Any) -> int:
        try:
            t = int(float(str(value).strip()))
        except Exception:
            t = 240
        return max(5, min(t, 600))


class ModelEntry(BaseModel):
    """A single model hosted by a provider.  The chat UI picks one of
    these entries from the active provider, and the orchestrator turns
    it into an HTTP request."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str  # the model identifier sent in the ``model`` field
    display_name: str | None = ""
    context_window_size: int = 128000
    max_output_tokens: int = 2048
    temperature: float = 0.2
    top_p: float = 1.0
    extra_params_json: dict[str, Any] = Field(default_factory=dict)
    is_default: bool = False
    enabled: bool = True
    created_at: str = ""
    updated_at: str = ""

    @field_validator("display_name", mode="before")
    @classmethod
    def normalize_display_name(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @field_validator("context_window_size", mode="before")
    @classmethod
    def normalize_context_window(cls, value: Any) -> int:
        try:
            size = int(float(str(value).strip()))
        except Exception:
            size = 128000
        return max(1, size)

    model_config = {"extra": "ignore"}


class ProviderConfig(BaseModel):
    """A provider hosts one or more models behind a single OpenAI-compatible
    endpoint.  ``request_timeout_sec`` lives here because it is a property of
    the network call, not of the model itself."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str  # human label, e.g. "Local LM Studio"
    provider_name: str = "openai-compatible"  # protocol marker for forward-compat
    base_url: str = ""
    endpoint: str = "/chat/completions"
    api_key: str = ""
    request_timeout_sec: int = 240
    enabled: bool = True
    notes: str = ""
    models: list[ModelEntry] = Field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    @field_validator("base_url", mode="before")
    @classmethod
    def normalize_base_url(cls, value: Any) -> str:
        if value is None:
            return ""
        url = str(value).strip()
        # Fix common typo: http://host/:8317/v1 -> http://host:8317/v1
        url = re.sub(r"^(https?://[^/]+)/:(\d+)(/?.*)$", r"\1:\2\3", url)
        return url

    @field_validator("request_timeout_sec", mode="before")
    @classmethod
    def normalize_request_timeout(cls, value: Any) -> int:
        try:
            timeout = int(float(str(value).strip()))
        except Exception:
            timeout = 240
        # Keep bounds practical for UI and provider calls.
        return max(5, min(timeout, 600))

    def find_model(self, model_id: str | None) -> ModelEntry | None:
        if not model_id:
            return None
        for m in self.models:
            if m.id == model_id:
                return m
        return None

    def default_model(self) -> ModelEntry | None:
        enabled = [m for m in self.models if m.enabled]
        if not enabled:
            return None
        explicit = [m for m in enabled if m.is_default]
        if explicit:
            return explicit[0]
        return enabled[0]


class EmbeddingConfig(BaseModel):
    provider: str = "local-hash"
    model: str = "hash-1536"
    dimensions: int = 128
    batch_size: int = 32


class StorageConfig(BaseModel):
    backend: str = "local_sqlite_lancedb"
    chunking_strategy: str = "hybrid"
    chunk_size_target: int = 800
    chunk_overlap: int = 100
    index_refresh_mode: str = "batched_every_2"
    retention_mode: str = "keep_raw_forever"


class RetrievalConfig(BaseModel):
    mode: str = "hybrid"
    top_k: int = 8
    neighbor_prev: int = 1
    neighbor_next: int = 1
    rerank_mode: str = "cheap"
    memory_extraction_mode: str = "facts_decisions_tasks_entities"


class RolloverConfig(BaseModel):
    pre_rollover_threshold: float = 0.80
    hard_rollover_threshold: float = 0.92
    # Fraction of the remaining window that a single tool result may consume.
    # Lower values leave more room for the system prompt, recall, and the
    # model's own response; higher values let big results through at the cost
    # of less context for the rest of the turn.
    tool_budget_ratio: float = 0.30
    # Per-tool absolute character caps. The first matching key wins.
    per_tool_max_chars: dict[str, int] = Field(default_factory=dict)


class AutoSearchConfig(BaseModel):
    """Settings for the orchestrator's "google where I don't know" behaviour.

    The UI exposes the ``enabled`` flag as a single "Google where you don't know"
    checkbox; the rest are fine-grained knobs for power users.  When
    ``policy`` is ``auto`` the orchestrator only triggers a search when its
    cheap heuristic decides the user is asking for fresh, local or
    fact-grounded information.  ``always`` triggers on every user turn;
    ``off`` disables the feature entirely (the model still has the raw
    web-search MCP tool available).
    """

    enabled: bool = False
    policy: str = "auto"  # off | auto | always
    max_chars: int = 4000
    cache_ttl_sec: int = 60 * 60 * 6  # 6h
    max_per_turn: int = 1
    max_citations: int = 5
    summary_max_chars: int = 1200
    snippet_per_source_chars: int = 320
    include_snippets: bool = True
    include_full_content: bool = False
    # Optional bias passed to the native web-search MCP: "general" by default,
    # or one of the engines it understands ("bing", "brave", "duckduckgo").
    prefer_engine: str = ""
    # Heuristic thresholds — kept here so power-users can tune them.
    freshness_hints: list[str] = Field(default_factory=list)
    factual_hints: list[str] = Field(default_factory=list)
    opinion_hints: list[str] = Field(default_factory=list)

    @field_validator("policy", mode="before")
    @classmethod
    def normalize_policy(cls, value: Any) -> str:
        raw = str(value or "auto").strip().lower()
        if raw not in {"off", "auto", "always"}:
            return "auto"
        return raw

    @field_validator("max_chars", "summary_max_chars", "snippet_per_source_chars", mode="before")
    @classmethod
    def normalize_positive_int(cls, value: Any) -> int:
        try:
            n = int(float(str(value).strip()))
        except Exception:
            n = 0
        return max(0, n)

    @field_validator("max_per_turn", "max_citations", "cache_ttl_sec", mode="before")
    @classmethod
    def normalize_count(cls, value: Any) -> int:
        try:
            n = int(float(str(value).strip()))
        except Exception:
            n = 0
        return max(0, n)


class MCPConfig(BaseModel):
    enabled: bool = True
    servers: list[dict[str, Any]] = Field(default_factory=list)
    native_web_search_enabled: bool = False
    native_web_search_path: str = "mcp/web-search-mcp/codex-wrapper.mjs"
    native_web_search_timeout_sec: int = 45
    native_web_search_env: dict[str, str] = Field(default_factory=dict)
    # Skills MCP — exposes skills as separate tools so the underlying code does
    # not leak into the chat surface.
    skills_mcp_enabled: bool = False
    skills_mcp_path: str = "skills/wrapper.mjs"
    # Auto web search: a router-level "google where I don't know" policy
    # that the orchestrator enforces before the model is asked to answer.
    # Distinct from the native-web-search MCP, which is the actual search
    # backend; the auto-search router decides *when* to call it.
    auto_search: AutoSearchConfig = Field(default_factory=AutoSearchConfig)

    @field_validator("native_web_search_timeout_sec", mode="before")
    @classmethod
    def normalize_native_web_search_timeout(cls, value: Any) -> int:
        try:
            timeout = int(float(str(value).strip()))
        except Exception:
            timeout = 45
        return max(3, min(timeout, 600))


class AppConfig(BaseModel):
    # New: list of providers, each with its own model list.  See
    # ``ProviderConfig`` and ``ModelEntry``.
    providers: list[ProviderConfig] = Field(default_factory=list)
    # Selector: which provider/model is used by default for new chats
    # and for actions that don't carry an explicit override (e.g.
    # background runs, evolution tests).
    active_provider_id: str | None = None
    active_model_id: str | None = None

    # Legacy single-config fields are kept around so that older
    # settings JSON files keep loading — they are migrated to
    # ``providers`` on first load.  Anything written by this version of
    # the app drops them, but readers tolerate them.
    primary_model_config: dict[str, Any] | None = None
    summary_model_config: dict[str, Any] | None = None

    embedding_config: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    storage_config: StorageConfig = Field(default_factory=StorageConfig)
    retrieval_config: RetrievalConfig = Field(default_factory=RetrievalConfig)
    rollover_config: RolloverConfig = Field(default_factory=RolloverConfig)
    mcp_config: MCPConfig = Field(default_factory=MCPConfig)
    model_context_window_size_override: int | None = None
    # Context-mode selector — how the orchestrator builds the prompt
    # that goes to the model. Two flavours are supported:
    #
    #   * ``"full"`` (default) — the legacy path: replay the recent
    #     chat history (``messages`` rows), durable facts, working
    #     set, recall pack, and checkpoint summary in the system
    #     prompt. Cheap to operate but grows linearly with execution
    #     history and is the classic source of context poisoning on
    #     long horizons.
    #
    #   * ``"skill_state"`` — the SKILL.state path (arXiv:2608.26263).
    #     The model only ever sees the (spec, state, observation)
    #     bundle from a registered skill plus the current user turn.
    #     When the user's prompt matches a registered skill (overlap
    #     of >= 2 tokens between the prompt and the skill's
    #     description / whenToUse), the orchestrator activates the
    #     skill and rebuilds the prompt from the validated state. When
    #     no skill matches the model still receives the legacy
    #     ``"full"`` prompt — this preserves backwards compatibility
    #     and lets users gradually opt in per skill.
    context_mode: Literal["full", "skill_state"] = "full"
    system_prompt: str = (
        "You are an assistant inside AI Infinite Session. Keep continuity, "
        "respect durable facts, and use retrieval when confidence is low."
    )
    session_memory_profile: str = "default"

    @field_validator("context_mode", mode="before")
    @classmethod
    def normalize_context_mode(cls, value: Any) -> str:
        """Tolerate legacy / malformed values from older config.json
        files. Anything that is not exactly ``"skill_state"`` is
        normalised to ``"full"`` so the runtime never crashes on a
        typo."""
        raw = str(value or "full").strip().lower()
        if raw in {"full", "skill_state", "skill-state", "skillstate"}:
            return "skill_state" if raw != "full" else "full"
        return "full"

    @model_validator(mode="after")
    def clamp_context_window_override(self) -> "AppConfig":
        if self.model_context_window_size_override is None:
            return self

        real_window = self._active_context_window_size() or 1
        override = max(1, int(self.model_context_window_size_override))
        self.model_context_window_size_override = min(override, real_window)
        return self

    def _active_context_window_size(self) -> int | None:
        provider, model = self.active_pair()
        if provider and model:
            return max(1, int(model.context_window_size))
        if self.providers:
            for p in self.providers:
                m = p.default_model()
                if m:
                    return max(1, int(m.context_window_size))
        if self.primary_model_config:
            try:
                return max(1, int(self.primary_model_config.get("context_window_size") or 1))
            except Exception:
                return None
        return None

    # ------------------------------------------------------------------
    # Active provider / model helpers
    # ------------------------------------------------------------------
    def active_provider(self) -> ProviderConfig | None:
        if self.active_provider_id:
            for p in self.providers:
                if p.id == self.active_provider_id:
                    return p
        # Fall back to first enabled provider, then first provider.
        enabled = [p for p in self.providers if p.enabled]
        return (enabled or self.providers)[0] if self.providers else None

    def active_model(self) -> ModelEntry | None:
        provider = self.active_provider()
        if provider is None:
            return None
        if self.active_model_id:
            m = provider.find_model(self.active_model_id)
            if m and m.enabled:
                return m
        return provider.default_model()

    def active_pair(self) -> tuple[ProviderConfig | None, ModelEntry | None]:
        p = self.active_provider()
        if p is None:
            return None, None
        m = self.active_model()
        return p, m

    def resolve_pair(
        self, provider_id: str | None, model_id: str | None
    ) -> tuple[ProviderConfig | None, ModelEntry | None]:
        """Resolve a (provider, model) pair from explicit IDs, falling back
        to the active pair when IDs are missing or unknown."""

        provider: ProviderConfig | None = None
        if provider_id:
            for p in self.providers:
                if p.id == provider_id:
                    provider = p
                    break
        if provider is None:
            return self.active_pair()

        model: ModelEntry | None = None
        if model_id:
            model = provider.find_model(model_id)
        if model is None:
            model = provider.default_model()
        return provider, model


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")

    data_dir: str = "data"
    sqlite_path: str = "data/app.db"
    lancedb_path: str = "data/lancedb"
    app_config_path: str = "data/config.json"
    cors_allow_origins: str = "*"
    background_worker_enabled: bool = True


settings = Settings()


def ensure_data_dirs() -> None:
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.lancedb_path).mkdir(parents=True, exist_ok=True)
    Path(settings.sqlite_path).parent.mkdir(parents=True, exist_ok=True)


def _migrate_legacy_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Translate the pre-CRUD shape (``primary_model_config`` etc.) into
    a ``providers`` list so old settings.json files keep loading and
    users don't lose their existing model."""

    if not isinstance(raw, dict):
        return raw
    if "providers" in raw:
        return raw

    legacy = raw.get("primary_model_config")
    if not isinstance(legacy, dict) or not legacy:
        # Either no legacy config or an empty one — nothing to migrate.
        raw.setdefault("providers", [])
        return raw

    provider_id = str(uuid.uuid4())
    model_id = str(uuid.uuid4())
    provider_name = str(legacy.get("provider_name") or "openai-compatible")
    name = str(legacy.get("name") or f"{provider_name} (legacy)")
    models = [
        {
            "id": model_id,
            "name": str(legacy.get("model_name") or "gpt-4o-mini"),
            "display_name": str(legacy.get("display_name") or legacy.get("model_name") or "gpt-4o-mini"),
            "context_window_size": int(legacy.get("context_window_size") or 128000),
            "max_output_tokens": int(legacy.get("max_output_tokens") or 2048),
            "temperature": float(legacy.get("temperature") or 0.2),
            "top_p": float(legacy.get("top_p") or 1.0),
            "extra_params_json": dict(legacy.get("extra_params_json") or {}),
            "is_default": True,
            "enabled": True,
        }
    ]
    raw["providers"] = [
        {
            "id": provider_id,
            "name": name,
            "provider_name": provider_name,
            "base_url": str(legacy.get("base_url") or ""),
            "endpoint": str(legacy.get("endpoint") or "/chat/completions"),
            "api_key": str(legacy.get("api_key") or ""),
            "request_timeout_sec": int(legacy.get("request_timeout_sec") or 240),
            "enabled": True,
            "notes": "Imported from legacy primary_model_config.",
            "models": models,
        }
    ]
    if not raw.get("active_provider_id"):
        raw["active_provider_id"] = provider_id
    if not raw.get("active_model_id"):
        raw["active_model_id"] = model_id
    return raw


def load_app_config() -> AppConfig:
    ensure_data_dirs()
    path = Path(settings.app_config_path)
    if not path.exists():
        cfg = AppConfig()
        save_app_config(cfg)
        return cfg
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw = _migrate_legacy_config(raw)
    try:
        return AppConfig.model_validate(raw)
    except Exception:
        # If validation still fails (e.g. genuinely corrupt file),
        # start fresh rather than crash the API.
        cfg = AppConfig()
        save_app_config(cfg)
        return cfg


def save_app_config(cfg: AppConfig) -> None:
    path = Path(settings.app_config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = cfg.model_dump(exclude_none=True, mode="json")
    # Strip legacy fields so we don't write them back to disk; readers
    # tolerate them, but keeping the on-disk shape clean makes the
    # intent obvious.
    payload.pop("primary_model_config", None)
    payload.pop("summary_model_config", None)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _resolve_native_web_search_path(raw_path: str) -> str:
    path = Path(raw_path)
    if path.is_absolute():
        return str(path)

    # The source tree has a different depth in development
    # (repo/apps/api/app/config.py) and in the Docker image
    # (/app/app/config.py).  Search from both the working directory and every
    # source parent instead of relying on a fixed parents[n] index.
    source_parents = list(Path(__file__).resolve().parents)
    candidates = [Path.cwd() / path, *(parent / path for parent in source_parents), Path("/app") / path]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str((Path.cwd() / path).resolve())


def _resolve_skills_mcp_path(raw_path: str) -> str:
    path = Path(raw_path)
    if path.is_absolute():
        return str(path)

    source_parents = list(Path(__file__).resolve().parents)
    candidates = [Path.cwd() / path, *(parent / path for parent in source_parents), Path("/app") / path]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str((Path.cwd() / path).resolve())


def effective_mcp_servers(cfg: AppConfig) -> list[dict[str, Any]]:
    servers = list(cfg.mcp_config.servers)
    if not cfg.mcp_config.enabled:
        return servers

    if cfg.mcp_config.native_web_search_enabled and not any(
        str(s.get("name", "")).strip() == "native-web-search" for s in servers
    ):
        resolved_path = _resolve_native_web_search_path(cfg.mcp_config.native_web_search_path)
        native_env = dict(cfg.mcp_config.native_web_search_env)
        native_timeout_sec = max(3, int(cfg.mcp_config.native_web_search_timeout_sec))
        timeout_ms = str(native_timeout_sec * 1000)
        # MCP request timeout should be higher than the internal
        # web-search / network timeout so the wrapper can finish a
        # full ``full_web_search`` (Brave → DuckDuckGo → Bing →
        # Google) without being cut off at the JSON-RPC layer. The
        # default of 25 s used to be too tight for the per-engine
        # roundtrip plus a body-extraction pass; raised to a flat
        # 90 s floor so the request can complete even on the first
        # retry.
        mcp_request_timeout_sec = min(600, max(90, native_timeout_sec + 45))
        mcp_startup_timeout_sec = min(120, max(20, native_timeout_sec))
        # Keep content extraction and search engine request budgets in sync.
        native_env["DEFAULT_TIMEOUT"] = timeout_ms
        native_env["SEARCH_TIMEOUT_MS"] = timeout_ms
        native_server = {
            "name": "native-web-search",
            "transport": "stdio",
            "message_mode": "line",
            "command": "node",
            "args": [resolved_path],
            "env": native_env,
            "startup_timeout_sec": mcp_startup_timeout_sec,
            "request_timeout_sec": mcp_request_timeout_sec,
        }
        servers = [native_server, *servers]

    # Skills MCP — exposes skills as separate tools so the underlying code does
    # not leak into the chat surface.
    if cfg.mcp_config.skills_mcp_enabled and not any(
        str(s.get("name", "")).strip() == "skills-mcp" for s in servers
    ):
        skills_path = _resolve_skills_mcp_path(cfg.mcp_config.skills_mcp_path)
        skills_server = {
            "name": "skills-mcp",
            "transport": "stdio",
            "message_mode": "line",
            "command": "node",
            "args": [skills_path],
            "env": {},
            "startup_timeout_sec": 12,
            "request_timeout_sec": 30,
        }
        servers = [skills_server, *servers]

    return servers


def effective_primary_context_window_size(cfg: AppConfig) -> int:
    """Window size used when starting new chat windows.  Comes from the
    active model; falls back to the first model of any provider if the
    active pair can't be resolved."""

    pair_size = cfg._active_context_window_size()
    if pair_size:
        real_window = max(1, int(pair_size))
    else:
        real_window = 128000
    override = cfg.model_context_window_size_override
    if override is None:
        return real_window
    return min(real_window, max(1, int(override)))