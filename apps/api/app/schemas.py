from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

ThinkingMode = Literal["off", "low", "medium", "high"]
ThinkingModeTurn = Literal["session", "off", "low", "medium", "high"]


class SessionCreate(BaseModel):
    title: str
    description: str | None = None
    workspace_path: str | None = None
    model_preset: str | None = None
    system_prompt_preset: str | None = None
    thinking_mode: ThinkingMode = "medium"
    message_prefix_prompt: str = ""
    provider_id: str | None = None
    model_id: str | None = None
    # Per-session chat UI toggles — saved so the chat header restores
    # its checkbox state when the user switches sessions.
    hide_system_messages: bool = False
    run_in_background: bool = False
    force_search_next: bool = False
    bypass_search_cache_next: bool = False
    context_mode: Literal["full", "skill_state"] = "full"


class SessionUpdate(BaseModel):
    title: str | None = None
    description: str | None = None
    workspace_path: str | None = None
    # ``stalled`` is set by the orchestrator when it detects a
    # repetition loop (see ``_maybe_mark_session_stalled``). ``active``
    # is restored automatically as soon as a fresh user message lands.
    status: Literal["active", "archived", "deleted", "stalled"] | None = None
    thinking_mode: ThinkingMode | None = None
    message_prefix_prompt: str | None = None
    provider_id: str | None = None
    model_id: str | None = None
    hide_system_messages: bool | None = None
    run_in_background: bool | None = None
    force_search_next: bool | None = None
    bypass_search_cache_next: bool | None = None
    # Per-session override of the global ``context_mode``. ``None``
    # leaves the session inheriting the AppConfig default; ``"full"``
    # or ``"skill_state"`` overrides it explicitly.
    context_mode: Literal["full", "skill_state"] | None = None


class SessionOut(BaseModel):
    id: str
    title: str
    description: str | None = None
    created_at: str
    updated_at: str
    status: str
    workspace_path: str | None = None
    last_window_id: str | None = None
    total_message_count: int
    total_token_count: int
    thinking_mode: ThinkingMode
    message_prefix_prompt: str
    provider_id: str | None = None
    model_id: str | None = None
    hide_system_messages: bool = False
    run_in_background: bool = False
    force_search_next: bool = False
    bypass_search_cache_next: bool = False
    context_mode: Literal["full", "skill_state"] = "full"


class MessagePrefixTemplateCreate(BaseModel):
    name: str
    prompt: str


class MessagePrefixTemplateOut(BaseModel):
    id: str
    name: str
    prompt: str
    created_at: str
    updated_at: str


class MessageCreate(BaseModel):
    content: str
    thinking_mode: ThinkingModeTurn | None = "session"
    provider_id: str | None = None
    model_id: str | None = None
    # When True, the auto-search router always runs regardless of the
    # global policy (off / auto / always).  Used by the UI's per-message
    # "Force web search" toggle.
    force_search: bool = False
    # When True, the auto-search router skips the local cache and always
    # fetches fresh results.
    bypass_search_cache: bool = False
    # SKILL.state: explicit skill to activate for this turn. When set,
    # the orchestrator swaps the append-only chat history for the
    # (spec, state, observation) bundle per arXiv:2608.26263.
    active_skill: str | None = None
    # Per-turn override of the global ``context_mode`` selector. When
    # ``"skill_state"`` the auto-router is enabled; when ``"full"`` the
    # chat history is always replayed. ``None`` inherits from the
    # session's ``context_mode`` field.
    context_mode_override: str | None = None


class RunCreate(BaseModel):
    content: str


class RunOut(BaseModel):
    id: str
    session_id: str
    window_id: str | None = None
    user_message_id: str | None = None
    result_message_id: str | None = None
    task_text: str
    status: Literal["queued", "running", "completed", "failed", "canceled"]
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None
    progress_json: dict[str, Any] = Field(default_factory=dict)
    error_text: str | None = None


class RunEventOut(BaseModel):
    id: str
    session_id: str
    run_id: str
    step_index: int | None = None
    event_type: str
    title: str
    detail: str
    payload_json: dict[str, Any] = Field(default_factory=dict)
    timestamp: str


class RunArtifactOut(BaseModel):
    id: str
    session_id: str
    run_id: str
    artifact_id: str | None = None
    step_index: int | None = None
    stage: str
    title: str
    path: str
    metadata_json: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class EvolutionStart(BaseModel):
    prompt: str = ""
    max_generations: int = Field(default=1, ge=1, le=20)
    mode: Literal["conservative", "experimental", "tests-only"] = "conservative"
    stop_on_failure: bool = True


class EvolutionRunOut(BaseModel):
    id: str
    prompt: str
    status: Literal["queued", "running", "completed", "failed", "canceled"]
    mode: str
    max_generations: int
    stop_on_failure: bool
    current_generation: int
    parent_generation: int | None = None
    child_generation: int | None = None
    lineage_root_path: str
    parent_repo_path: str
    child_repo_path: str | None = None
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None
    progress_json: dict[str, Any] = Field(default_factory=dict)
    score_json: dict[str, Any] = Field(default_factory=dict)
    error_text: str | None = None


class EvolutionEventOut(BaseModel):
    id: str
    run_id: str
    generation: int | None = None
    event_type: str
    title: str
    detail: str
    payload_json: dict[str, Any] = Field(default_factory=dict)
    timestamp: str


class EvolutionGenerationOut(BaseModel):
    generation: int
    name: str
    status: Literal["passed", "failed", "unknown"]
    active: bool
    artifact_dir: str
    child_repo_path: str | None = None
    prompt: str = ""
    improvement_summary: str = ""
    mode: str | None = None
    created_at: str | None = None
    self_test_ok: bool | None = None
    tests_ok: bool | None = None
    has_handoff: bool = False
    deletable: bool


class EvolutionDeleteResponse(BaseModel):
    ok: bool
    deleted_generation: int
    active_generation: int | None = None


class EvolutionCopyToRootResponse(BaseModel):
    ok: bool
    generation: int
    root_repo_path: str
    source_repo_path: str
    copied_at: str


class MessageOut(BaseModel):
    id: str
    session_id: str
    window_id: str
    turn_id: str
    role: str
    timestamp: str
    content_text: str
    content_json: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    token_count: int
    message_type: str
    is_pinned: bool
    is_anchor: bool


class WindowState(BaseModel):
    session_id: str
    window_id: str
    token_limit: int
    used_tokens: int
    used_percent: float
    pre_rollover_threshold: float
    hard_rollover_threshold: float


class WorkspaceEventIn(BaseModel):
    event_type: Literal[
        "file_snapshot",
        "file_change",
        "diff_summary",
        "terminal_command",
        "terminal_output",
        "mcp_tool_call",
        "mcp_tool_result",
        "test_result",
        "build_error",
        "image_created",
        "artifact_created",
    ]
    payload_json: dict[str, Any] = Field(default_factory=dict)
    summary_text: str = ""


class CheckpointOut(BaseModel):
    id: str
    session_id: str
    source_window_id: str
    checkpoint_index: int
    created_at: str
    summary_text: str
    working_set_json: dict[str, Any]
    decisions_json: list[dict[str, Any]]
    open_questions_json: list[str]
    constraints_json: list[str]
    artifacts_json: list[dict[str, Any]]
    files_touched_json: list[str]
    retrieval_anchors_json: list[dict[str, Any]]


class PinRequest(BaseModel):
    message_id: str
    pinned: bool = True
    anchor: bool = False


class ProviderValidationRequest(BaseModel):
    base_url: str
    api_key: str | None = None
    endpoint: str = "/models"
    model_name: str | None = None
    timeout_sec: int | None = None


class ProviderValidationResponse(BaseModel):
    ok: bool
    detail: str
    models: list[str] = Field(default_factory=list)


# --- Provider / model CRUD ---------------------------------------------
class ModelEntryOut(BaseModel):
    id: str
    name: str
    display_name: str = ""
    context_window_size: int
    max_output_tokens: int
    temperature: float
    top_p: float
    extra_params_json: dict[str, Any] = Field(default_factory=dict)
    is_default: bool = False
    enabled: bool = True
    created_at: str = ""
    updated_at: str = ""


class ModelEntryCreate(BaseModel):
    # ``min_length=1`` rejects blank model ids so we don't end up with
    # ghost rows whose name can't be addressed again.  ``strip`` makes
    # sure "   " is treated as empty too.
    name: str = Field(min_length=1)
    display_name: str | None = None
    context_window_size: int = 128000
    max_output_tokens: int = 2048
    temperature: float = 0.2
    top_p: float = 1.0
    extra_params_json: dict[str, Any] = Field(default_factory=dict)
    is_default: bool = False
    enabled: bool = True

    @field_validator("name")
    @classmethod
    def _name_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("name must not be blank")
        return value


class ModelEntryUpdate(BaseModel):
    name: str | None = None
    display_name: str | None = None
    context_window_size: int | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    extra_params_json: dict[str, Any] | None = None
    is_default: bool | None = None
    enabled: bool | None = None


class ProviderOut(BaseModel):
    id: str
    name: str
    provider_name: str = "openai-compatible"
    base_url: str = ""
    endpoint: str = "/chat/completions"
    api_key: str = ""
    request_timeout_sec: int = 240
    enabled: bool = True
    notes: str = ""
    models: list[ModelEntryOut] = Field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


class ProviderCreate(BaseModel):
    name: str
    provider_name: str = "openai-compatible"
    base_url: str = ""
    endpoint: str = "/chat/completions"
    api_key: str = ""
    request_timeout_sec: int = 240
    enabled: bool = True
    notes: str = ""
    models: list[ModelEntryCreate] = Field(default_factory=list)


class ProviderUpdate(BaseModel):
    name: str | None = None
    provider_name: str | None = None
    base_url: str | None = None
    endpoint: str | None = None
    api_key: str | None = None
    request_timeout_sec: int | None = None
    enabled: bool | None = None
    notes: str | None = None


class ProviderListResponse(BaseModel):
    providers: list[ProviderOut]
    active_provider_id: str | None = None
    active_model_id: str | None = None


class ActiveSelectionUpdate(BaseModel):
    provider_id: str | None = None
    model_id: str | None = None


class MCPToolOut(BaseModel):
    name: str
    server_name: str
    mcp_tool_name: str
    description: str
    input_schema: dict[str, Any]


class MCPDiscoveryResponse(BaseModel):
    ok: bool
    tools: list[MCPToolOut] = Field(default_factory=list)
    errors: list[dict[str, str]] = Field(default_factory=list)


class ExportImportResponse(BaseModel):
    path: str


class ImportSessionRequest(BaseModel):
    archive_path: str
