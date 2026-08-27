from __future__ import annotations

import asyncio
import json
import re
import uuid
from pathlib import Path
from typing import Any

import httpx

from ..config import effective_mcp_servers, load_app_config
from ..db import execute, fetch_all, fetch_one, get_conn, utcnow_iso
from ..storage import ensure_session_dirs, write_session_json
from .memory import create_checkpoint
from .mcp import MCPToolRegistry
from .orchestrator import (
    _execute_tool_call,
    _file_tool_prompt_line,
    _file_tool_schema,
    _run_with_tool_loop,
    _save_message,
    _set_hard_rollover_started,
    _set_pre_rollover_started,
    _stream_from_provider,
    _terminal_prompt_line,
    _terminal_tool_schema,
    _window_usage,
    _with_message_prefix_prompt,
)
from .search_budget import (
    SearchBudgetTracker,
    exhausted_payload,
    filter_search_results_for_budget,
    looks_like_empty_search_result,
)
from .prompt import assemble_prompt
from .provider_http import (
    build_payload,
    provider_timeout_seconds,
    resolve_provider_model,
)
from .sessions import create_next_window, get_last_window, get_session, update_session

RUN_ACTIVE_STATUSES = {"queued", "running"}
RUN_TERMINAL_STATUSES = {"completed", "failed", "canceled"}
MAX_PLAN_STEPS = 8
MAX_FINAL_CONTEXT_CHARS = 12000
MAX_STEP_ATTEMPTS = 3


def _json_loads_safe(raw: Any, fallback: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return fallback
    if raw is None:
        return fallback
    return raw


def _row_to_run(row: Any) -> dict[str, Any]:
    progress = _json_loads_safe(row["progress_json"], {})
    if not isinstance(progress, dict):
        progress = {}
    return {
        "id": row["id"],
        "session_id": row["session_id"],
        "window_id": row["window_id"],
        "user_message_id": row["user_message_id"],
        "result_message_id": row["result_message_id"],
        "task_text": row["task_text"],
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "progress_json": progress,
        "error_text": row["error_text"],
    }


def _row_to_run_event(row: Any) -> dict[str, Any]:
    payload = _json_loads_safe(row["payload_json"], {})
    return {
        "id": row["id"],
        "session_id": row["session_id"],
        "run_id": row["run_id"],
        "step_index": row["step_index"],
        "event_type": row["event_type"],
        "title": row["title"],
        "detail": row["detail"],
        "payload_json": payload if isinstance(payload, dict) else {},
        "timestamp": row["timestamp"],
    }


def _row_to_run_artifact(row: Any) -> dict[str, Any]:
    metadata = _json_loads_safe(row["metadata_json"], {})
    return {
        "id": row["id"],
        "session_id": row["session_id"],
        "run_id": row["run_id"],
        "artifact_id": row["artifact_id"],
        "step_index": row["step_index"],
        "stage": row["stage"],
        "title": row["title"],
        "path": row["path"],
        "metadata_json": metadata if isinstance(metadata, dict) else {},
        "created_at": row["created_at"],
    }


def _ensure_run(session_id: str, run_id: str) -> dict[str, Any]:
    row = fetch_one("SELECT * FROM runs WHERE id=? AND session_id=?", (run_id, session_id))
    if row is None:
        raise KeyError("run_not_found")
    return _row_to_run(row)


def list_runs(session_id: str) -> list[dict[str, Any]]:
    _ = get_session(session_id)
    rows = fetch_all(
        "SELECT * FROM runs WHERE session_id=? ORDER BY created_at DESC LIMIT 100",
        (session_id,),
    )
    return [_row_to_run(row) for row in rows]


def get_run(session_id: str, run_id: str) -> dict[str, Any]:
    return _ensure_run(session_id, run_id)


def list_run_events(session_id: str, run_id: str) -> list[dict[str, Any]]:
    _ = _ensure_run(session_id, run_id)
    rows = fetch_all(
        "SELECT * FROM run_events WHERE session_id=? AND run_id=? ORDER BY timestamp ASC",
        (session_id, run_id),
    )
    return [_row_to_run_event(row) for row in rows]


def list_run_artifacts(session_id: str, run_id: str) -> list[dict[str, Any]]:
    _ = _ensure_run(session_id, run_id)
    rows = fetch_all(
        "SELECT * FROM run_artifacts WHERE session_id=? AND run_id=? ORDER BY created_at ASC",
        (session_id, run_id),
    )
    return [_row_to_run_artifact(row) for row in rows]


def _set_run_progress(run_id: str, progress: dict[str, Any]) -> None:
    now = utcnow_iso()
    execute(
        "UPDATE runs SET progress_json=?, updated_at=? WHERE id=?",
        (json.dumps(progress, ensure_ascii=False), now, run_id),
    )


def _set_run_status(
    run_id: str,
    status: str,
    *,
    error_text: str | None = None,
    result_message_id: str | None = None,
    finished: bool = False,
) -> None:
    now = utcnow_iso()
    finished_at = now if finished else None
    execute(
        """
        UPDATE runs
        SET status=?, error_text=?, result_message_id=COALESCE(?, result_message_id),
            finished_at=COALESCE(?, finished_at), updated_at=?
        WHERE id=?
        """,
        (status, error_text, result_message_id, finished_at, now, run_id),
    )


def _record_run_event(
    *,
    session_id: str,
    run_id: str,
    event_type: str,
    title: str,
    detail: str = "",
    payload: dict[str, Any] | None = None,
    step_index: int | None = None,
) -> dict[str, Any]:
    event_id = str(uuid.uuid4())
    timestamp = utcnow_iso()
    execute(
        """
        INSERT INTO run_events (
          id, session_id, run_id, step_index, event_type, title, detail, payload_json, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_id,
            session_id,
            run_id,
            step_index,
            event_type,
            title,
            detail,
            json.dumps(payload or {}, ensure_ascii=False),
            timestamp,
        ),
    )
    return {
        "id": event_id,
        "session_id": session_id,
        "run_id": run_id,
        "step_index": step_index,
        "event_type": event_type,
        "title": title,
        "detail": detail,
        "payload_json": payload or {},
        "timestamp": timestamp,
    }


def _record_run_artifact(
    *,
    session_id: str,
    run_id: str,
    stage: str,
    title: str,
    path: str,
    metadata: dict[str, Any] | None = None,
    artifact_id: str | None = None,
    step_index: int | None = None,
) -> dict[str, Any]:
    item_id = str(uuid.uuid4())
    created_at = utcnow_iso()
    execute(
        """
        INSERT INTO run_artifacts (
          id, session_id, run_id, artifact_id, step_index, stage, title, path, metadata_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            item_id,
            session_id,
            run_id,
            artifact_id,
            step_index,
            stage,
            title,
            path,
            json.dumps(metadata or {}, ensure_ascii=False),
            created_at,
        ),
    )
    return {
        "id": item_id,
        "session_id": session_id,
        "run_id": run_id,
        "artifact_id": artifact_id,
        "step_index": step_index,
        "stage": stage,
        "title": title,
        "path": path,
        "metadata_json": metadata or {},
        "created_at": created_at,
    }


def _run_dir(session_id: str, run_id: str) -> Path:
    path = ensure_session_dirs(session_id) / "runs" / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_run_text_artifact(
    *,
    session_id: str,
    run_id: str,
    stage: str,
    title: str,
    relative_path: str,
    content: str,
    step_index: int | None = None,
) -> dict[str, Any]:
    rel = Path(relative_path)
    if rel.is_absolute() or any(part == ".." for part in rel.parts):
        raise ValueError("run artifact path must be relative")
    target = _run_dir(session_id, run_id) / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    stored_path = f"runs/{run_id}/{rel.as_posix()}"
    return _record_run_artifact(
        session_id=session_id,
        run_id=run_id,
        stage=stage,
        title=title,
        path=stored_path,
        metadata={"bytes": len(content.encode("utf-8"))},
        step_index=step_index,
    )


def _write_run_json_artifact(
    *,
    session_id: str,
    run_id: str,
    stage: str,
    title: str,
    relative_path: str,
    payload: Any,
    step_index: int | None = None,
) -> dict[str, Any]:
    return _write_run_text_artifact(
        session_id=session_id,
        run_id=run_id,
        stage=stage,
        title=title,
        relative_path=relative_path,
        content=json.dumps(payload, ensure_ascii=False, indent=2),
        step_index=step_index,
    )


def _append_run_log(session_id: str, run_id: str, event: dict[str, Any]) -> None:
    target = _run_dir(session_id, run_id) / "run_log.jsonl"
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")


def _is_canceled(run_id: str) -> bool:
    row = fetch_one("SELECT status FROM runs WHERE id=?", (run_id,))
    if row is None:
        return True
    return str(row["status"]) == "canceled"


def _save_run_system_message(*, session_id: str, run_id: str, text: str, window_id: str) -> dict[str, Any]:
    return _save_message(
        session_id=session_id,
        window_id=window_id,
        role="system",
        content_text=f"[Run {run_id}] {text}",
        message_type="internal_event",
        turn_id=run_id,
        source="run",
        content_json={"run_id": run_id},
    )


def _auto_enable_skill_state_for_run(session_id: str) -> None:
    """Opt a session into ``context_mode=skill_state`` the first time
    a background run starts on it, unless the user has already
    configured a different value.

    Background runs are a long-lived executor (4+ steps) that
    replays the chat history at every step in the legacy
    ``"full"`` mode. The same prompt shape repeats each step
    anyway, so SKILL.state — where the model only ever sees
    ``(spec, state, observation)`` — is a strict improvement:
    smaller prompts, less noise, and the per-step engine
    rotation we just wired up actually fires.

    The opt-in is conservative: we only flip the session when it
    hasn't been touched at all (``context_mode`` is the default
    ``"full"``). If the user has explicitly chosen ``"full"``
    we leave the session alone. We never override an explicit
    ``"skill_state"`` choice either — that one is already what
    we want.
    """
    try:
        from ..schemas import SessionUpdate
        current = get_session(session_id)
    except Exception:
        return
    if not current:
        return
    current_mode = current.get("context_mode") or "full"
    if current_mode == "skill_state":
        return  # already in the right mode
    # Only flip when the session is on the default "full" mode.
    # We do not distinguish "user-set full" from "default full" in
    # the persistence layer, so we use a soft opt-in: emit a
    # system message that documents the switch so the user can
    # see what happened and revert if they want to.
    try:
        update_session(
            session_id,
            SessionUpdate(context_mode="skill_state"),
        )
    except Exception:
        return


def create_run(session_id: str, task_text: str) -> dict[str, Any]:
    content = task_text.strip()
    if not content:
        raise ValueError("content is required")

    session_info = get_session(session_id)
    window = get_last_window(session_id)
    if not window:
        raise KeyError("window_not_found")

    run_id = str(uuid.uuid4())
    now = utcnow_iso()

    # Background runs are exactly the workload SKILL.state was
    # designed for: a 4-step pipeline, a long-lived executor, the
    # same prompt shape repeated at every step. The legacy "full"
    # mode replays the entire chat history at every step, which
    # grows the per-step prompt by 1-3k tokens for no benefit and
    # dilutes the model's attention. We opt the session in to
    # SKILL.state automatically the first time a background run
    # starts, unless the user has explicitly set a different mode
    # (full or skill_state) at the session or global level.
    _auto_enable_skill_state_for_run(session_id)

    user_msg = _save_message(
        session_id=session_id,
        window_id=window["id"],
        role="user",
        content_text=content,
        message_type="user",
        turn_id=run_id,
        source="run",
        content_json={"run_id": run_id, "run_mode": "background"},
    )

    execute(
        """
        INSERT INTO runs (
            id, session_id, window_id, user_message_id, task_text, status,
            created_at, updated_at, progress_json
        ) VALUES (?, ?, ?, ?, ?, 'queued', ?, ?, ?)
        """,
        (
            run_id,
            session_id,
            window["id"],
            user_msg["id"],
            content,
            now,
            now,
            json.dumps({"phase": "queued", "current_step": 0, "total_steps": 0}, ensure_ascii=False),
        ),
    )

    _save_run_system_message(
        session_id=session_id,
        run_id=run_id,
        text="Task queued. I will start it as soon as a worker is free.",
        window_id=window["id"],
    )

    _ = session_info  # read to ensure KeyError consistency if session is missing
    return _ensure_run(session_id, run_id)


def cancel_run(session_id: str, run_id: str) -> dict[str, Any]:
    run = _ensure_run(session_id, run_id)
    if run["status"] in RUN_TERMINAL_STATUSES:
        return run

    now = utcnow_iso()
    finished_at = now if run["status"] == "queued" else None
    execute(
        "UPDATE runs SET status='canceled', updated_at=?, finished_at=COALESCE(?, finished_at) WHERE id=? AND session_id=?",
        (now, finished_at, run_id, session_id),
    )

    window_id = run["window_id"] or (get_last_window(session_id) or {}).get("id")
    if window_id:
        _save_run_system_message(
            session_id=session_id,
            run_id=run_id,
            text="Execution canceled by the user.",
            window_id=window_id,
        )

    return _ensure_run(session_id, run_id)


def _claim_next_run() -> dict[str, Any] | None:
    now = utcnow_iso()
    with get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM runs WHERE status='queued' ORDER BY created_at ASC LIMIT 1"
        ).fetchone()
        if row is None:
            return None

        conn.execute(
            """
            UPDATE runs
            SET status='running', started_at=COALESCE(started_at, ?), updated_at=?
            WHERE id=? AND status='queued'
            """,
            (now, now, row["id"]),
        )
        claimed = conn.execute("SELECT * FROM runs WHERE id=?", (row["id"],)).fetchone()
        if claimed is None or claimed["status"] != "running":
            return None
        return _row_to_run(claimed)


def _requeue_inflight_runs() -> None:
    now = utcnow_iso()
    execute(
        "UPDATE runs SET status='queued', updated_at=? WHERE status='running' AND finished_at IS NULL",
        (now,),
    )


async def _complete_text(prompt_messages: list[dict[str, str]]) -> str:
    provider, model = resolve_provider_model()
    if provider is None or model is None:
        return ""
    if not provider.base_url:
        chunks: list[str] = []
        async for delta in _stream_from_provider(prompt_messages):
            chunks.append(delta)
        return "".join(chunks).strip()

    url, headers, payload = build_payload(
        provider=provider,
        model=model,
        messages=prompt_messages,
        stream=False,
    )

    try:
        async with httpx.AsyncClient(timeout=provider_timeout_seconds(provider)) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            obj = resp.json()
    except Exception:
        return ""

    choices = obj.get("choices") or []
    message = (choices[0] if choices else {}).get("message") or {}
    return str(message.get("content") or "").strip()


def _extract_plan_steps(raw_text: str, task_text: str) -> list[dict[str, str]]:
    text = (raw_text or "").strip()
    candidates: list[str] = []

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fence:
        candidates.append(fence.group(1).strip())
    candidates.append(text)

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue

        payload = parsed.get("steps") if isinstance(parsed, dict) else parsed
        if not isinstance(payload, list):
            continue

        steps: list[dict[str, str]] = []
        for item in payload:
            if isinstance(item, str) and item.strip():
                step_text = item.strip()
                steps.append({"title": step_text[:120], "instruction": step_text})
            elif isinstance(item, dict):
                title = str(item.get("title") or "").strip()
                instruction = str(item.get("instruction") or item.get("details") or title).strip()
                if instruction:
                    steps.append({"title": title or instruction[:120], "instruction": instruction})
        if steps:
            return steps[:MAX_PLAN_STEPS]

    bullet_lines = [
        re.sub(r"^\s*(?:[-*]|\d+\.)\s*", "", line).strip()
        for line in text.splitlines()
        if re.match(r"^\s*(?:[-*]|\d+\.)\s+", line)
    ]
    bullet_lines = [line for line in bullet_lines if line]
    if bullet_lines:
        return [{"title": line[:120], "instruction": line} for line in bullet_lines[:MAX_PLAN_STEPS]]

    return [
        {
            "title": "Working on the task",
            "instruction": task_text.strip(),
        }
    ]


def _extract_first_json_object(raw_text: str) -> dict[str, Any]:
    text = (raw_text or "").strip()
    candidates: list[str] = []
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fence:
        candidates.append(fence.group(1).strip())
    candidates.append(text)

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _normalize_workflow_step(item: Any, index: int) -> dict[str, Any] | None:
    if isinstance(item, str):
        instruction = item.strip()
        if not instruction:
            return None
        return {
            "title": instruction[:120],
            "instruction": instruction,
            "worker_type": "executor",
            "acceptance_criteria": [],
            "expected_artifacts": [],
        }

    if not isinstance(item, dict):
        return None

    title = str(item.get("title") or item.get("name") or "").strip()
    instruction = str(item.get("instruction") or item.get("details") or item.get("task") or title).strip()
    if not instruction:
        return None

    worker_type = str(item.get("worker_type") or item.get("role") or "executor").strip().lower()
    worker_type = re.sub(r"[^a-z0-9_-]+", "-", worker_type).strip("-") or "executor"

    def _string_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    return {
        "title": title or f"Step {index}",
        "instruction": instruction,
        "worker_type": worker_type,
        "acceptance_criteria": _string_list(item.get("acceptance_criteria")),
        "expected_artifacts": _string_list(item.get("expected_artifacts")),
    }


def _fallback_workflow_plan(task_text: str) -> dict[str, Any]:
    steps = [
        {
            "title": "Clarify task and constraints",
            "instruction": "Identify task type, required tools, acceptance criteria, and risks before changing anything.",
            "worker_type": "router",
            "acceptance_criteria": ["Task type and risks are explicit."],
            "expected_artifacts": ["plan.json"],
        },
        {
            "title": "Inspect and implement",
            "instruction": task_text.strip(),
            "worker_type": "executor",
            "acceptance_criteria": ["Requested behavior is implemented."],
            "expected_artifacts": [],
        },
        {
            "title": "Validate outcome",
            "instruction": "Run the most relevant available checks and summarize failures or evidence of success.",
            "worker_type": "verifier",
            "acceptance_criteria": ["Validation commands or explicit skipped-check reasons are recorded."],
            "expected_artifacts": ["verifier.json"],
        },
    ]
    return {
        "workflow_type": "general",
        "needs_research": _wants_research(task_text),
        "risk_level": "medium",
        "steps": steps[:MAX_PLAN_STEPS],
    }


def _wants_research(task_text: str) -> bool:
    lowered = task_text.lower()
    hints = [
        # English
        "latest", "research", "github", "docs", "documentation",
        "release", "news", "lookup", "search", "find",
        # Spanish
        "último", "investigación", "documentación", "noticias", "buscar", "encontrar",
        # French
        "dernier", "recherche", "documentation", "actualités", "chercher", "trouver",
        # Portuguese
        "último", "pesquisa", "documentação", "notícias", "pesquisar", "encontrar",
        # German
        "neueste", "recherche", "dokumentation", "nachrichten", "suchen", "finden",
        # Italian
        "ultimo", "ricerca", "documentazione", "notizie", "cercare", "trovare",
        # Polish
        "najnowszy", "badania", "dokumentacja", "wiadomości", "szukaj", "znajdź",
        # Dutch
        "laatste", "onderzoek", "documentatie", "nieuws", "zoeken", "vinden",
        # Turkish
        "son", "araştırma", "belgelendirme", "haberler", "ara", "bul",
        # Vietnamese
        "mới nhất", "nghiên cứu", "tài liệu", "tin tức", "tìm kiếm", "tìm",
        # Japanese
        "最新", "研究", "ドキュメント", "ニュース", "検索", "見つけて",
        # Korean
        "최신", "연구", "문서", "뉴스", "검색", "찾기",
        # Chinese
        "最新", "研究", "文档", "新闻", "搜索", "查找",
        # Hindi
        "नवीनतम", "अनुसंधान", "दस्तावेज़", "समाचार", "खोज", "खोजो",
        # Arabic
        "أحدث", "بحث", "وثائق", "أخبار", "ابحث",
        # Ukrainian
        "знайди", "пошукай", "документ", "реліз",
    ]
    return any(h in lowered for h in hints)


async def _build_workflow_plan(task_text: str) -> dict[str, Any]:
    planner_messages = [
        {
            "role": "system",
            "content": (
                "You are a strict workflow planner for a small-model agent system. "
                "Return strict JSON only with this shape: "
                "{\"workflow_type\":\"coding|research|mixed|general\","
                "\"needs_research\":true,"
                "\"risk_level\":\"low|medium|high\","
                "\"steps\":[{\"title\":\"...\",\"instruction\":\"...\","
                "\"worker_type\":\"router|researcher|github_researcher|executor|extractor|critic|verifier|synthesizer\","
                "\"acceptance_criteria\":[\"...\"],\"expected_artifacts\":[\"...\"]}]}. "
                "Use 3-8 concrete steps. Split research, implementation, criticism, validation, and synthesis."
            ),
        },
        {"role": "user", "content": task_text},
    ]
    raw_plan = await _complete_text(planner_messages)
    parsed = _extract_first_json_object(raw_plan)
    fallback = _fallback_workflow_plan(task_text)

    raw_steps = parsed.get("steps") if isinstance(parsed.get("steps"), list) else []
    steps: list[dict[str, Any]] = []
    for index, item in enumerate(raw_steps, start=1):
        step = _normalize_workflow_step(item, index)
        if step:
            steps.append(step)

    if not steps:
        legacy_steps = _extract_plan_steps(raw_plan, task_text)
        for index, item in enumerate(legacy_steps, start=1):
            step = _normalize_workflow_step(item, index)
            if step:
                steps.append(step)

    if not steps:
        steps = fallback["steps"]

    workflow_type = str(parsed.get("workflow_type") or fallback["workflow_type"]).strip() or "general"
    risk_level = str(parsed.get("risk_level") or fallback["risk_level"]).strip().lower()
    if risk_level not in {"low", "medium", "high"}:
        risk_level = fallback["risk_level"]

    return {
        "workflow_type": workflow_type,
        "needs_research": bool(parsed.get("needs_research", fallback["needs_research"])),
        "risk_level": risk_level,
        "steps": steps[:MAX_PLAN_STEPS],
    }


async def _build_plan(task_text: str) -> list[dict[str, str]]:
    planner_messages = [
        {
            "role": "system",
            "content": (
                "You are a planner for long-running coding tasks. "
                "Return strict JSON only: {\"steps\":[{\"title\":\"...\",\"instruction\":\"...\"}]}. "
                "Use 3-8 concrete steps in imperative form."
            ),
        },
        {
            "role": "user",
            "content": task_text,
        },
    ]
    raw_plan = await _complete_text(planner_messages)
    return _extract_plan_steps(raw_plan, task_text)


def _serialize_step_plan(steps: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for index, step in enumerate(steps, start=1):
        worker_type = str(step.get("worker_type") or "executor")
        lines.append(f"{index}. [{worker_type}] {step['title']}")
    return "\n".join(lines)


def _build_final_context(task_text: str, step_outputs: list[dict[str, str]]) -> str:
    chunks = [f"Task:\n{task_text.strip()}\n", "Step outputs:\n"]
    for idx, item in enumerate(step_outputs, start=1):
        chunks.append(f"{idx}. {item['title']}\n{item['output']}\n")
    text = "\n".join(chunks)
    if len(text) <= MAX_FINAL_CONTEXT_CHARS:
        return text
    return text[:MAX_FINAL_CONTEXT_CHARS].rstrip() + "\n\n[truncated]"


def _extract_step_facts(step_output: str, step_artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    facts: list[dict[str, Any]] = []
    files_touched: list[str] = []

    for artifact in step_artifacts:
        path = str(artifact.get("workspace_path") or artifact.get("file_name") or "").strip()
        if path:
            files_touched.append(path)
            facts.append({"claim": f"Created or updated {path}", "source": "artifact", "confidence": "high"})

    for line in (step_output or "").splitlines():
        normalized = line.strip(" -\t")
        if not normalized:
            continue
        lowered = normalized.lower()
        title_match = re.match(r"^\*\*\d+\.\s*(.+?)\*\*", normalized)
        if title_match:
            facts.append({"claim": title_match.group(1).strip()[:500], "source": "web_search_result", "confidence": "medium"})
            continue
        if lowered.startswith("url:"):
            url = normalized.split(":", 1)[1].strip()
            if url:
                facts.append({"claim": f"Source URL: {url[:450]}", "source": "web_search_result", "confidence": "medium"})
            continue
        if any(h in lowered for h in (
    # English
    "created", "updated", "implemented", "fixed", "tested",
    # Spanish
    "creado", "actualizado", "implementado", "corregido", "probado",
    # French
    "créé", "mis à jour", "implémenté", "corrigé", "testé",
    # Portuguese
    "criado", "atualizado", "implementado", "corrigido", "testado",
    # German
    "erstellt", "aktualisiert", "implementiert", "behoben", "getestet",
    # Italian
    "creato", "aggiornato", "implementato", "corretto", "testato",
    # Polish
    "utworzony", "zaktualizowany", "zaimplementowany", "naprawiony", "przetestowany",
    # Dutch
    "gemaakt", "bijgewerkt", "geïmplementeerd", "opgelost", "getest",
    # Turkish
    "oluşturuldu", "güncellendi", "uygulandı", "düzeltildi", "test edildi",
    # Vietnamese
    "đã tạo", "đã cập nhật", "đã triển khai", "đã sửa", "đã kiểm tra",
    # Japanese
    "作成", "更新", "実装", "修正", "テスト",
    # Korean
    "생성", "업데이트", "구현", "수정", "테스트",
    # Chinese
    "创建", "更新", "实现", "修复", "测试",
    # Hindi
    "बनाया", "अद्यतन", "लागू", "ठीक", "परीक्षण",
    # Arabic
    "تم إنشاؤه", "تم تحديثه", "تم تنفيذه", "تم إصلاحه", "تم اختباره",
    # Ukrainian
    "змін", "створ", "онов",
)):
            facts.append({"claim": normalized[:500], "source": "step_output", "confidence": "medium"})

    return {
        "facts": facts[:20],
        "files_touched": sorted(set(files_touched)),
        "open_questions": [],
    }


def _default_critique(step: dict[str, Any], output: str, extraction: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    if not output.strip():
        issues.append("Step produced no output.")
    if not extraction.get("facts"):
        issues.append("No concrete facts or changed files were extracted from the step output.")

    criteria = step.get("acceptance_criteria")
    criteria_list = criteria if isinstance(criteria, list) else []
    if criteria_list and len(output.strip()) < 80:
        issues.append("Step output is too short to demonstrate acceptance criteria.")

    return {
        "status": "needs_attention" if issues else "pass",
        "issues": issues,
        "confidence": "medium" if issues else "high",
        "retry_recommended": False,
    }


async def _critique_step_output(step: dict[str, Any], output: str, extraction: dict[str, Any]) -> dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a skeptical run critic. Return strict JSON only: "
                "{\"status\":\"pass|needs_attention|fail\",\"issues\":[\"...\"],"
                "\"confidence\":\"low|medium|high\",\"retry_recommended\":false}. "
                "Do not add new facts."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "step": step,
                    "output": output[:6000],
                    "extraction": extraction,
                },
                ensure_ascii=False,
            ),
        },
    ]
    try:
        raw = await _complete_text(messages)
        parsed = _extract_first_json_object(raw)
    except Exception:
        parsed = {}

    fallback = _default_critique(step, output, extraction)
    status = str(parsed.get("status") or fallback["status"]).strip()
    if status not in {"pass", "needs_attention", "fail"}:
        status = fallback["status"]
    issues = parsed.get("issues")
    if not isinstance(issues, list):
        issues = fallback["issues"]
    confidence = str(parsed.get("confidence") or fallback["confidence"]).strip()
    if confidence not in {"low", "medium", "high"}:
        confidence = fallback["confidence"]

    return {
        "status": status,
        "issues": [str(item).strip() for item in issues if str(item).strip()][:20],
        "confidence": confidence,
        "retry_recommended": bool(parsed.get("retry_recommended", fallback["retry_recommended"])),
    }


def _read_package_scripts(package_json_path: Path) -> dict[str, str]:
    try:
        raw = json.loads(package_json_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    scripts = raw.get("scripts")
    if not isinstance(scripts, dict):
        return {}
    return {str(key): str(value) for key, value in scripts.items()}


def _build_verification_commands(session_info: dict[str, Any]) -> list[dict[str, Any]]:
    workspace_path = str(session_info.get("workspace_path") or "").strip()
    if not workspace_path:
        return []

    root = Path(workspace_path).expanduser()
    commands: list[dict[str, Any]] = []

    api_venv_python = root / "apps" / "api" / ".venv" / "bin" / "python"
    python_cmd = "apps/api/.venv/bin/python" if api_venv_python.exists() else "python"

    if (root / "apps" / "api" / "pyproject.toml").exists():
        commands.append({"command": f"{python_cmd} -m pytest apps/api/tests", "timeout_sec": 120, "label": "api tests"})
    elif (root / "pyproject.toml").exists() or (root / "pytest.ini").exists() or (root / "tests").exists():
        commands.append({"command": "python -m pytest", "timeout_sec": 120, "label": "python tests"})

    root_package = root / "package.json"
    scripts = _read_package_scripts(root_package) if root_package.exists() else {}
    if "build:web" in scripts:
        commands.append({"command": "npm run build:web", "timeout_sec": 120, "label": "web build"})
    elif "build" in scripts:
        commands.append({"command": "npm run build", "timeout_sec": 120, "label": "npm build"})
    elif "test" in scripts and scripts["test"] and "no test specified" not in scripts["test"].lower():
        commands.append({"command": "npm test", "timeout_sec": 120, "label": "npm test"})

    api_package = root / "apps" / "web" / "package.json"
    web_scripts = _read_package_scripts(api_package) if api_package.exists() else {}
    if "build" in web_scripts and not any(item["command"] == "npm run build:web" for item in commands):
        commands.append({"command": "npm --workspace apps/web run build", "timeout_sec": 120, "label": "web build"})

    return commands[:3]


def _step_worker_type(step: dict[str, Any]) -> str:
    worker_type = str(step.get("worker_type") or "executor").strip().lower()
    return re.sub(r"[^a-z0-9_-]+", "-", worker_type).strip("-") or "executor"


def _step_needs_web_tools(step: dict[str, Any]) -> bool:
    worker_type = _step_worker_type(step)
    if worker_type in {"researcher", "github_researcher", "web-researcher", "extractor"}:
        return True
    if worker_type in {"router", "critic", "synthesizer", "verifier"}:
        return False
    text = f"{step.get('title') or ''}\n{step.get('instruction') or ''}"
    return _wants_research(text)


def _step_needs_local_tools(step: dict[str, Any]) -> bool:
    worker_type = _step_worker_type(step)
    if worker_type in {"executor", "verifier", "code_runner", "ffmpeg_worker"}:
        return True
    text = f"{step.get('title') or ''}\n{step.get('instruction') or ''}".lower()
    return any(h in text for h in (
    # English
    "run ", "test", "build", "file", "code", "execute", "verify",
    # Spanish
    "ejecutar", "prueba", "construir", "archivo", "código", "verificar",
    # French
    "exécuter", "test", "construire", "fichier", "code", "vérifier",
    # Portuguese
    "executar", "teste", "construir", "arquivo", "código", "verificar",
    # German
    "ausführen", "test", "erstellen", "datei", "code", "prüfen",
    # Italian
    "esegui", "test", "costruisci", "file", "codice", "verifica",
    # Polish
    "uruchom", "test", "buduj", "plik", "kod", "sprawdź",
    # Dutch
    "uitvoeren", "test", "bouwen", "bestand", "code", "controleren",
    # Turkish
    "çalıştır", "test", "inşa", "dosya", "kod", "kontrol",
    # Vietnamese
    "chạy", "kiểm tra", "xây dựng", "tệp", "mã", "xác minh",
    # Japanese
    "実行", "テスト", "ビルド", "ファイル", "コード", "確認",
    # Korean
    "실행", "테스트", "빌드", "파일", "코드", "확인",
    # Chinese
    "运行", "测试", "构建", "文件", "代码", "检查",
    # Hindi
    "चलाओ", "परीक्षण", "बनाओ", "फाइल", "कोड", "जांचो",
    # Arabic
    "تشغيل", "اختبار", "بناء", "ملف", "كود", "تحقق",
    # Ukrainian
    "код", "файл", "запусти", "перевір",
))


def _build_research_search_queries(task_text: str) -> list[str]:
    base = task_text.strip()
    if not base:
        return []

    queries = [
        base,
        f"{base} rating reviews",
        f"site:top20.ua {base}",
        f"site:ratelist.top {base}",
        f"site:tomato.ua {base}",
        f"site:google.com/maps {base}",
    ]
    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = re.sub(r"\s+", " ", query).strip()
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            deduped.append(normalized)
    return deduped


def _build_step_search_query(task_text: str, step: dict[str, Any], *, attempt: int = 1) -> str:
    instruction = str(step.get("instruction") or "").strip()
    role = _step_worker_type(step)
    if role in {"researcher", "github_researcher", "web-researcher", "extractor"}:
        queries = _build_research_search_queries(task_text)
        if not queries:
            return task_text.strip()
        return queries[min(max(attempt, 1) - 1, len(queries) - 1)]
    if _wants_research(instruction):
        return f"{task_text.strip()}\n\nSpecific focus: {instruction}"
    return task_text.strip()


def _summarize_previous_step_outputs(step_outputs: list[dict[str, Any]], max_chars: int = 5000) -> str:
    if not step_outputs:
        return "(none)"
    chunks: list[str] = []
    for index, item in enumerate(step_outputs, start=1):
        title = str(item.get("title") or f"Step {index}")
        output = str(item.get("output") or "").strip()
        critique = item.get("critique") if isinstance(item.get("critique"), dict) else {}
        chunks.append(
            f"{index}. {title}\n"
            f"Output:\n{output[:1200]}\n"
            f"Critique: {json.dumps(critique, ensure_ascii=False)}"
        )
    text = "\n\n".join(chunks)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n[previous outputs truncated]"


def _build_deterministic_final_answer(
    task_text: str,
    step_outputs: list[dict[str, Any]],
    verification: dict[str, Any] | None = None,
) -> str:
    facts: list[str] = []
    provider_notes: list[str] = []
    for item in step_outputs:
        output = str(item.get("output") or "")
        if "provider rejected" in output.lower():
            provider_notes.append("Model provider rejected tool-schema or follow-up synthesis, so raw tool evidence was preserved.")
        extraction = item.get("extraction") if isinstance(item.get("extraction"), dict) else {}
        for fact in extraction.get("facts") or []:
            if not isinstance(fact, dict):
                continue
            claim = str(fact.get("claim") or "").strip()
            if claim and claim not in facts:
                facts.append(claim)

    lines = [
        "# Final Delivery Report",
        "",
        f"Task: {task_text}",
        "",
        "## Result",
    ]
    if facts:
        for fact in facts[:10]:
            lines.append(f"- {fact}")
    else:
        lines.append("- No structured facts were extracted, but step artifacts were saved for inspection.")

    if provider_notes:
        lines.extend(["", "## Notes"])
        for note in sorted(set(provider_notes)):
            lines.append(f"- {note}")

    verification = verification or {}
    status = str(verification.get("status") or "unknown")
    lines.extend(["", "## Verification", f"- Status: {status}"])
    for result in verification.get("results") or []:
        if not isinstance(result, dict):
            continue
        label = str(result.get("label") or result.get("command") or "check")
        ok = "passed" if result.get("ok") else "failed"
        lines.append(f"- {label}: {ok}")

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Search-budget aware tool executor
# ---------------------------------------------------------------------------

# Tool names (or suffixes) that the orchestrator will route through the
# SearchBudgetTracker. The match is intentionally lenient: any tool
# whose name contains "full_web_search" goes through the budget.
_SEARCH_TOOL_MARKERS = (
    "full_web_search",
    "deep_web_search",
    "search_engine_query",
)


def _is_search_tool(fn_name: str) -> bool:
    name = (fn_name or "").lower()
    return any(marker in name for marker in _SEARCH_TOOL_MARKERS)


def _engine_from_args(args: dict[str, Any]) -> str:
    for key in ("prefer_engine", "engine", "search_engine"):
        v = args.get(key)
        if isinstance(v, str) and v:
            return v
    return ""


def _result_text_for_budget(result: Any) -> str:
    """Best-effort extraction of the result text we want to feed to
    the budget tracker. Handles both ``{"ok": True, "result": [...]}``
    (MCP tool calls) and plain ``{"text": "..."}`` dicts."""
    if not isinstance(result, dict):
        return str(result or "")
    for key in ("text", "content", "summary", "answer"):
        v = result.get(key)
        if isinstance(v, str) and v:
            return v
    # The MCP tool result shape is usually a list of content blocks.
    blocks = result.get("result") or result.get("content") or []
    if isinstance(blocks, list):
        chunks: list[str] = []
        for block in blocks:
            if isinstance(block, dict):
                txt = block.get("text") or block.get("content")
                if isinstance(txt, str):
                    chunks.append(txt)
        if chunks:
            return "\n".join(chunks)
    return str(result)


def make_search_aware_executor(
    *,
    run_id: str,
    tracker: SearchBudgetTracker,
    session_id: str,
    session_info: dict[str, Any],
    mcp_registry: Any | None,
):
    """Build a ``lambda(fn_name, args)`` that wraps
    ``_execute_tool_call`` and routes web-search calls through the
    SearchBudgetTracker.

    The wrapper:

    * Asks the tracker which engine to prefer (or whether the
      budget is already exhausted and we should short-circuit).
    * If the budget is exhausted, returns an ``ok=False`` payload
      with a clear "stop searching, summarise" message so the model
      can finalise the report instead of looping.
    * Otherwise calls ``_execute_tool_call`` and feeds the result
      back to the tracker so the next call can pick a different
      engine if needed.
    """

    async def execute(fn_name: str, args: dict[str, Any]) -> dict[str, Any]:
        if not _is_search_tool(fn_name):
            return await _execute_tool_call(
                fn_name=fn_name,
                args=args,
                session_id=session_id,
                session_info=session_info,
                mcp_registry=mcp_registry,
            )

        planned = tracker.plan_next_call(run_id, args)
        if planned.get("__search_budget_exhausted"):
            query = (args or {}).get("query") or (args or {}).get("q") or ""
            return exhausted_payload(str(query))

        # Strip the internal marker before forwarding.
        clean_args = {k: v for k, v in planned.items() if not k.startswith("__search_budget")}

        result = await _execute_tool_call(
            fn_name=fn_name,
            args=clean_args,
            session_id=session_id,
            session_info=session_info,
            mcp_registry=mcp_registry,
        )

        engine = _engine_from_args(clean_args) or "<default>"
        result_text = _result_text_for_budget(result)
        # Drop search-engine-UI hits (Yahoo Suggestions, Bing Privacy
        # Dashboard, …) before we count the result as "non-empty".
        # Otherwise the budget tracker reports a healthy 5-result run
        # even though every URL was a yahoo.uservoice.com page.
        if isinstance(result, dict):
            for key in ("results", "content", "items"):
                payload = result.get(key)
                if isinstance(payload, list):
                    filtered = filter_search_results_for_budget(payload)
                    if len(filtered) != len(payload):
                        result[key] = filtered
                        if key == "results":
                            result.setdefault(
                                "__filtered_out_count",
                                len(payload) - len(filtered),
                            )
                            result_text = _result_text_for_budget(result)
        summary = tracker.record(run_id, engine, result_text)

        # Surface the budget state to the next iteration of the tool
        # loop by augmenting the result. The orchestrator strips this
        # before persisting the message, but the model still sees the
        # hint in the next user/tool message.
        if isinstance(result, dict):
            result.setdefault("__search_budget", summary)
        return result

    return execute


async def _run_verifier(
    *,
    session_id: str,
    run_id: str,
    session_info: dict[str, Any],
    mcp_registry: MCPToolRegistry | None,
) -> dict[str, Any]:
    commands = _build_verification_commands(session_info)
    if not commands:
        return {
            "status": "skipped",
            "reason": "No recognized test/build commands in the session workspace.",
            "commands": [],
            "results": [],
        }

    results: list[dict[str, Any]] = []
    for item in commands:
        result = await _execute_tool_call(
            fn_name="run_terminal_command",
            args={"command": item["command"], "timeout_sec": item["timeout_sec"]},
            session_id=session_id,
            session_info=session_info,
            mcp_registry=mcp_registry,
        )
        results.append({"label": item["label"], **result})
        _record_run_event(
            session_id=session_id,
            run_id=run_id,
            event_type="verification_command",
            title=item["label"],
            detail=f"{item['command']} -> exit={result.get('exit_code')}",
            payload=result,
        )

    failed = [item for item in results if not item.get("ok")]
    return {
        "status": "failed" if failed else "passed",
        "commands": commands,
        "results": results,
    }


def _maybe_rollover_for_background_run(*, session_id: str, window_id: str, run_id: str) -> str:
    cfg = load_app_config()
    row = fetch_one(
        "SELECT id, pre_rollover_started_at, hard_rollover_started_at FROM windows WHERE id=?",
        (window_id,),
    )
    if row is None:
        return window_id

    _token_limit, _used_tokens, used_percent = _window_usage(session_id, window_id)
    pre_th = cfg.rollover_config.pre_rollover_threshold
    hard_th = cfg.rollover_config.hard_rollover_threshold

    active_window_id = window_id

    if used_percent >= pre_th and row["pre_rollover_started_at"] is None:
        _set_pre_rollover_started(window_id)
        cp = create_checkpoint(session_id, source_window_id=window_id, reason="pre_rollover")
        _save_run_system_message(
            session_id=session_id,
            run_id=run_id,
            text=f"Pre-rollover prepared. Checkpoint: {cp['id']}",
            window_id=window_id,
        )

    _token_limit, _used_tokens, used_percent = _window_usage(session_id, window_id)
    if used_percent >= hard_th:
        _set_hard_rollover_started(window_id)
        cp = create_checkpoint(session_id, source_window_id=window_id, reason="hard_rollover")
        new_window = create_next_window(session_id, closing_reason="token_limit", checkpoint_id=cp["id"])
        active_window_id = new_window["id"]
        _save_run_system_message(
            session_id=session_id,
            run_id=run_id,
            text=f"Hard rollover completed. New window: {active_window_id}. Checkpoint: {cp['id']}",
            window_id=active_window_id,
        )

    # Keep file metadata in sync after possible window switch.
    write_session_json(session_id, get_session(session_id))
    return active_window_id


async def _build_final_answer(
    task_text: str,
    step_outputs: list[dict[str, Any]],
    verification: dict[str, Any] | None = None,
) -> str:
    if not step_outputs:
        return "Task finished, but no steps produced any results."

    final_messages = [
        {
            "role": "system",
            "content": (
                "Create a final delivery report for the user. "
                "Be concrete: what was changed, what was tested, what remains. "
                "Use concise markdown with sections."
            ),
        },
        {
            "role": "user",
            "content": _build_final_context(task_text, step_outputs)
            + "\n\nVerification:\n"
            + json.dumps(verification or {}, ensure_ascii=False, indent=2),
        },
    ]
    text = await _complete_text(final_messages)
    if len((text or "").strip()) < 80:
        return _build_deterministic_final_answer(task_text, step_outputs, verification)
    return text


async def _execute_run(run: dict[str, Any]) -> None:
    run_id = run["id"]
    session_id = run["session_id"]
    task_text = run["task_text"]

    session_info = get_session(session_id)
    cfg = load_app_config()
    window = get_last_window(session_id)
    if not window:
        raise KeyError("window_not_found")
    window_id = window["id"]

    # Per-run search-retry budget. The local small model sometimes
    # calls full_web_search repeatedly with similar queries, gets 0
    # results back (engine=None, rate limit, or just an unanswerable
    # question), and keeps burning turns. The tracker below caps the
    # number of consecutive empty results and rotates the engine
    # automatically so the run has a chance to recover on a different
    # backend.
    search_budget = SearchBudgetTracker()

    # Refresh the working-set snapshot so the next prompt the model
    # sees includes the run's task and the latest run-step output.
    # Without this the working_set stays empty for the entire run
    # and any sub-prompt that consults it sees stale or missing
    # fields.
    try:
        from .memory import update_working_set
        update_working_set(session_id)
    except Exception:
        pass

    _save_run_system_message(
        session_id=session_id,
        run_id=run_id,
        text="Starting the task. Building the execution plan.",
        window_id=window_id,
    )
    window_id = _maybe_rollover_for_background_run(
        session_id=session_id,
        window_id=window_id,
        run_id=run_id,
    )
    _set_run_progress(run_id, {"phase": "planning", "current_step": 0, "total_steps": 0})
    _record_run_event(
        session_id=session_id,
        run_id=run_id,
        event_type="planning_started",
        title="Planning",
        detail="Building structured workflow plan.",
    )
    _write_run_text_artifact(
        session_id=session_id,
        run_id=run_id,
        stage="input",
        title="User task",
        relative_path="input.md",
        content=task_text,
    )

    workflow_plan = await _build_workflow_plan(task_text)
    steps = workflow_plan["steps"]
    if _is_canceled(run_id):
        _set_run_status(run_id, "canceled", finished=True)
        return
    _write_run_json_artifact(
        session_id=session_id,
        run_id=run_id,
        stage="plan",
        title="Workflow plan",
        relative_path="plan.json",
        payload=workflow_plan,
    )
    event = _record_run_event(
        session_id=session_id,
        run_id=run_id,
        event_type="plan_ready",
        title="Plan ready",
        detail=_serialize_step_plan(steps),
        payload=workflow_plan,
    )
    _append_run_log(session_id, run_id, event)

    _save_run_system_message(
        session_id=session_id,
        run_id=run_id,
        text=(
            f"Plan ready. Workflow: {workflow_plan['workflow_type']}, "
            f"risk: {workflow_plan['risk_level']}.\n" + _serialize_step_plan(steps)
        ),
        window_id=window_id,
    )
    window_id = _maybe_rollover_for_background_run(
        session_id=session_id,
        window_id=window_id,
        run_id=run_id,
    )

    mcp_registry: MCPToolRegistry | None = None
    mcp_tools_schema: list[dict[str, Any]] = []
    mcp_prompt_tool_lines: list[str] = []

    mcp_servers = effective_mcp_servers(cfg)
    if cfg.mcp_config.enabled and mcp_servers:
        mcp_registry = await MCPToolRegistry.from_server_configs(mcp_servers)
        mcp_tools_schema = mcp_registry.tool_schemas()
        mcp_prompt_tool_lines = mcp_registry.prompt_tool_lines()

    step_outputs: list[dict[str, Any]] = []
    all_extractions: list[dict[str, Any]] = []
    critiques: list[dict[str, Any]] = []

    try:
        total_steps = len(steps)
        for index, step in enumerate(steps, start=1):
            if _is_canceled(run_id):
                _save_run_system_message(
                    session_id=session_id,
                    run_id=run_id,
                    text="Cancellation received. Stopping execution.",
                    window_id=window_id,
                )
                _set_run_status(run_id, "canceled", finished=True)
                return

            title = step["title"].strip() or f"Step {index}"
            instruction = step["instruction"].strip() or title
            worker_type = _step_worker_type(step)

            _set_run_progress(
                run_id,
                {
                    "phase": title,
                    "workflow_type": workflow_plan["workflow_type"],
                    "risk_level": workflow_plan["risk_level"],
                    "worker_type": worker_type,
                    "current_step": index,
                    "total_steps": total_steps,
                },
            )
            event = _record_run_event(
                session_id=session_id,
                run_id=run_id,
                event_type="step_started",
                title=title,
                detail=instruction,
                payload={"step": step},
                step_index=index,
            )
            _append_run_log(session_id, run_id, event)
            _save_run_system_message(
                session_id=session_id,
                run_id=run_id,
                text=f"Step {index}/{total_steps} [{worker_type}]: {title}",
                window_id=window_id,
            )
            window_id = _maybe_rollover_for_background_run(
                session_id=session_id,
                window_id=window_id,
                run_id=run_id,
            )

            step_tool_lines: list[str] = []
            step_tools_schema: list[dict[str, Any]] = []
            local_tools_enabled = _step_needs_local_tools(step)
            web_tools_enabled = _step_needs_web_tools(step)
            if local_tools_enabled:
                step_tool_lines.extend([_terminal_prompt_line(), _file_tool_prompt_line()])
                step_tools_schema.extend([_terminal_tool_schema(), _file_tool_schema()])
            if web_tools_enabled:
                step_tool_lines.extend(mcp_prompt_tool_lines)
                step_tools_schema.extend(mcp_tools_schema)

            max_attempts = MAX_STEP_ATTEMPTS if web_tools_enabled else 1
            attempt = 1
            previous_attempt_note = ""
            while True:
                search_query = _build_step_search_query(task_text, step, attempt=attempt)
                retry_context = ""
                if attempt > 1:
                    retry_context = (
                        f"\nRetry attempt {attempt}/{max_attempts}.\n"
                        f"Previous attempt issue:\n{previous_attempt_note}\n\n"
                        f"Use this search query first: {search_query}\n"
                        "Actually call the web-search tool again. Do not only say that another search is needed.\n"
                        "Ignore prior irrelevant results unless they are independently confirmed by the new search.\n\n"
                    )

                step_prompt = (
                    "You are executing one role in a structured workflow agent.\n"
                    f"Global task:\n{task_text}\n\n"
                    f"Workflow type: {workflow_plan['workflow_type']}\n"
                    f"Risk level: {workflow_plan['risk_level']}\n"
                    f"Worker role: {worker_type}\n"
                    f"Current step {index}/{total_steps}: {title}\n"
                    f"Instruction:\n{instruction}\n\n"
                    f"Acceptance criteria:\n{json.dumps(step.get('acceptance_criteria') or [], ensure_ascii=False)}\n"
                    f"Expected artifacts:\n{json.dumps(step.get('expected_artifacts') or [], ensure_ascii=False)}\n\n"
                    f"Previous step outputs:\n{_summarize_previous_step_outputs(step_outputs)}\n\n"
                    f"{retry_context}"
                    "When using web search, search for the global task or a concise factual query, "
                    "not for this workflow instruction text.\n"
                    "If relevant, run terminal commands, edit files, and create artifacts. "
                    "Return a concise completion note with facts, files touched, tests run, and unresolved risks. "
                    "Do not claim success unless there is evidence."
                )

                current_window = get_last_window(session_id)
                if current_window:
                    window_id = current_window["id"]

                prompt_messages = assemble_prompt(
                    session_id,
                    _with_message_prefix_prompt(step_prompt, str(session_info.get("message_prefix_prompt") or "")),
                    cfg,
                    None,
                    thinking_mode=str(session_info.get("thinking_mode") or "medium"),
                    terminal_tool_enabled=local_tools_enabled,
                    tool_instruction_lines=step_tool_lines or None,
                )

                step_artifacts: list[dict[str, Any]] = []
                if step_tools_schema:
                    step_output = await _run_with_tool_loop(
                        prompt_messages=prompt_messages,
                        tools_schema=step_tools_schema,
                        execute_tool_call=make_search_aware_executor(
                            run_id=run_id,
                            tracker=search_budget,
                            session_id=session_id,
                            session_info=session_info,
                            mcp_registry=mcp_registry,
                        ),
                        session_id=session_id,
                        window_id=window_id,
                        turn_id=f"{run_id}:step:{index}:attempt:{attempt}",
                        raw_user_query=search_query,
                        artifact_sink=step_artifacts,
                    )
                else:
                    step_output = await _complete_text(prompt_messages)

                extraction = _extract_step_facts(step_output, step_artifacts)
                extraction_payload = {"step_index": index, "attempt": attempt, "title": title, **extraction}
                artifact_suffix = "" if attempt == 1 else f"-attempt-{attempt:02d}"
                _write_run_text_artifact(
                    session_id=session_id,
                    run_id=run_id,
                    stage="step_output",
                    title=f"Step {index} output attempt {attempt}",
                    relative_path=f"steps/{index:02d}{artifact_suffix}-output.md",
                    content=step_output,
                    step_index=index,
                )
                _write_run_json_artifact(
                    session_id=session_id,
                    run_id=run_id,
                    stage="extractor",
                    title=f"Step {index} extracted facts attempt {attempt}",
                    relative_path=f"steps/{index:02d}{artifact_suffix}-facts.json",
                    payload=extraction_payload,
                    step_index=index,
                )
                for artifact in step_artifacts:
                    artifact_id = str(artifact.get("id") or "").strip() or None
                    path = str(
                        artifact.get("workspace_path") or artifact.get("file_name") or artifact.get("artifact_path") or ""
                    )
                    if path:
                        _record_run_artifact(
                            session_id=session_id,
                            run_id=run_id,
                            artifact_id=artifact_id,
                            step_index=index,
                            stage="worker_artifact",
                            title=str(artifact.get("file_name") or path),
                            path=path,
                            metadata={**artifact, "attempt": attempt},
                        )

                critique = await _critique_step_output(step, step_output, extraction)
                critique_payload = {"step_index": index, "attempt": attempt, "title": title, **critique}
                _write_run_json_artifact(
                    session_id=session_id,
                    run_id=run_id,
                    stage="critic",
                    title=f"Step {index} critique attempt {attempt}",
                    relative_path=f"steps/{index:02d}{artifact_suffix}-critique.json",
                    payload=critique_payload,
                    step_index=index,
                )
                event = _record_run_event(
                    session_id=session_id,
                    run_id=run_id,
                    event_type="step_completed",
                    title=title,
                    detail=f"Attempt {attempt}/{max_attempts}; critic status: {critique['status']}",
                    payload={
                        "attempt": attempt,
                        "extraction": extraction,
                        "critique": critique,
                        "artifacts": step_artifacts,
                        "search_query": search_query,
                    },
                    step_index=index,
                )
                _append_run_log(session_id, run_id, event)

                should_retry = (
                    web_tools_enabled
                    and attempt < max_attempts
                    and (critique.get("retry_recommended") or critique.get("status") == "fail")
                )
                if not should_retry:
                    break

                previous_attempt_note = "; ".join(critique.get("issues") or []) or "Step did not meet acceptance criteria."
                attempt += 1
                event = _record_run_event(
                    session_id=session_id,
                    run_id=run_id,
                    event_type="step_retrying",
                    title=title,
                    detail=f"Retrying with query: {_build_step_search_query(task_text, step, attempt=attempt)}",
                    payload={"attempt": attempt, "previous_critique": critique},
                    step_index=index,
                )
                _append_run_log(session_id, run_id, event)
                _save_run_system_message(
                    session_id=session_id,
                    run_id=run_id,
                    text=f"Step {index}: retrying attempt {attempt}/{max_attempts} with a different search query.",
                    window_id=window_id,
                )

            all_extractions.append(extraction_payload)
            critiques.append(critique_payload)

            step_outputs.append(
                {
                    "title": title,
                    "output": step_output,
                    "worker_type": worker_type,
                    "extraction": extraction,
                    "critique": critique,
                }
            )
            _save_message(
                session_id=session_id,
                window_id=window_id,
                role="assistant",
                content_text=step_output,
                message_type="assistant",
                turn_id=f"{run_id}:step:{index}:result",
                source="run",
                content_json={
                    "run_id": run_id,
                    "step_index": index,
                    "worker_type": worker_type,
                    "extraction": extraction,
                    "critique": critique,
                    "artifacts": step_artifacts,
                },
            )
            window_id = _maybe_rollover_for_background_run(
                session_id=session_id,
                window_id=window_id,
                run_id=run_id,
            )

        _set_run_progress(
            run_id,
            {
                "phase": "verifying",
                "workflow_type": workflow_plan["workflow_type"],
                "risk_level": workflow_plan["risk_level"],
                "current_step": len(steps),
                "total_steps": len(steps),
            },
        )
        _save_run_system_message(
            session_id=session_id,
            run_id=run_id,
            text="Running the final verification and compiling the summary.",
            window_id=window_id,
        )
        window_id = _maybe_rollover_for_background_run(
            session_id=session_id,
            window_id=window_id,
            run_id=run_id,
        )

        facts_payload = {
            "facts": [fact for item in all_extractions for fact in item.get("facts", [])],
            "files_touched": sorted(
                {
                    path
                    for item in all_extractions
                    for path in item.get("files_touched", [])
                    if isinstance(path, str) and path
                }
            ),
            "critiques": critiques,
        }
        _write_run_json_artifact(
            session_id=session_id,
            run_id=run_id,
            stage="extractor",
            title="Aggregated facts",
            relative_path="facts.json",
            payload=facts_payload,
        )

        # SKILL.state carry-over: persist the bullet-facts the
        # researcher / extractor / synthesizer roles discovered into
        # ``durable_facts.json`` and pin the corresponding assistant
        # messages as retrieval anchors.  Without this the next chat
        # turn (or a fresh ``Run``) has to re-search the same queries
        # from scratch, which is what produced the "Будь ласка,
        # надайте список сервісів, про які ви питаєте" loop in the
        # last session.
        try:
            from .memory import record_durable_facts_from_run
            carryover = record_durable_facts_from_run(
                session_id=session_id,
                run_id=run_id,
                steps=step_outputs,
            )
        except Exception as exc:
            carryover = {"added_facts": 0, "added_anchors": 0, "error": str(exc)}
        _write_run_json_artifact(
            session_id=session_id,
            run_id=run_id,
            stage="extractor",
            title="Memory carry-over",
            relative_path="carryover.json",
            payload=carryover,
        )

        verification = await _run_verifier(
            session_id=session_id,
            run_id=run_id,
            session_info=session_info,
            mcp_registry=mcp_registry,
        )
        _write_run_json_artifact(
            session_id=session_id,
            run_id=run_id,
            stage="verifier",
            title="Verification report",
            relative_path="verifier.json",
            payload=verification,
        )
        event = _record_run_event(
            session_id=session_id,
            run_id=run_id,
            event_type="verification_completed",
            title="Verification completed",
            detail=f"Verifier status: {verification.get('status')}",
            payload=verification,
        )
        _append_run_log(session_id, run_id, event)

        _set_run_progress(
            run_id,
            {
                "phase": "finalizing",
                "workflow_type": workflow_plan["workflow_type"],
                "risk_level": workflow_plan["risk_level"],
                "verification_status": verification.get("status"),
                "current_step": len(steps),
                "total_steps": len(steps),
            },
        )

        final_answer = await _build_final_answer(task_text, step_outputs, verification)
        _write_run_text_artifact(
            session_id=session_id,
            run_id=run_id,
            stage="final",
            title="Final answer",
            relative_path="final.md",
            content=final_answer,
        )
        final_msg = _save_message(
            session_id=session_id,
            window_id=window_id,
            role="assistant",
            content_text=final_answer,
            message_type="assistant",
            turn_id=f"{run_id}:final",
            source="run",
            content_json={
                "run_id": run_id,
                "final": True,
                "workflow_plan": workflow_plan,
                "facts": facts_payload,
                "verification": verification,
            },
        )
        window_id = _maybe_rollover_for_background_run(
            session_id=session_id,
            window_id=window_id,
            run_id=run_id,
        )

        _set_run_status(run_id, "completed", result_message_id=final_msg["id"], finished=True)
        _set_run_progress(
            run_id,
            {
                "phase": "completed",
                "workflow_type": workflow_plan["workflow_type"],
                "risk_level": workflow_plan["risk_level"],
                "verification_status": verification.get("status"),
                "current_step": len(steps),
                "total_steps": len(steps),
            },
        )
        event = _record_run_event(
            session_id=session_id,
            run_id=run_id,
            event_type="completed",
            title="Run completed",
            detail="Final answer saved to chat.",
            payload={"result_message_id": final_msg["id"], "verification": verification.get("status")},
        )
        _append_run_log(session_id, run_id, event)
        _save_run_system_message(
            session_id=session_id,
            run_id=run_id,
            text="Done. The result has been added to the chat.",
            window_id=window_id,
        )
    finally:
        if mcp_registry is not None:
            await mcp_registry.close()


async def _run_once() -> bool:
    run = _claim_next_run()
    if run is None:
        return False

    run_id = run["id"]
    try:
        await _execute_run(run)
    except Exception as exc:
        _set_run_status(run_id, "failed", error_text=f"{type(exc).__name__}: {exc}", finished=True)
        event = _record_run_event(
            session_id=run["session_id"],
            run_id=run_id,
            event_type="failed",
            title="Run failed",
            detail=f"{type(exc).__name__}: {exc}",
            payload={"error_type": type(exc).__name__, "error": str(exc)},
        )
        _append_run_log(run["session_id"], run_id, event)
        latest = get_last_window(run["session_id"])
        if latest:
            _save_run_system_message(
                session_id=run["session_id"],
                run_id=run_id,
                text=f"Execution error: {type(exc).__name__}: {exc}",
                window_id=latest["id"],
            )
    return True


async def run_worker_loop(stop_event: asyncio.Event) -> None:
    _requeue_inflight_runs()
    while not stop_event.is_set():
        handled = await _run_once()
        if handled:
            continue
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=1.5)
        except asyncio.TimeoutError:
            continue
