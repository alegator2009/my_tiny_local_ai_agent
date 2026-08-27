from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from ..config import load_app_config
from ..db import execute, fetch_all, fetch_one, utcnow_iso
from .provider_http import (
    build_payload,
    provider_timeout_seconds,
    resolve_provider_model,
)

EVOLUTION_ACTIVE_STATUSES = {"queued", "running"}
EVOLUTION_TERMINAL_STATUSES = {"completed", "failed", "canceled"}
DEFAULT_PROMPT = (
    "Inspect this project and implement one conservative improvement. Preserve existing behavior, "
    "add or update tests when the change needs them, and keep the change small enough to validate."
)
ANYWHERE_IGNORES = {
    ".DS_Store",
    ".git",
    ".mypy_cache",
    ".next",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "dist",
    "node_modules",
}
TOP_LEVEL_IGNORES = {
    ".playwright-mcp",
    "data",
    "evolution",
}
PROJECT_CONTEXT_MAX_CHARS = 36000
PROJECT_CONTEXT_FILE_MAX_CHARS = 12000
IMPLEMENTATION_MAX_FILES = 16
TEXT_CONTEXT_EXTENSIONS = {
    ".css",
    ".env",
    ".html",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".mjs",
    ".py",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}


@dataclass
class CommandResult:
    label: str
    command: list[str]
    cwd: str
    ok: bool
    returncode: int | None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None


@dataclass
class GenerationResult:
    generation: int
    artifact_dir: Path
    child_repo: Path
    ok: bool
    implementation: dict[str, Any]
    self_test: dict[str, Any]
    change_check: dict[str, Any]
    test_results: dict[str, Any]
    handoff: dict[str, Any] = field(default_factory=dict)


def repo_root() -> Path:
    """Return the repository root in both source and container layouts."""
    configured_root = os.getenv("EVOLUTION_REPO_ROOT", "").strip()
    if configured_root:
        candidate = Path(configured_root).expanduser()
        if candidate.is_dir():
            return candidate.resolve()

    source_file = Path(__file__).resolve()
    for candidate in (Path.cwd(), *source_file.parents):
        # `app/evolution` is this API's Python package, not the project
        # lineage. A valid repository root also contains the application tree.
        if (candidate / "evolution").is_dir() and (candidate / "apps").is_dir():
            return candidate

    # Docker places the application under /app/app, while only the API source
    # is copied into the image.  /app is still the correct safe fallback.
    return source_file.parents[2]


def default_lineage_root(root: Path | None = None) -> Path:
    return (root or repo_root()) / "evolution"


def _json_loads_safe(raw: Any, fallback: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return fallback
    return fallback if raw is None else raw


def _row_to_run(row: Any) -> dict[str, Any]:
    progress = _json_loads_safe(row["progress_json"], {})
    score = _json_loads_safe(row["score_json"], {})
    return {
        "id": row["id"],
        "prompt": row["prompt"],
        "status": row["status"],
        "mode": row["mode"],
        "max_generations": row["max_generations"],
        "stop_on_failure": bool(row["stop_on_failure"]),
        "current_generation": row["current_generation"],
        "parent_generation": row["parent_generation"],
        "child_generation": row["child_generation"],
        "lineage_root_path": row["lineage_root_path"],
        "parent_repo_path": row["parent_repo_path"],
        "child_repo_path": row["child_repo_path"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "progress_json": progress if isinstance(progress, dict) else {},
        "score_json": score if isinstance(score, dict) else {},
        "error_text": row["error_text"],
    }


def _row_to_event(row: Any) -> dict[str, Any]:
    payload = _json_loads_safe(row["payload_json"], {})
    return {
        "id": row["id"],
        "run_id": row["run_id"],
        "generation": row["generation"],
        "event_type": row["event_type"],
        "title": row["title"],
        "detail": row["detail"],
        "payload_json": payload if isinstance(payload, dict) else {},
        "timestamp": row["timestamp"],
    }


def get_evolution_run(run_id: str) -> dict[str, Any]:
    row = fetch_one("SELECT * FROM evolution_runs WHERE id=?", (run_id,))
    if row is None:
        raise KeyError("evolution_run_not_found")
    return _row_to_run(row)


def list_evolution_runs(limit: int = 50) -> list[dict[str, Any]]:
    rows = fetch_all(
        "SELECT * FROM evolution_runs ORDER BY created_at DESC LIMIT ?",
        (max(1, min(limit, 200)),),
    )
    return [_row_to_run(row) for row in rows]


def list_evolution_events(run_id: str) -> list[dict[str, Any]]:
    _ = get_evolution_run(run_id)
    rows = fetch_all(
        "SELECT * FROM evolution_events WHERE run_id=? ORDER BY timestamp ASC",
        (run_id,),
    )
    return [_row_to_event(row) for row in rows]


def list_generations(lineage_root: Path | None = None) -> list[dict[str, Any]]:
    lineage = (lineage_root or default_lineage_root()).resolve()
    active = read_active_generation(lineage)
    generations = []
    if not lineage.exists():
        return []
    for artifact_dir in sorted(lineage.iterdir(), key=lambda path: generation_number_from_name(path.name) or -1):
        generation = generation_number_from_name(artifact_dir.name)
        if generation is None or not artifact_dir.is_dir():
            continue
        item = generation_summary(artifact_dir, generation, active)
        generations.append(item)
    return generations


def activate_generation(generation: int, lineage_root: Path | None = None) -> dict[str, Any]:
    lineage = (lineage_root or default_lineage_root()).resolve()
    artifact_dir = generation_artifact_dir(lineage, generation)
    if not artifact_dir.exists():
        raise KeyError("generation_not_found")
    child_repo = artifact_dir / "repo"
    if not child_repo.exists():
        raise ValueError("generation has no child repository")

    summary = generation_summary(artifact_dir, generation, active_generation=None)
    write_json(
        lineage / "active.json",
        {
            "active_generation": generation,
            "latest_generation": latest_generation_number(lineage),
            "latest_ok": summary["status"] == "passed",
            "updated_at": utcnow_iso(),
            "child_repo": str(child_repo),
            "manual_activation": True,
        },
    )
    return generation_summary(artifact_dir, generation, active_generation=generation)


def delete_generation(generation: int, *, force: bool = False, lineage_root: Path | None = None) -> dict[str, Any]:
    lineage = (lineage_root or default_lineage_root()).resolve()
    artifact_dir = generation_artifact_dir(lineage, generation)
    if not artifact_dir.exists():
        raise KeyError("generation_not_found")
    active = read_active_generation(lineage)
    if active == generation and not force:
        raise ValueError("cannot delete active generation without force")
    if not is_relative_to(artifact_dir.resolve(), lineage):
        raise ValueError("generation path is outside lineage root")

    shutil.rmtree(artifact_dir)
    new_active = read_active_generation(lineage)
    if new_active == generation:
        new_active = latest_passed_generation(lineage)
        latest = latest_generation_number(lineage)
        payload: dict[str, Any] = {
            "active_generation": new_active,
            "latest_generation": latest,
            "latest_ok": new_active is not None,
            "updated_at": utcnow_iso(),
        }
        if new_active is not None:
            payload["child_repo"] = str(generation_artifact_dir(lineage, new_active) / "repo")
        write_json(lineage / "active.json", payload)
    return {"ok": True, "deleted_generation": generation, "active_generation": new_active}


def copy_generation_to_root(
    generation: int,
    *,
    lineage_root: Path | None = None,
    root_repo: Path | None = None,
) -> dict[str, Any]:
    lineage = (lineage_root or default_lineage_root()).resolve()
    root = (root_repo or repo_root()).resolve()
    artifact_dir = generation_artifact_dir(lineage, generation)
    if not artifact_dir.exists():
        raise KeyError("generation_not_found")

    source_repo = (artifact_dir / "repo").resolve()
    if not source_repo.exists():
        raise ValueError("generation has no child repository")
    if source_repo == root:
        raise ValueError("generation repository is already the root repository")
    if is_relative_to(root, source_repo):
        raise ValueError("refusing to copy into a path inside the source generation")
    if is_relative_to(source_repo, root) and not is_relative_to(source_repo, lineage):
        raise ValueError("refusing to copy from a nested root path outside the lineage")

    sync_project_tree(source_repo=source_repo, root_repo=root)
    copied_at = utcnow_iso()
    append_lineage_event(
        lineage,
        {
            "type": "copy_to_root",
            "generation": generation,
            "source_repo": str(source_repo),
            "root_repo": str(root),
            "timestamp": copied_at,
        },
    )
    return {
        "ok": True,
        "generation": generation,
        "root_repo_path": str(root),
        "source_repo_path": str(source_repo),
        "copied_at": copied_at,
    }


def create_evolution_run(
    *,
    prompt: str,
    max_generations: int,
    mode: str,
    stop_on_failure: bool,
    parent_repo: Path | None = None,
    lineage_root: Path | None = None,
) -> dict[str, Any]:
    root = repo_root()
    lineage = (lineage_root or default_lineage_root(root)).resolve()
    parent = (parent_repo.resolve() if parent_repo is not None else default_evolution_parent_repo(lineage)).resolve()
    parent_generation = parse_generation_from_repo(parent)
    run_id = str(uuid.uuid4())
    now = utcnow_iso()
    normalized_prompt = prompt.strip() or DEFAULT_PROMPT
    execute(
        """
        INSERT INTO evolution_runs (
          id, prompt, status, mode, max_generations, stop_on_failure, current_generation, parent_generation,
          lineage_root_path, parent_repo_path, created_at, updated_at, progress_json, score_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            normalized_prompt,
            "queued",
            mode,
            max(1, min(int(max_generations), 20)),
            1 if stop_on_failure else 0,
            0,
            parent_generation,
            str(lineage),
            str(parent),
            now,
            now,
            json.dumps({"stage": "queued"}, ensure_ascii=False),
            "{}",
        ),
    )
    _record_event(
        run_id=run_id,
        event_type="queued",
        title="Evolution queued",
        detail="A project copy will be created, modified, and validated.",
        payload={"parent_repo": str(parent), "lineage_root": str(lineage)},
    )
    return get_evolution_run(run_id)


def default_evolution_parent_repo(lineage_root: Path) -> Path:
    active_generation = read_active_generation(lineage_root)
    if active_generation is not None:
        active_repo = generation_artifact_dir(lineage_root, active_generation) / "repo"
        if active_repo.exists():
            return active_repo

    active_payload = _json_file(lineage_root / "active.json", {})
    active_child = active_payload.get("child_repo") if isinstance(active_payload, dict) else None
    if isinstance(active_child, str) and active_child.strip():
        active_repo = Path(active_child).expanduser()
        if active_repo.exists():
            return active_repo

    return repo_root()


def cancel_evolution_run(run_id: str) -> dict[str, Any]:
    run = get_evolution_run(run_id)
    if run["status"] not in EVOLUTION_ACTIVE_STATUSES:
        return run
    now = utcnow_iso()
    execute(
        """
        UPDATE evolution_runs
        SET status='canceled', updated_at=?, finished_at=COALESCE(finished_at, ?), error_text=?
        WHERE id=?
        """,
        (now, now, "Canceled by user.", run_id),
    )
    _record_event(
        run_id=run_id,
        event_type="canceled",
        title="Evolution canceled",
        detail="Cancellation was recorded. Already running child processes are not force-killed.",
    )
    return get_evolution_run(run_id)


def run_evolution(run_id: str) -> dict[str, Any]:
    run = get_evolution_run(run_id)
    if run["status"] == "canceled":
        return run
    if run["status"] not in EVOLUTION_ACTIVE_STATUSES:
        return run

    now = utcnow_iso()
    execute(
        """
        UPDATE evolution_runs
        SET status='running', started_at=COALESCE(started_at, ?), updated_at=?, progress_json=?
        WHERE id=?
        """,
        (now, now, json.dumps({"stage": "starting"}, ensure_ascii=False), run_id),
    )
    _record_event(
        run_id=run_id,
        event_type="started",
        title="Evolution started",
        detail="Preparing an isolated project workspace.",
    )

    try:
        result = perform_generation(
            parent_repo=Path(run["parent_repo_path"]),
            lineage_root=Path(run["lineage_root_path"]),
            prompt=run["prompt"],
            mode=run["mode"],
            stop_on_failure=bool(run["stop_on_failure"]),
            run_id=run_id,
            parent_generation=run["parent_generation"],
            run_tests=True,
        )
        remaining = int(run["max_generations"]) - 1
        if result.ok and remaining > 0:
            handoff = handoff_to_child(
                child_repo=result.child_repo,
                lineage_root=Path(run["lineage_root_path"]),
                prompt=run["prompt"],
                mode=run["mode"],
                stop_on_failure=bool(run["stop_on_failure"]),
                remaining_generations=remaining,
                run_id=run_id,
                generation=result.generation,
            )
            result.handoff = handoff
            write_json(result.artifact_dir / "handoff-result.json", handoff)
            _record_event(
                run_id=run_id,
                generation=result.generation,
                event_type="handoff_finished",
                title="Generation handoff finished",
                detail="Next project runner completed." if handoff["ok"] else "Next project runner failed.",
                payload=handoff,
            )
            result.ok = result.ok and handoff["ok"]

        status = "completed" if result.ok else "failed"
        deepest_generation = deepest_generation_from_result(result)
        deepest_child_repo = generation_artifact_dir(Path(run["lineage_root_path"]), deepest_generation) / "repo"
        finished = utcnow_iso()
        score = {
            "self_test_ok": bool(result.self_test.get("ok")),
            "change_check_ok": bool(result.change_check.get("ok")),
            "tests_ok": bool(result.test_results.get("ok")),
            "handoff_ok": bool(result.handoff.get("ok", True)),
            "deepest_generation": deepest_generation,
        }
        execute(
            """
            UPDATE evolution_runs
            SET status=?, current_generation=?, child_generation=?, child_repo_path=?,
                updated_at=?, finished_at=?, progress_json=?, score_json=?, error_text=?
            WHERE id=?
            """,
            (
                status,
                deepest_generation,
                deepest_generation,
                str(deepest_child_repo),
                finished,
                finished,
                json.dumps(
                    {
                        "stage": status,
                        "generation": deepest_generation,
                        "artifact_dir": str(generation_artifact_dir(Path(run["lineage_root_path"]), deepest_generation)),
                        "child_repo": str(deepest_child_repo),
                    },
                    ensure_ascii=False,
                ),
                json.dumps(score, ensure_ascii=False),
                None if result.ok else "Evolution validation failed.",
                run_id,
            ),
        )
        _record_event(
            run_id=run_id,
            generation=deepest_generation,
            event_type=status,
            title="Evolution completed" if result.ok else "Evolution failed",
            detail=f"Generation agent-{deepest_generation:03d} validation {'passed' if result.ok else 'failed'}.",
            payload=score,
        )
    except Exception as exc:
        finished = utcnow_iso()
        execute(
            """
            UPDATE evolution_runs
            SET status='failed', updated_at=?, finished_at=?, error_text=?, progress_json=?
            WHERE id=?
            """,
            (
                finished,
                finished,
                str(exc),
                json.dumps({"stage": "failed", "error": str(exc)}, ensure_ascii=False),
                run_id,
            ),
        )
        _record_event(
            run_id=run_id,
            event_type="failed",
            title="Evolution crashed",
            detail=str(exc),
        )

    return get_evolution_run(run_id)


def perform_generation(
    *,
    parent_repo: Path,
    lineage_root: Path,
    prompt: str,
    mode: str,
    stop_on_failure: bool,
    run_id: str | None = None,
    parent_generation: int | None = None,
    run_tests: bool = True,
) -> GenerationResult:
    parent = parent_repo.resolve()
    lineage = lineage_root.resolve()
    if not parent.exists():
        raise FileNotFoundError(f"Parent repository does not exist: {parent}")
    lineage.mkdir(parents=True, exist_ok=True)
    previous_active_generation = read_active_generation(lineage)
    generation = next_generation_number(lineage)
    artifact_dir = lineage / f"agent-{generation:03d}"
    child_repo = artifact_dir / "repo"

    if artifact_dir.exists():
        raise FileExistsError(f"Evolution generation already exists: {artifact_dir}")

    write_progress(run_id, "copying", generation, child_repo)
    copy_repository(parent, child_repo)
    write_scaffolding(
        artifact_dir=artifact_dir,
        parent_repo=parent,
        child_repo=child_repo,
        lineage_root=lineage,
        prompt=prompt,
        mode=mode,
        generation=generation,
        parent_generation=parent_generation,
    )
    _record_event(
        run_id=run_id,
        generation=generation,
        event_type="copied",
        title=f"agent-{generation:03d} copied",
        detail=f"Child repository created at {child_repo}",
        payload={"child_repo": str(child_repo)},
    )

    write_progress(run_id, "implementation", generation, child_repo)
    implementation = apply_implementation_worker(child_repo=child_repo, prompt=prompt, mode=mode)
    write_json(artifact_dir / "implementation-result.json", implementation)
    _record_event(
        run_id=run_id,
        generation=generation,
        event_type="implementation",
        title="Implementation applied" if implementation["ok"] else "Implementation skipped",
        detail=implementation["summary"],
        payload=implementation,
    )
    update_changes_artifact(artifact_dir, prompt=prompt, implementation=implementation)

    write_progress(run_id, "self_test", generation, child_repo)
    self_test = run_self_test(
        parent_repo=parent,
        child_repo=child_repo,
        artifact_dir=artifact_dir,
        lineage_root=lineage,
        generation=generation,
    )
    write_json(artifact_dir / "self-test.json", self_test)
    _record_event(
        run_id=run_id,
        generation=generation,
        event_type="self_test",
        title="Self-test passed" if self_test["ok"] else "Self-test failed",
        detail="Child write scope and artifact layout were checked.",
        payload=self_test,
    )

    write_progress(run_id, "change_check", generation, child_repo)
    change_check = detect_substantive_changes(parent_repo=parent, child_repo=child_repo)
    write_json(artifact_dir / "change-check.json", change_check)
    _record_event(
        run_id=run_id,
        generation=generation,
        event_type="change_check",
        title="Implementation changes found" if change_check["ok"] else "No implementation changes",
        detail=change_check["summary"],
        payload=change_check,
    )

    write_progress(run_id, "validation", generation, child_repo)
    test_results = run_validation_tests(child_repo=child_repo, parent_repo=parent, enabled=run_tests and change_check["ok"])
    write_json(artifact_dir / "test-results.json", test_results)
    _record_event(
        run_id=run_id,
        generation=generation,
        event_type="validation",
        title="Validation passed" if test_results["ok"] else "Validation failed",
        detail="Configured test commands finished.",
        payload={"ok": test_results["ok"], "commands": _compact_command_results(test_results["results"])},
    )

    ok = bool(self_test["ok"]) and bool(change_check["ok"]) and bool(test_results["ok"])
    final_report = build_final_report(
        generation=generation,
        prompt=prompt,
        mode=mode,
        child_repo=child_repo,
        implementation=implementation,
        self_test=self_test,
        change_check=change_check,
        test_results=test_results,
        ok=ok,
    )
    (artifact_dir / "final-report.md").write_text(final_report, encoding="utf-8")
    append_lineage_event(
        lineage,
        {
            "timestamp": utcnow_iso(),
            "generation": generation,
            "parent_generation": parent_generation,
            "parent_repo": str(parent),
            "child_repo": str(child_repo),
            "mode": mode,
            "ok": ok,
            "prompt": prompt,
        },
    )
    active_generation = generation if ok else (parent_generation if parent_generation is not None else previous_active_generation)
    active_child_repo = generation_artifact_dir(lineage, active_generation) / "repo" if active_generation is not None else None
    write_json(
        lineage / "active.json",
        {
            "active_generation": active_generation,
            "latest_generation": generation,
            "latest_ok": ok,
            "updated_at": utcnow_iso(),
            "child_repo": str(active_child_repo) if active_child_repo is not None else None,
        },
    )
    return GenerationResult(
        generation=generation,
        artifact_dir=artifact_dir,
        child_repo=child_repo,
        ok=ok,
        implementation=implementation,
        self_test=self_test,
        change_check=change_check,
        test_results=test_results,
    )


def run_lineage_cli(
    *,
    parent_repo: Path,
    lineage_root: Path,
    prompt: str,
    mode: str,
    stop_on_failure: bool,
    remaining_generations: int,
) -> dict[str, Any]:
    created: list[dict[str, Any]] = []
    current_parent = parent_repo.resolve()
    parent_generation = parse_generation_from_repo(current_parent)
    ok = True

    for _ in range(max(0, remaining_generations)):
        result = perform_generation(
            parent_repo=current_parent,
            lineage_root=lineage_root,
            prompt=prompt,
            mode=mode,
            stop_on_failure=stop_on_failure,
            parent_generation=parent_generation,
            run_tests=True,
        )
        created.append(
            {
                "generation": result.generation,
                "child_repo": str(result.child_repo),
                "ok": result.ok,
            }
        )
        ok = ok and result.ok
        if not result.ok and stop_on_failure:
            break
        current_parent = result.child_repo
        parent_generation = result.generation

    return {"ok": ok, "created": created}


def handoff_to_child(
    *,
    child_repo: Path,
    lineage_root: Path,
    prompt: str,
    mode: str,
    stop_on_failure: bool,
    remaining_generations: int,
    run_id: str | None,
    generation: int,
) -> dict[str, Any]:
    artifact_dir = lineage_root / f"agent-{generation:03d}"
    prompt_file = artifact_dir / "prompt.md"
    prompt_file.write_text(prompt, encoding="utf-8")
    python_cmd = python_executable(child_repo, repo_root())
    command = [
        python_cmd,
        "-m",
        "app.evolution.runner",
        "continue",
        "--lineage-root",
        str(lineage_root),
        "--prompt-file",
        str(prompt_file),
        "--remaining-generations",
        str(remaining_generations),
        "--mode",
        mode,
    ]
    if stop_on_failure:
        command.append("--stop-on-failure")
    handoff = {
        "from_generation": generation,
        "remaining_generations": remaining_generations,
        "command": command,
        "cwd": str(child_repo),
        "created_at": utcnow_iso(),
    }
    write_json(artifact_dir / "handoff.json", handoff)
    _record_event(
        run_id=run_id,
        generation=generation,
        event_type="handoff_started",
        title="Handing off to next generation",
        detail=f"agent-{generation:03d} will run the next project task from its own workspace.",
        payload={"command": command, "cwd": str(child_repo)},
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(child_repo / "apps" / "api")
    completed = run_command(
        label="child handoff",
        command=command,
        cwd=child_repo,
        timeout_sec=max(180, 600 * remaining_generations),
        env=env,
    )
    parsed_stdout = parse_child_handoff_stdout(completed.stdout)
    return {
        **handoff,
        "ok": completed.ok,
        "returncode": completed.returncode,
        "created": parsed_stdout.get("created", []),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "error": completed.error,
    }


def next_generation_number(lineage_root: Path) -> int:
    existing: list[int] = []
    if lineage_root.exists():
        for path in lineage_root.iterdir():
            match = re.fullmatch(r"agent-(\d{3,})", path.name)
            if match and path.is_dir():
                existing.append(int(match.group(1)))
    return (max(existing) + 1) if existing else 1


def latest_generation_number(lineage_root: Path) -> int | None:
    existing = [
        generation
        for path in lineage_root.iterdir()
        if path.is_dir() and (generation := generation_number_from_name(path.name)) is not None
    ] if lineage_root.exists() else []
    return max(existing) if existing else None


def generation_number_from_name(name: str) -> int | None:
    match = re.fullmatch(r"agent-(\d{3,})", name)
    return int(match.group(1)) if match else None


def generation_artifact_dir(lineage_root: Path, generation: int) -> Path:
    if generation < 1:
        raise ValueError("generation must be positive")
    return lineage_root.resolve() / f"agent-{generation:03d}"


def read_active_generation(lineage_root: Path) -> int | None:
    payload = _json_file(lineage_root / "active.json", {})
    raw = payload.get("active_generation") if isinstance(payload, dict) else None
    try:
        return int(raw) if raw is not None else None
    except Exception:
        return None


def latest_passed_generation(lineage_root: Path) -> int | None:
    passed = [item["generation"] for item in list_generations(lineage_root) if item["status"] == "passed"]
    return max(passed) if passed else None


def generation_summary(artifact_dir: Path, generation: int, active_generation: int | None) -> dict[str, Any]:
    meta = _json_file(artifact_dir / "meta.json", {})
    self_test = _json_file(artifact_dir / "self-test.json", {})
    test_results = _json_file(artifact_dir / "test-results.json", {})
    change_check = _json_file(artifact_dir / "change-check.json", {})
    prompt_path = artifact_dir / "prompt.md"
    prompt = prompt_path.read_text(encoding="utf-8").strip() if prompt_path.exists() else ""
    improvement_summary = read_improvement_summary(artifact_dir, prompt)
    self_test_ok = self_test.get("ok") if isinstance(self_test, dict) else None
    tests_ok = test_results.get("ok") if isinstance(test_results, dict) else None
    change_ok = change_check.get("ok") if isinstance(change_check, dict) and "ok" in change_check else None
    if self_test_ok is True and tests_ok is True and change_ok is not False:
        status = "passed"
    elif self_test_ok is False or tests_ok is False or change_ok is False:
        status = "failed"
    else:
        status = "unknown"
    child_repo = artifact_dir / "repo"
    active = active_generation == generation
    return {
        "generation": generation,
        "name": f"agent-{generation:03d}",
        "status": status,
        "active": active,
        "artifact_dir": str(artifact_dir),
        "child_repo_path": str(child_repo) if child_repo.exists() else None,
        "prompt": prompt,
        "improvement_summary": improvement_summary,
        "mode": meta.get("mode") if isinstance(meta, dict) else None,
        "created_at": meta.get("created_at") if isinstance(meta, dict) else None,
        "self_test_ok": self_test_ok if isinstance(self_test_ok, bool) else None,
        "tests_ok": tests_ok if isinstance(tests_ok, bool) else None,
        "has_handoff": (artifact_dir / "handoff-result.json").exists(),
        "deletable": (not active) or status == "failed",
    }


def read_improvement_summary(artifact_dir: Path, prompt: str) -> str:
    changes_path = artifact_dir / "changes.md"
    if changes_path.exists():
        text = changes_path.read_text(encoding="utf-8").strip()
        for line in text.splitlines():
            normalized = line.strip().lstrip("-* ").strip()
            if normalized.lower().startswith("selected task:"):
                value = normalized.split(":", 1)[1].strip()
                if value:
                    return clamp_summary(value)
            if normalized.lower().startswith("requested task:"):
                value = normalized.split(":", 1)[1].strip()
                if value:
                    return clamp_summary(value)
            if normalized.lower().startswith("requested improvement:"):
                value = normalized.split(":", 1)[1].strip()
                if value:
                    return clamp_summary(value)
            if (
                normalized
                and not normalized.startswith("#")
                and "isolated project repository" not in normalized
                and "isolated child repository" not in normalized
            ):
                return clamp_summary(normalized)
    return clamp_summary(prompt or "Evolution scaffold and validation protocol.")


def clamp_summary(value: str, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", value).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def deepest_generation_from_result(result: GenerationResult) -> int:
    deepest = result.generation
    for item in result.handoff.get("created", []):
        try:
            deepest = max(deepest, int(item.get("generation")))
        except Exception:
            continue
    return deepest


def parse_child_handoff_stdout(stdout: str) -> dict[str, Any]:
    text = (stdout or "").strip()
    if not text:
        return {}
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _json_file(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def parse_generation_from_repo(path: Path) -> int | None:
    for part in path.resolve().parts:
        match = re.fullmatch(r"agent-(\d{3,})", part)
        if match:
            return int(match.group(1))
    return None


def copy_repository(parent_repo: Path, child_repo: Path) -> None:
    parent = parent_repo.resolve()

    def ignore(src: str, names: list[str]) -> set[str]:
        ignored = set(ANYWHERE_IGNORES.intersection(names))
        if Path(src).resolve() == parent:
            ignored.update(TOP_LEVEL_IGNORES.intersection(names))
        return ignored

    shutil.copytree(parent, child_repo, ignore=ignore)


def sync_project_tree(*, source_repo: Path, root_repo: Path) -> None:
    source = source_repo.resolve()
    root = root_repo.resolve()
    if not source.exists():
        raise ValueError("source repository does not exist")
    if not root.exists():
        raise ValueError("root repository does not exist")
    if not source.is_dir() or not root.is_dir():
        raise ValueError("source and root must be directories")
    _sync_project_dir(source, root, source, root)


def _sync_project_dir(source_dir: Path, target_dir: Path, source_root: Path, target_root: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)

    for target_child in list(target_dir.iterdir()):
        rel = target_child.relative_to(target_root)
        if should_preserve_root_copy_path(rel):
            continue
        source_child = source_root / rel
        if not source_child.exists():
            if target_child.is_dir() and not target_child.is_symlink() and has_preserved_root_copy_descendant(target_child, target_root):
                prune_project_dir(target_child, target_root)
            else:
                remove_project_path(target_child, target_root)
            continue
        if source_child.is_dir() != target_child.is_dir():
            if target_child.is_dir() and not target_child.is_symlink() and has_preserved_root_copy_descendant(target_child, target_root):
                raise ValueError(f"refusing to replace directory containing preserved runtime files: {target_child}")
            remove_project_path(target_child, target_root)

    for source_child in source_dir.iterdir():
        rel = source_child.relative_to(source_root)
        if should_preserve_root_copy_path(rel):
            continue
        target_child = target_root / rel
        copy_project_path(source_child, target_child, source_root, target_root)


def should_preserve_root_copy_path(rel: Path) -> bool:
    parts = rel.parts
    if not parts:
        return True
    return any(part in ANYWHERE_IGNORES or part in TOP_LEVEL_IGNORES for part in parts)


def has_preserved_root_copy_descendant(path: Path, target_root: Path) -> bool:
    for descendant in path.rglob("*"):
        try:
            rel = descendant.resolve().relative_to(target_root.resolve())
        except ValueError:
            continue
        if should_preserve_root_copy_path(rel):
            return True
    return False


def prune_project_dir(path: Path, target_root: Path) -> None:
    for child in list(path.iterdir()):
        rel = child.resolve().relative_to(target_root.resolve())
        if should_preserve_root_copy_path(rel):
            continue
        if child.is_dir() and not child.is_symlink() and has_preserved_root_copy_descendant(child, target_root):
            prune_project_dir(child, target_root)
            continue
        remove_project_path(child, target_root)


def remove_project_path(path: Path, target_root: Path) -> None:
    resolved = path.resolve()
    root = target_root.resolve()
    if not is_relative_to(resolved, root):
        raise ValueError(f"refusing to remove path outside root: {path}")
    rel = resolved.relative_to(root)
    if should_preserve_root_copy_path(rel):
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def copy_project_path(source: Path, target: Path, source_root: Path, target_root: Path) -> None:
    rel = source.relative_to(source_root)
    if should_preserve_root_copy_path(rel):
        return
    if not is_relative_to(target.resolve() if target.exists() else target.absolute(), target_root.resolve()):
        raise ValueError(f"refusing to copy outside root: {target}")

    if source.is_dir() and not source.is_symlink():
        if target.exists() and not target.is_dir():
            remove_project_path(target, target_root)
        _sync_project_dir(source, target, source_root, target_root)
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        remove_project_path(target, target_root)
    if source.is_symlink():
        os.symlink(os.readlink(source), target)
    else:
        shutil.copy2(source, target)


def write_scaffolding(
    *,
    artifact_dir: Path,
    parent_repo: Path,
    child_repo: Path,
    lineage_root: Path,
    prompt: str,
    mode: str,
    generation: int,
    parent_generation: int | None,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "generation": generation,
        "parent_generation": parent_generation,
        "created_at": utcnow_iso(),
        "parent_repo": str(parent_repo),
        "child_repo": str(child_repo),
        "lineage_root": str(lineage_root),
        "mode": mode,
        "write_scope": str(child_repo),
    }
    write_json(artifact_dir / "meta.json", meta)
    (artifact_dir / "prompt.md").write_text(prompt, encoding="utf-8")
    plan = {
        "protocol": [
            "Copy the source project into an adjacent generation directory.",
            "Modify only files under the new project write scope.",
            "Run self-test and validation commands from the new project directory.",
            "Continue to the next generation only after validation passes.",
        ],
        "requested_task": prompt,
        "mode": mode,
    }
    write_json(artifact_dir / "plan.json", plan)
    (artifact_dir / "changes.md").write_text(
        "# Changes\n\n"
        f"- Requested task: {prompt}\n"
        "- Added isolated generation scaffold, lineage metadata, and validation artifacts.\n"
        "- Preserved source project write-scope by applying changes only inside the new project copy.\n\n"
        "This generation was created as an isolated project repository. "
        "Scaffold files are not counted as implementation changes.\n",
        encoding="utf-8",
    )
    (child_repo / "EVOLUTION.md").write_text(
        "# Project Task Protocol\n\n"
        f"Project generation: agent-{generation:03d}\n\n"
        f"Mode: {mode}\n\n"
        "Working directory: this repository. Modify only files inside this repository.\n\n"
        "Requested task:\n\n"
        f"{prompt}\n\n"
        "Validation artifacts are stored beside this repository by the generation runner.\n",
        encoding="utf-8",
    )


def run_self_test(
    *,
    parent_repo: Path,
    child_repo: Path,
    artifact_dir: Path,
    lineage_root: Path,
    generation: int,
) -> dict[str, Any]:
    checks = [
        ("parent_exists", parent_repo.exists()),
        ("child_exists", child_repo.exists()),
        ("child_differs_from_parent", child_repo.resolve() != parent_repo.resolve()),
        ("child_inside_lineage", is_relative_to(child_repo.resolve(), lineage_root.resolve())),
        ("meta_exists", (artifact_dir / "meta.json").exists()),
        ("prompt_exists", (artifact_dir / "prompt.md").exists()),
        ("repo_has_readme_or_package", (child_repo / "README.md").exists() or (child_repo / "package.json").exists()),
        ("top_level_data_excluded", not (child_repo / "data").exists()),
        ("top_level_evolution_excluded", not (child_repo / "evolution").exists()),
        ("nested_evolution_code_present", (child_repo / "apps" / "api" / "app" / "evolution").exists()),
    ]
    write_scope_ok = assert_child_write_scope(child_repo / "EVOLUTION.md", child_repo)
    checks.append(("write_scope_guard", write_scope_ok))
    ok = all(value for _, value in checks)
    return {
        "ok": ok,
        "generation": generation,
        "checks": [{"name": name, "ok": bool(value)} for name, value in checks],
    }


def apply_implementation_worker(*, child_repo: Path, prompt: str, mode: str) -> dict[str, Any]:
    context = build_project_context(child_repo, task=prompt)
    task_plan = plan_generation_task(project_context=context, prompt=prompt, mode=mode)
    selected_task = task_plan.get("selected_task") if isinstance(task_plan.get("selected_task"), str) else ""
    selected_task = selected_task.strip() or prompt
    messages = build_project_agent_messages(
        project_context=context,
        user_prompt=prompt,
        task=selected_task,
        mode=mode,
        task_plan=task_plan,
    )
    raw = complete_implementation_json(messages)
    parsed = extract_json_object(raw)
    if not isinstance(parsed, dict):
        return {
            "ok": False,
            "summary": "Project agent did not return a valid JSON edit plan.",
            "files_changed": [],
            "worker": "model-project-agent",
            "mode": mode,
            "prompt": prompt,
            "selected_task": selected_task,
            "task_plan": task_plan,
            "raw_response_tail": raw[-2000:] if raw else "",
        }

    try:
        changed = write_model_file_edits(child_repo=child_repo, edit_plan=parsed)
        if not changed:
            retry_raw = complete_implementation_json(
                build_operation_retry_messages(
                    snippet_context=build_operation_snippet_context(child_repo=child_repo, task=selected_task, task_plan=task_plan),
                    user_prompt=prompt,
                    task=selected_task,
                    mode=mode,
                    task_plan=task_plan,
                )
            )
            retry_parsed = extract_json_object(retry_raw)
            if isinstance(retry_parsed, dict):
                parsed = retry_parsed
                raw = retry_raw
                changed = write_model_file_edits(child_repo=child_repo, edit_plan=parsed)
    except Exception as exc:
        return {
            "ok": False,
            "summary": f"Project agent edit plan was rejected: {exc}",
            "files_changed": [],
            "worker": "model-project-agent",
            "mode": mode,
            "prompt": prompt,
            "selected_task": selected_task,
            "task_plan": task_plan,
            "model_summary": str(parsed.get("summary") or ""),
            "raw_response_tail": raw[-2000:] if raw else "",
        }

    summary = str(parsed.get("summary") or "").strip()
    return {
        "ok": bool(changed),
        "summary": summary if changed else (summary or "Project agent returned no file changes."),
        "files_changed": changed,
        "worker": "model-project-agent",
        "mode": mode,
        "prompt": prompt,
        "selected_task": selected_task,
        "task_plan": task_plan,
        "raw_response_tail": raw[-2000:] if raw else "",
        "notes": parsed.get("notes") if isinstance(parsed.get("notes"), list) else [],
    }


def plan_generation_task(*, project_context: str, prompt: str, mode: str) -> dict[str, Any]:
    raw = complete_implementation_json(build_task_planner_messages(project_context=project_context, prompt=prompt, mode=mode))
    parsed = extract_json_object(raw)
    if not isinstance(parsed, dict):
        return {
            "selected_task": prompt,
            "reason": "Task planner did not return valid JSON; using the original user prompt.",
            "remaining_tasks": [],
            "raw_response_tail": raw[-2000:] if raw else "",
        }
    selected = str(parsed.get("selected_task") or "").strip()
    if not selected:
        parsed["selected_task"] = prompt
    parsed["raw_response_tail"] = raw[-2000:] if raw else ""
    if not isinstance(parsed.get("remaining_tasks"), list):
        parsed["remaining_tasks"] = []
    return parsed


def build_task_planner_messages(*, project_context: str, prompt: str, mode: str) -> list[dict[str, str]]:
    system = (
        "You are a senior product engineer planning one safe code change for a copied project directory. "
        "The user may give a broad or abstract request. Convert it into exactly one small, complete, testable task "
        "that can be implemented in this generation without breaking the project. "
        "If the user request contains many possible improvements, choose the first useful incomplete slice. "
        "If the project context already appears to contain that slice, choose the next useful incomplete slice. "
        "Do not choose a purely analytical task. The selected task must require at least one concrete source-file edit. "
        "Return only JSON and no markdown. "
        "Schema: "
        '{"selected_task":"one concrete implementation task",'
        '"reason":"why this slice is safe and useful",'
        '"likely_files":["relative/path"],'
        '"acceptance_checks":["observable checks"],'
        '"remaining_tasks":["later slices"]}.'
    )
    user = f"Mode: {mode}\n\nUser request:\n{prompt}\n\nProject context:\n{project_context}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_project_agent_messages(
    *,
    project_context: str,
    user_prompt: str,
    task: str,
    mode: str,
    task_plan: dict[str, Any] | None = None,
    retry_reason: str = "",
) -> list[dict[str, str]]:
    system = (
        "You are a coding agent working in a copied project directory. "
        "Implement the selected task by editing project files. "
        "Do not discuss identity, generations, self-improvement, or handoff. "
        "Return only one JSON object and no markdown. "
        "The JSON schema is: "
        '{"summary":"short summary",'
        '"operations":[{"path":"relative/path","find":"exact existing text","replace":"replacement text"}],'
        '"files":[{"path":"relative/path","content":"full file content"}],'
        '"delete_paths":["relative/path"],"notes":["optional note"]}. '
        "Use relative paths only. Do not include unchanged files. Do not write outside the project. "
        "For edits to existing files, prefer small operations over returning full file contents. "
        "Return full file contents only when creating a new file or when a small replacement is impossible. "
        "Prefer small, coherent changes that can pass the existing tests. "
        "The response must change at least one source file unless the selected task is impossible; if impossible, "
        "return a short note explaining the blocker and no files."
    )
    user = (
        f"Mode: {mode}\n\n"
        f"Original user request:\n{user_prompt}\n\n"
        f"Selected task for this generation:\n{task}\n\n"
        f"Task plan JSON:\n{json.dumps(task_plan or {}, ensure_ascii=False, indent=2)}\n\n"
        + (f"Retry instruction:\n{retry_reason}\n\n" if retry_reason else "")
        + "Project context follows. Use it to decide which files to edit.\n\n"
        f"{project_context}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_operation_retry_messages(
    *,
    snippet_context: str,
    user_prompt: str,
    task: str,
    mode: str,
    task_plan: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    system = (
        "You are fixing a failed edit plan. Return operation-only JSON. "
        "Do not return full file contents. Do not return markdown. "
        "Use exact text from the provided snippets in each find value. "
        "Schema: "
        '{"summary":"short summary",'
        '"operations":[{"path":"relative/path","find":"exact existing text from snippet","replace":"replacement text"}],'
        '"files":[],"delete_paths":[],"notes":["optional note"]}. '
        "The operations array must contain at least one operation unless the selected task is impossible."
    )
    user = (
        f"Mode: {mode}\n\n"
        f"Original user request:\n{user_prompt}\n\n"
        f"Selected task:\n{task}\n\n"
        f"Task plan JSON:\n{json.dumps(task_plan or {}, ensure_ascii=False, indent=2)}\n\n"
        "The previous attempt changed no files. Produce a minimal find/replace edit using only the snippets below.\n\n"
        f"{snippet_context}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_operation_snippet_context(*, child_repo: Path, task: str, task_plan: dict[str, Any]) -> str:
    likely_files = task_plan.get("likely_files") if isinstance(task_plan, dict) else []
    files: list[str] = []
    if isinstance(likely_files, list):
        files.extend(str(item) for item in likely_files if isinstance(item, str))
    if not files:
        files = select_project_context_files(child_repo, task=task)[:8]

    keywords = task_search_keywords(task)
    keywords.update(extract_literal_tokens(task))
    chunks = ["# Target Snippets"]
    for rel in files[:8]:
        try:
            path = resolve_model_edit_path(child_repo, rel)
        except Exception:
            continue
        if not path.is_file() or not is_context_file(path):
            continue
        snippet = matching_line_snippets(path, keywords)
        if not snippet:
            snippet = read_text_safe(path)[:PROJECT_CONTEXT_FILE_MAX_CHARS]
        chunks.append(f"\n## {rel}\n```{language_for_path(rel)}\n{snippet}\n```")
    return "\n".join(chunks)


def extract_literal_tokens(task: str) -> set[str]:
    tokens = set(re.findall(r"\b\d+(?:px|rem|em|vh|vw|%)?\b|#[0-9a-fA-F]{3,8}", task))
    quoted = re.findall(r"`([^`]+)`|'([^']+)'|\"([^\"]+)\"", task)
    for groups in quoted:
        for value in groups:
            if value:
                tokens.add(value)
    return {token for token in tokens if token}


def matching_line_snippets(path: Path, keywords: set[str], *, radius: int = 8, max_lines: int = 140) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return ""
    if not lines:
        return ""

    lower_keywords = {keyword.lower() for keyword in keywords if keyword}
    matching_indexes: set[int] = set()
    for index, line in enumerate(lines):
        lower = line.lower()
        if any(keyword in lower for keyword in lower_keywords):
            for snippet_index in range(max(0, index - radius), min(len(lines), index + radius + 1)):
                matching_indexes.add(snippet_index)

    if not matching_indexes:
        return ""
    selected = sorted(matching_indexes)
    if len(selected) > max_lines:
        selected = selected[:max_lines]

    out: list[str] = []
    previous = -2
    for index in selected:
        if previous != -2 and index > previous + 1:
            out.append("...")
        out.append(lines[index])
        previous = index
    return "\n".join(out)


def complete_implementation_json(prompt_messages: list[dict[str, str]]) -> str:
    provider, model = resolve_provider_model()
    if provider is None or model is None or not provider.base_url:
        return ""

    raw = post_implementation_completion(
        provider=provider, model=model, prompt_messages=prompt_messages
    )
    if raw:
        return raw
    fallback_pool: list[str] = []
    extras = model.extra_params_json or {}
    pool = extras.get("fallback_models")
    if isinstance(pool, list):
        fallback_pool = [str(p) for p in pool if str(p).strip()]
    if not fallback_pool:
        fallback_pool = ["smart-pool", "free-pool"]
    for fallback_model in fallback_pool:
        fallback = str(fallback_model or "").strip()
        if not fallback or fallback == model.name:
            continue
        raw = post_implementation_completion(
            provider=provider,
            model=model,
            prompt_messages=prompt_messages,
            override_model_name=fallback,
        )
        if raw:
            return raw
    return ""


def post_implementation_completion(
    *,
    provider: Any,
    model: Any,
    prompt_messages: list[dict[str, str]],
    override_model_name: str | None = None,
) -> str:
    """Synchronous single-shot completion used by the evolution harness.

    ``override_model_name`` lets the caller retry with a fallback model
    identifier while keeping the same provider's URL/auth/timeout."""

    url, headers, payload = build_payload(
        provider=provider,
        model=model,
        messages=prompt_messages,
        stream=False,
    )
    if override_model_name:
        payload["model"] = override_model_name
    # The evolution tests want a low temperature and a generous output
    # budget so the model can produce code.
    payload["temperature"] = min(float(payload.get("temperature") or 0.2), 0.2)
    payload["max_tokens"] = max(int(payload.get("max_tokens") or 4096), 4096)

    try:
        with httpx.Client(timeout=provider_timeout_seconds(provider)) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception:
        return ""

    choices = data.get("choices") or []
    message = (choices[0] if choices else {}).get("message") or {}
    return str(message.get("content") or "").strip()


def build_project_context(child_repo: Path, task: str = "") -> str:
    files = select_project_context_files(child_repo, task=task)
    chunks = ["# Project File Tree", "", "\n".join(files), "", "# File Contents"]
    budget = PROJECT_CONTEXT_MAX_CHARS - sum(len(chunk) for chunk in chunks)

    for rel in files:
        if budget <= 0:
            break
        path = child_repo / rel
        try:
            content = path.read_text(encoding="utf-8")
        except Exception:
            continue
        if len(content) > PROJECT_CONTEXT_FILE_MAX_CHARS:
            content = content[:PROJECT_CONTEXT_FILE_MAX_CHARS] + "\n...[truncated]\n"
        block = f"\n\n## {rel}\n```{language_for_path(rel)}\n{content}\n```"
        if len(block) > budget:
            continue
        chunks.append(block)
        budget -= len(block)
    return "\n".join(chunks)


def select_project_context_files(child_repo: Path, task: str = "") -> list[str]:
    base = child_repo.resolve()
    selected: list[str] = []
    priority = [
        "README.md",
        "package.json",
        "apps/web/package.json",
        "apps/web/app/page.tsx",
        "apps/web/app/layout.tsx",
        "apps/web/app/globals.css",
        "apps/web/components/EvolutionPanel.tsx",
        "apps/web/lib/api.ts",
        "apps/api/app/services/evolution.py",
        "apps/api/app/evolution/runner.py",
        "apps/api/app/routes/evolution.py",
        "apps/api/app/schemas.py",
        "apps/api/tests/test_evolution.py",
    ]
    for rel in priority:
        path = base / rel
        if path.is_file() and is_context_file(path):
            selected.append(rel)

    task_keywords = task_search_keywords(task)
    scored: list[tuple[int, str]] = []
    for path in sorted(base.rglob("*")):
        if not path.is_file() or not is_context_file(path):
            continue
        rel = path.relative_to(base)
        if should_ignore_context_path(rel):
            continue
        rel_text = rel.as_posix()
        if rel_text in selected:
            continue
        score = context_relevance_score(path, rel_text, task_keywords)
        if score > 0:
            scored.append((score, rel_text))

    for _, rel_text in sorted(scored, key=lambda item: (-item[0], item[1])):
        if rel_text not in selected:
            selected.append(rel_text)
        if len(selected) >= 28:
            break
    return selected


def task_search_keywords(task: str) -> set[str]:
    raw = re.findall(r"[\w#.-]{3,}", task.lower(), flags=re.UNICODE)
    synonyms = {
        # English
        "ui": {"ui", "web", "app", "tsx", "css", "component"},
        "interface": {"ui", "web", "app", "tsx", "css", "component"},
        "frontend": {"ui", "web", "app", "tsx", "css", "component"},
        "color": {"color", "accent", "css", "theme"},
        "colour": {"color", "accent", "css", "theme"},
        "accent": {"accent", "css", "theme"},
        "green": {"green", "accent", "css"},
        "teal": {"teal", "accent", "css"},
        "cyan": {"teal", "accent", "css"},
        "dark": {"theme", "dark", "css"},
        "light": {"theme", "light", "css"},
        "search": {"search", "mcp", "tool"},
        "agent": {"agent", "evolution", "runs", "orchestrator"},
        # Spanish
        "interfaz": {"ui", "web", "app", "tsx", "css", "component"},
        "color": {"color", "accent", "css", "theme"},
        "verde": {"green", "accent", "css"},
        "oscuro": {"theme", "dark", "css"},
        "claro": {"theme", "light", "css"},
        "búsqueda": {"search", "mcp", "tool"},
        "agente": {"agent", "evolution", "runs", "orchestrator"},
        # French
        "interface": {"ui", "web", "app", "tsx", "css", "component"},
        "couleur": {"color", "accent", "css", "theme"},
        "vert": {"green", "accent", "css"},
        "sombre": {"theme", "dark", "css"},
        "clair": {"theme", "light", "css"},
        "recherche": {"search", "mcp", "tool"},
        "agent": {"agent", "evolution", "runs", "orchestrator"},
        # Portuguese
        "interface": {"ui", "web", "app", "tsx", "css", "component"},
        "cor": {"color", "accent", "css", "theme"},
        "verde": {"green", "accent", "css"},
        "escuro": {"theme", "dark", "css"},
        "claro": {"theme", "light", "css"},
        "pesquisa": {"search", "mcp", "tool"},
        "agente": {"agent", "evolution", "runs", "orchestrator"},
        # German
        "oberfläche": {"ui", "web", "app", "tsx", "css", "component"},
        "farbe": {"color", "accent", "css", "theme"},
        "grün": {"green", "accent", "css"},
        "dunkel": {"theme", "dark", "css"},
        "hell": {"theme", "light", "css"},
        "suche": {"search", "mcp", "tool"},
        # Italian
        "interfaccia": {"ui", "web", "app", "tsx", "css", "component"},
        "colore": {"color", "accent", "css", "theme"},
        "verde": {"green", "accent", "css"},
        "scuro": {"theme", "dark", "css"},
        "chiaro": {"theme", "light", "css"},
        "ricerca": {"search", "mcp", "tool"},
        # Polish
        "interfejs": {"ui", "web", "app", "tsx", "css", "component"},
        "kolor": {"color", "accent", "css", "theme"},
        "zielon": {"green", "accent", "css"},
        "ciemn": {"theme", "dark", "css"},
        "jasn": {"theme", "light", "css"},
        "wyszukiw": {"search", "mcp", "tool"},
        # Dutch
        "interface": {"ui", "web", "app", "tsx", "css", "component"},
        "kleur": {"color", "accent", "css", "theme"},
        "groen": {"green", "accent", "css"},
        "donker": {"theme", "dark", "css"},
        "licht": {"theme", "light", "css"},
        "zoeken": {"search", "mcp", "tool"},
        # Turkish
        "arayüz": {"ui", "web", "app", "tsx", "css", "component"},
        "renk": {"color", "accent", "css", "theme"},
        "yeşil": {"green", "accent", "css"},
        "koyu": {"theme", "dark", "css"},
        "açık": {"theme", "light", "css"},
        "arama": {"search", "mcp", "tool"},
        # Japanese
        "インターフェース": {"ui", "web", "app", "tsx", "css", "component"},
        "色": {"color", "accent", "css", "theme"},
        "緑": {"green", "accent", "css"},
        "ダーク": {"theme", "dark", "css"},
        "ライト": {"theme", "light", "css"},
        "検索": {"search", "mcp", "tool"},
        # Korean
        "인터페이스": {"ui", "web", "app", "tsx", "css", "component"},
        "색상": {"color", "accent", "css", "theme"},
        "녹색": {"green", "accent", "css"},
        "다크": {"theme", "dark", "css"},
        "라이트": {"theme", "light", "css"},
        "검색": {"search", "mcp", "tool"},
        # Chinese
        "界面": {"ui", "web", "app", "tsx", "css", "component"},
        "颜色": {"color", "accent", "css", "theme"},
        "绿色": {"green", "accent", "css"},
        "深色": {"theme", "dark", "css"},
        "浅色": {"theme", "light", "css"},
        "搜索": {"search", "mcp", "tool"},
        # Hindi
        "इंटरफ़ेस": {"ui", "web", "app", "tsx", "css", "component"},
        "रंग": {"color", "accent", "css", "theme"},
        "हरा": {"green", "accent", "css"},
        "खोज": {"search", "mcp", "tool"},
        # Arabic
        "واجهة": {"ui", "web", "app", "tsx", "css", "component"},
        "لون": {"color", "accent", "css", "theme"},
        "أخضر": {"green", "accent", "css"},
        "بحث": {"search", "mcp", "tool"},
        # Ukrainian
        "юай": {"ui", "web", "app", "tsx", "css", "component"},
        "інтерф": {"ui", "web", "app", "tsx", "css", "component"},
        "колір": {"color", "accent", "css", "theme"},
        "кольор": {"color", "accent", "css", "theme"},
        "акцент": {"accent", "css", "theme"},
        "зел": {"green", "accent", "css"},
        "бірюз": {"teal", "accent", "css"},
        "темн": {"theme", "dark", "css"},
        "світл": {"theme", "light", "css"},
        "пошук": {"search", "mcp", "tool"},
        "агент": {"agent", "evolution", "runs", "orchestrator"},
    }
    keywords = set(raw)
    for token in raw:
        for prefix, values in synonyms.items():
            if token.startswith(prefix):
                keywords.update(values)
    return {keyword for keyword in keywords if len(keyword) >= 3}


def context_relevance_score(path: Path, rel_text: str, keywords: set[str]) -> int:
    lower_path = rel_text.lower()
    score = 0
    for keyword in keywords:
        if keyword in lower_path:
            score += 6
    if path.suffix.lower() in {".css", ".tsx", ".ts", ".py"}:
        score += 1
    try:
        sample = path.read_text(encoding="utf-8")[:20000].lower()
    except Exception:
        return score
    for keyword in keywords:
        if keyword in sample:
            score += min(sample.count(keyword), 5)
    return score


def is_context_file(path: Path) -> bool:
    if path.name in {"Dockerfile", "Makefile"}:
        return True
    if path.suffix.lower() not in TEXT_CONTEXT_EXTENSIONS:
        return False
    try:
        return path.stat().st_size <= 250_000
    except OSError:
        return False


def should_ignore_context_path(rel: Path) -> bool:
    parts = rel.parts
    if any(part in ANYWHERE_IGNORES or part in TOP_LEVEL_IGNORES for part in parts):
        return True
    if parts and parts[0] in {".codex", ".playwright-mcp"}:
        return True
    return False


def language_for_path(path: str) -> str:
    suffix = Path(path).suffix.lower().lstrip(".")
    return {"py": "python", "ts": "typescript", "tsx": "tsx", "js": "javascript", "md": "markdown"}.get(suffix, suffix)


def extract_json_object(raw: str) -> Any:
    text = (raw or "").strip()
    if not text:
        return None

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    candidates = [fence.group(1).strip()] if fence else []
    candidates.append(text)

    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            pass
        for index, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(candidate[index:])
                return parsed
            except Exception:
                continue
    return None


def write_model_file_edits(*, child_repo: Path, edit_plan: dict[str, Any]) -> list[str]:
    changed: list[str] = []
    operations = edit_plan.get("operations")
    if operations is None:
        operations = edit_plan.get("replacements")
    if isinstance(operations, list):
        for item in operations[: IMPLEMENTATION_MAX_FILES * 4]:
            if not isinstance(item, dict):
                continue
            rel = str(item.get("path") or "").strip()
            find = item.get("find")
            replace = item.get("replace")
            if not rel or not isinstance(find, str) or not isinstance(replace, str):
                continue
            target = resolve_model_edit_path(child_repo, rel)
            original = read_text_safe(target)
            if not original:
                raise ValueError(f"replacement target does not exist or is empty: {rel}")
            if find not in original:
                raise ValueError(f"replacement text not found in {rel}")
            updated = original.replace(find, replace, 1)
            if updated != original:
                target.write_text(updated, encoding="utf-8")
                changed.append(relative_to_repo(target, child_repo))

    files = edit_plan.get("files")
    if files is None:
        files = edit_plan.get("edits")
    if not isinstance(files, list):
        files = []

    for item in files[:IMPLEMENTATION_MAX_FILES]:
        if not isinstance(item, dict):
            continue
        rel = str(item.get("path") or "").strip()
        if not rel:
            continue
        target = resolve_model_edit_path(child_repo, rel)
        content = item.get("content")
        if item.get("delete") is True:
            if target.exists():
                target.unlink()
                changed.append(relative_to_repo(target, child_repo))
            continue
        if not isinstance(content, str):
            continue
        if target.exists() and read_text_safe(target) == content:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        changed.append(relative_to_repo(target, child_repo))

    delete_paths = edit_plan.get("delete_paths")
    if isinstance(delete_paths, list):
        for rel_raw in delete_paths[:IMPLEMENTATION_MAX_FILES]:
            rel = str(rel_raw or "").strip()
            if not rel:
                continue
            target = resolve_model_edit_path(child_repo, rel)
            if target.exists():
                target.unlink()
                changed.append(relative_to_repo(target, child_repo))

    return sorted(set(changed))


def resolve_model_edit_path(child_repo: Path, rel_path: str) -> Path:
    if Path(rel_path).is_absolute():
        raise ValueError(f"absolute paths are not allowed: {rel_path}")
    if rel_path.startswith("../") or "/../" in rel_path or rel_path == "..":
        raise ValueError(f"parent directory traversal is not allowed: {rel_path}")
    target = (child_repo / rel_path).resolve()
    assert_child_write_scope(target, child_repo)
    return target


def update_changes_artifact(artifact_dir: Path, *, prompt: str, implementation: dict[str, Any]) -> None:
    selected_task = str(implementation.get("selected_task") or "").strip()
    lines = [
        "# Changes",
        "",
        f"- Requested task: {prompt}",
        f"- Selected task: {selected_task}" if selected_task and selected_task != prompt else "",
        f"- Implementation worker: {implementation.get('worker', 'unknown')}",
        f"- Result: {implementation.get('summary', '')}",
    ]
    lines = [line for line in lines if line != ""]
    for path in implementation.get("files_changed", []):
        lines.append(f"- Changed: {path}")
    remaining = implementation.get("task_plan", {}).get("remaining_tasks")
    if isinstance(remaining, list) and remaining:
        lines.append("")
        lines.append("## Remaining Task Ideas")
        for item in remaining[:8]:
            lines.append(f"- {item}")
    lines.append("")
    (artifact_dir / "changes.md").write_text("\n".join(lines), encoding="utf-8")


def read_text_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def relative_to_repo(path: Path, repo: Path) -> str:
    try:
        return path.relative_to(repo).as_posix()
    except ValueError:
        return str(path)


SOURCE_DIFF_IGNORES = {
    *ANYWHERE_IGNORES,
    *TOP_LEVEL_IGNORES,
    ".playwright-mcp",
    ".venv",
    "dist",
    "EVOLUTION.md",
}


def detect_substantive_changes(*, parent_repo: Path, child_repo: Path) -> dict[str, Any]:
    parent_files = source_file_hashes(parent_repo)
    child_files = source_file_hashes(child_repo)
    added = sorted(path for path in child_files if path not in parent_files)
    deleted = sorted(path for path in parent_files if path not in child_files)
    modified = sorted(path for path in child_files if path in parent_files and child_files[path] != parent_files[path])
    changed = added + modified + deleted
    ok = bool(changed)
    return {
        "ok": ok,
        "summary": (
            f"{len(changed)} substantive source file change(s): "
            f"{len(added)} added, {len(modified)} modified, {len(deleted)} deleted."
            if ok
            else "No substantive source files changed; scaffold-only generations are not valid improvements."
        ),
        "added": added[:200],
        "modified": modified[:200],
        "deleted": deleted[:200],
        "changed_count": len(changed),
    }


def source_file_hashes(root: Path) -> dict[str, str]:
    import hashlib

    base = root.resolve()
    hashes: dict[str, str] = {}
    if not base.exists():
        return hashes
    for path in base.rglob("*"):
        rel = path.relative_to(base)
        if should_ignore_source_path(rel):
            if path.is_dir():
                continue
            continue
        if not path.is_file():
            continue
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            continue
        hashes[rel.as_posix()] = digest
    return hashes


def should_ignore_source_path(rel: Path) -> bool:
    parts = rel.parts
    if not parts:
        return True
    if any(part in SOURCE_DIFF_IGNORES for part in parts):
        return True
    if parts[-1] in {".DS_Store"}:
        return True
    return False


def assert_child_write_scope(path: Path, child_repo: Path) -> bool:
    try:
        resolved = path.resolve()
        child = child_repo.resolve()
    except FileNotFoundError:
        resolved = path.absolute()
        child = child_repo.resolve()
    if not is_relative_to(resolved, child):
        raise ValueError(f"Path is outside child write scope: {path}")
    return True


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def run_validation_tests(*, child_repo: Path, parent_repo: Path, enabled: bool = True) -> dict[str, Any]:
    if not enabled:
        return {"ok": False, "skipped": True, "reason": "No substantive source changes to validate.", "results": []}

    commands = validation_commands(child_repo=child_repo, parent_repo=parent_repo)
    results = []
    for label, command, timeout_sec, cwd in commands:
        results.append(
            run_command(
                label=label,
                command=command,
                cwd=cwd,
                timeout_sec=timeout_sec,
            )
        )
    return {
        "ok": all(result.ok for result in results),
        "skipped": False,
        "results": [command_result_to_dict(result) for result in results],
    }


def validation_commands(*, child_repo: Path, parent_repo: Path) -> list[tuple[str, list[str], int, Path]]:
    commands: list[tuple[str, list[str], int, Path]] = []
    if (child_repo / "apps" / "api" / "tests").exists():
        commands.append(
            (
                "api pytest",
                [python_executable(child_repo, parent_repo), "-m", "pytest", "apps/api/tests"],
                240,
                child_repo,
            )
        )
    if (child_repo / "apps" / "web" / "package.json").exists() and (child_repo / "package.json").exists():
        commands.append(("web build", ["npm", "--workspace", "apps/web", "run", "build"], 300, child_repo))
    if (child_repo / "mcp" / "web-search-mcp" / "package.json").exists():
        mcp_cwd = child_repo / "mcp" / "web-search-mcp"
        if not (mcp_cwd / "node_modules").exists():
            commands.append(
                (
                    "web-search mcp install",
                    ["npm", "install"],
                    300,
                    mcp_cwd,
                )
            )
        commands.append(
            (
                "web-search mcp build",
                ["npm", "run", "build"],
                180,
                mcp_cwd,
            )
        )
    return commands


def python_executable(child_repo: Path, parent_repo: Path) -> str:
    candidates = [
        child_repo / "apps" / "api" / ".venv" / "bin" / "python",
        parent_repo / "apps" / "api" / ".venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def run_command(
    *,
    label: str,
    command: list[str],
    cwd: Path,
    timeout_sec: int,
    env: dict[str, str] | None = None,
) -> CommandResult:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
        return CommandResult(
            label=label,
            command=command,
            cwd=str(cwd),
            ok=completed.returncode == 0,
            returncode=completed.returncode,
            stdout=completed.stdout[-12000:],
            stderr=completed.stderr[-12000:],
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            label=label,
            command=command,
            cwd=str(cwd),
            ok=False,
            returncode=None,
            stdout=(exc.stdout or "")[-12000:] if isinstance(exc.stdout, str) else "",
            stderr=(exc.stderr or "")[-12000:] if isinstance(exc.stderr, str) else "",
            error=f"Timed out after {timeout_sec}s",
        )
    except Exception as exc:
        return CommandResult(
            label=label,
            command=command,
            cwd=str(cwd),
            ok=False,
            returncode=None,
            error=str(exc),
        )


def command_result_to_dict(result: CommandResult) -> dict[str, Any]:
    return {
        "label": result.label,
        "command": result.command,
        "cwd": result.cwd,
        "ok": result.ok,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "error": result.error,
    }


def _compact_command_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "label": item.get("label"),
            "ok": item.get("ok"),
            "returncode": item.get("returncode"),
            "error": item.get("error"),
        }
        for item in results
    ]


def build_final_report(
    *,
    generation: int,
    prompt: str,
    mode: str,
    child_repo: Path,
    implementation: dict[str, Any],
    self_test: dict[str, Any],
    change_check: dict[str, Any],
    test_results: dict[str, Any],
    ok: bool,
) -> str:
    lines = [
        f"# agent-{generation:03d} Final Report",
        "",
        f"Status: {'passed' if ok else 'failed'}",
        f"Mode: {mode}",
        f"Child repo: `{child_repo}`",
        "",
        "## Requested Improvement",
        "",
        prompt,
        "",
        "## Selected Task",
        "",
        str(implementation.get("selected_task") or prompt),
        "",
        "## Implementation Worker",
        "",
        f"- {'passed' if implementation.get('ok') else 'failed'}: {implementation.get('summary', '')}",
        f"- worker: {implementation.get('worker', 'unknown')}",
        "",
        "## Self-Test",
        "",
    ]
    for check in self_test.get("checks", []):
        lines.append(f"- {'passed' if check.get('ok') else 'failed'}: {check.get('name')}")
    lines.extend(["", "## Implementation Changes", ""])
    lines.append(f"- {'passed' if change_check.get('ok') else 'failed'}: {change_check.get('summary', '')}")
    changed_files = list(change_check.get("added", [])) + list(change_check.get("modified", [])) + list(change_check.get("deleted", []))
    for path in changed_files[:20]:
        lines.append(f"  - {path}")
    lines.extend(["", "## Validation", ""])
    for result in test_results.get("results", []):
        lines.append(f"- {'passed' if result.get('ok') else 'failed'}: {result.get('label')}")
    if test_results.get("skipped"):
        lines.append("- skipped: validation disabled")
    return "\n".join(lines) + "\n"


def write_progress(run_id: str | None, stage: str, generation: int, child_repo: Path) -> None:
    if not run_id:
        return
    now = utcnow_iso()
    execute(
        """
        UPDATE evolution_runs
        SET current_generation=?, child_generation=?, child_repo_path=?, updated_at=?, progress_json=?
        WHERE id=?
        """,
        (
            generation,
            generation,
            str(child_repo),
            now,
            json.dumps({"stage": stage, "generation": generation, "child_repo": str(child_repo)}, ensure_ascii=False),
            run_id,
        ),
    )


def _record_event(
    *,
    run_id: str | None,
    event_type: str,
    title: str,
    detail: str = "",
    payload: dict[str, Any] | None = None,
    generation: int | None = None,
) -> None:
    if not run_id:
        return
    execute(
        """
        INSERT INTO evolution_events (
          id, run_id, generation, event_type, title, detail, payload_json, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            run_id,
            generation,
            event_type,
            title,
            detail,
            json.dumps(payload or {}, ensure_ascii=False),
            utcnow_iso(),
        ),
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_lineage_event(lineage_root: Path, payload: dict[str, Any]) -> None:
    lineage_root.mkdir(parents=True, exist_ok=True)
    with (lineage_root / "lineage.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
