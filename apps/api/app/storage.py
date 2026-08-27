from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .config import settings


SESSION_SUBDIRS = [
    "checkpoints",
    "memory",
    "artifacts",
    "workspace",
    "exports",
]

MEMORY_FILES_DEFAULTS: dict[str, Any] = {
    "durable_facts.json": [],
    "working_set.json": {
        "current_objective": "",
        "current_subtask": "",
        "last_completed_step": "",
        "next_suggested_step": "",
        "open_loops": [],
        "active_files": [],
        "active_tools": [],
        "recent_blockers": [],
    },
    "decisions.json": [],
    "tasks.json": [],
    "entities.json": [],
    "retrieval_anchors.json": [],
}


def data_root() -> Path:
    return Path(settings.data_dir)


def sessions_root() -> Path:
    return data_root() / "sessions"


def session_root(session_id: str) -> Path:
    return sessions_root() / session_id


def ensure_session_dirs(session_id: str) -> Path:
    root = session_root(session_id)
    root.mkdir(parents=True, exist_ok=True)
    for sub in SESSION_SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)
    for filename, default_payload in MEMORY_FILES_DEFAULTS.items():
        fpath = root / "memory" / filename
        if not fpath.exists():
            atomic_json_write(fpath, default_payload)
    session_json = root / "session.json"
    if not session_json.exists():
        atomic_json_write(session_json, {"id": session_id})
    transcript = root / "transcript.jsonl"
    if not transcript.exists():
        transcript.write_text("", encoding="utf-8")
    return root


def atomic_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def write_session_json(session_id: str, payload: dict[str, Any]) -> None:
    root = ensure_session_dirs(session_id)
    atomic_json_write(root / "session.json", payload)


def append_transcript_event(session_id: str, event: dict[str, Any]) -> None:
    root = ensure_session_dirs(session_id)
    line = json.dumps(event, ensure_ascii=False)
    transcript = root / "transcript.jsonl"
    with transcript.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_checkpoint_file(session_id: str, checkpoint_index: int, checkpoint_payload: dict[str, Any]) -> str:
    root = ensure_session_dirs(session_id)
    name = f"checkpoint_{checkpoint_index:04d}.json"
    path = root / "checkpoints" / name
    atomic_json_write(path, checkpoint_payload)
    return str(path)


def read_memory_file(session_id: str, filename: str) -> Any:
    root = ensure_session_dirs(session_id)
    path = root / "memory" / filename
    if not path.exists():
        return MEMORY_FILES_DEFAULTS.get(filename, {})
    return json.loads(path.read_text(encoding="utf-8"))


def write_memory_file(session_id: str, filename: str, payload: Any) -> None:
    root = ensure_session_dirs(session_id)
    atomic_json_write(root / "memory" / filename, payload)


def list_artifacts(session_id: str) -> list[str]:
    root = ensure_session_dirs(session_id) / "artifacts"
    result: list[str] = []
    for p in root.rglob("*"):
        if p.is_file():
            result.append(str(p.relative_to(root)))
    return sorted(result)


def list_workspace_files(session_id: str) -> list[str]:
    root = ensure_session_dirs(session_id) / "workspace"
    result: list[str] = []
    for p in root.rglob("*"):
        if p.is_file():
            result.append(str(p.relative_to(root)))
    return sorted(result)


def export_session_archive(session_id: str) -> str:
    import tarfile

    root = ensure_session_dirs(session_id)
    export_path = root / "exports" / f"{session_id}.tar.gz"
    with tarfile.open(export_path, "w:gz") as tar:
        tar.add(root, arcname=session_id)
    return str(export_path)


def delete_session_dir(session_id: str) -> None:
    root = sessions_root().resolve()
    target = session_root(session_id).resolve()
    if root not in target.parents:
        return
    if target.exists() and target.is_dir():
        shutil.rmtree(target)
