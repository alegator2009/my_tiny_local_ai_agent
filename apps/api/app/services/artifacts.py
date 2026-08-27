from __future__ import annotations

import base64
import hashlib
import mimetypes
import uuid
from pathlib import Path
from typing import Any

from ..db import execute, fetch_one, utcnow_iso
from ..storage import ensure_session_dirs
from .workspace import log_workspace_event

MAX_ARTIFACT_BYTES = 5 * 1024 * 1024


def _workspace_root(session_id: str, session_info: dict[str, Any]) -> Path:
    workspace_path = str(session_info.get("workspace_path") or "").strip()
    if workspace_path:
        root = Path(workspace_path).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root
    root = ensure_session_dirs(session_id) / "workspace"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _safe_relative_path(raw_path: str) -> Path:
    normalized = raw_path.strip().replace("\\", "/")
    if not normalized:
        raise ValueError("path is required")

    candidate = Path(normalized)
    if candidate.is_absolute():
        raise ValueError("path must be relative")
    if any(part == ".." for part in candidate.parts):
        raise ValueError("path must not escape workspace")
    if candidate.name in {"", "."}:
        raise ValueError("path must include file name")
    return candidate


def _decode_content(args: dict[str, Any]) -> bytes:
    if args.get("content") is not None and args.get("content_base64") is not None:
        raise ValueError("provide either content or content_base64, not both")

    content = args.get("content")
    if content is not None:
        if not isinstance(content, str):
            raise ValueError("content must be a string")
        encoding = str(args.get("encoding") or "utf-8").strip() or "utf-8"
        try:
            payload = content.encode(encoding)
        except LookupError as exc:
            raise ValueError(f"unknown encoding: {encoding}") from exc
    else:
        content_base64 = args.get("content_base64")
        if content_base64 is None:
            raise ValueError("content or content_base64 is required")
        if not isinstance(content_base64, str):
            raise ValueError("content_base64 must be a string")
        try:
            payload = base64.b64decode(content_base64, validate=True)
        except Exception as exc:
            raise ValueError("content_base64 is not valid base64") from exc

    if len(payload) > MAX_ARTIFACT_BYTES:
        raise ValueError(f"file too large: max {MAX_ARTIFACT_BYTES} bytes")
    return payload


def _artifact_media_type(download_name: str, explicit: Any) -> str:
    explicit_mime = str(explicit or "").strip()
    if explicit_mime:
        return explicit_mime
    guessed, _ = mimetypes.guess_type(download_name)
    return guessed or "application/octet-stream"


def write_file_artifact(session_id: str, session_info: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    try:
        rel_path = _safe_relative_path(str(args.get("path") or ""))
        payload = _decode_content(args)
        overwrite = bool(args.get("overwrite", True))

        workspace_root = _workspace_root(session_id, session_info)
        target_path = (workspace_root / rel_path).resolve()
        if workspace_root != target_path and workspace_root not in target_path.parents:
            raise ValueError("resolved path is outside workspace")

        if target_path.exists() and not overwrite:
            raise ValueError("target file already exists and overwrite=false")

        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(payload)

        session_root = ensure_session_dirs(session_id)
        artifact_id = str(uuid.uuid4())
        download_name = Path(str(args.get("download_name") or rel_path.name)).name or rel_path.name
        artifact_rel_path = Path(artifact_id) / download_name
        artifact_abs_path = (session_root / "artifacts" / artifact_rel_path).resolve()
        artifact_abs_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_abs_path.write_bytes(payload)

        now = utcnow_iso()
        digest = hashlib.sha256(payload).hexdigest()
        media_type = _artifact_media_type(download_name, args.get("mime_type"))

        execute(
            """
            INSERT INTO artifacts (id, session_id, path, artifact_type, title, description, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact_id,
                session_id,
                artifact_rel_path.as_posix(),
                media_type,
                download_name,
                f"Snapshot of {rel_path.as_posix()}",
                now,
                now,
            ),
        )

        workspace_relative = target_path.relative_to(workspace_root).as_posix()
        artifact_meta = {
            "id": artifact_id,
            "file_name": download_name,
            "mime_type": media_type,
            "size_bytes": len(payload),
            "sha256": digest,
            "download_url": f"/api/workspace/{session_id}/artifacts/{artifact_id}/download",
            "workspace_path": workspace_relative,
            "workspace_abs_path": str(target_path),
            "artifact_path": artifact_rel_path.as_posix(),
            "created_at": now,
        }

        log_workspace_event(
            session_id=session_id,
            event_type="artifact_created",
            payload_json=artifact_meta,
            summary_text=f"artifact created: {download_name}",
        )

        return {
            "ok": True,
            "path": workspace_relative,
            "bytes_written": len(payload),
            "artifact": artifact_meta,
            "artifacts": [artifact_meta],
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
        }


def resolve_artifact_download(session_id: str, artifact_id: str) -> dict[str, Any]:
    row = fetch_one(
        "SELECT id, path, artifact_type, title FROM artifacts WHERE id=? AND session_id=?",
        (artifact_id, session_id),
    )
    if row is None:
        raise KeyError("artifact_not_found")

    session_root = ensure_session_dirs(session_id)
    artifacts_root = (session_root / "artifacts").resolve()
    rel_path = Path(row["path"])
    abs_path = (artifacts_root / rel_path).resolve()

    if artifacts_root != abs_path and artifacts_root not in abs_path.parents:
        raise KeyError("artifact_not_found")
    if not abs_path.is_file():
        raise KeyError("artifact_not_found")

    return {
        "id": row["id"],
        "path": str(abs_path),
        "media_type": row["artifact_type"] or "application/octet-stream",
        "filename": row["title"] or abs_path.name,
    }
