from __future__ import annotations

import tarfile
from pathlib import Path

from ..storage import export_session_archive, sessions_root


def export_session(session_id: str) -> str:
    return export_session_archive(session_id)


def import_session(archive_path: str) -> str:
    src = Path(archive_path)
    if not src.exists():
        raise FileNotFoundError(archive_path)

    root = sessions_root().resolve()
    with tarfile.open(src, "r:gz") as tar:
        for member in tar.getmembers():
            target = (root / member.name).resolve()
            if not str(target).startswith(str(root)):
                raise ValueError("invalid_archive_path")
        tar.extractall(path=sessions_root())
        names = tar.getnames()

    if not names:
        raise ValueError("empty_archive")
    session_id = names[0].split("/")[0]
    return session_id
