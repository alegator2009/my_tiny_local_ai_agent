from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from app.services.artifacts import write_file_artifact
from app.services.sessions import get_session


def test_write_file_artifact_and_download(isolated_data_dir):
    app = create_app()
    client = TestClient(app)

    workspace_root = isolated_data_dir / "workspace"
    created = client.post(
        "/api/sessions",
        json={"title": "Artifacts Session", "workspace_path": str(workspace_root)},
    )
    assert created.status_code == 200
    session_id = created.json()["id"]

    session_info = get_session(session_id)
    result = write_file_artifact(
        session_id=session_id,
        session_info=session_info,
        args={"path": "pages/frogs_page.html", "content": "<h1>Frogs</h1>"},
    )

    assert result["ok"] is True
    artifact = result["artifact"]
    assert artifact["file_name"] == "frogs_page.html"

    written_file = workspace_root / "pages" / "frogs_page.html"
    assert written_file.exists()
    assert written_file.read_text(encoding="utf-8") == "<h1>Frogs</h1>"

    download = client.get(artifact["download_url"])
    assert download.status_code == 200
    assert download.text == "<h1>Frogs</h1>"


def test_write_file_artifact_rejects_path_escape(isolated_data_dir):
    app = create_app()
    client = TestClient(app)

    created = client.post(
        "/api/sessions",
        json={"title": "Artifacts Session", "workspace_path": str(isolated_data_dir / "workspace")},
    )
    assert created.status_code == 200
    session_id = created.json()["id"]

    session_info = get_session(session_id)
    result = write_file_artifact(
        session_id=session_id,
        session_info=session_info,
        args={"path": "../escape.txt", "content": "nope"},
    )

    assert result["ok"] is False
    assert "escape" in result["error"].lower()
    assert not (Path(session_info["workspace_path"]).parent / "escape.txt").exists()
