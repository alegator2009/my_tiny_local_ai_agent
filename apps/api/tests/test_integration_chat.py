from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app


def test_stream_chat_roundtrip(isolated_data_dir):
    app = create_app()
    client = TestClient(app)

    created = client.post(
        "/api/sessions",
        json={"title": "Integration Session", "workspace_path": "/tmp/workspace"},
    )
    assert created.status_code == 200
    session_id = created.json()["id"]

    response = client.post(
        f"/api/chat/{session_id}/stream",
        json={"content": "Hi, give a short answer"},
    )
    assert response.status_code == 200
    body = response.text
    assert "event: final_message" in body

    transcript = client.get(f"/api/chat/{session_id}/transcript")
    assert transcript.status_code == 200
    data = transcript.json()
    assert len(data) >= 2
