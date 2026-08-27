from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app


def test_message_prefix_templates_crud(isolated_data_dir):
    app = create_app()
    client = TestClient(app)

    listed = client.get("/api/message-prefix-templates")
    assert listed.status_code == 200
    assert listed.json() == []

    created = client.post(
        "/api/message-prefix-templates",
        json={"name": "Brief", "prompt": "Reply briefly"},
    )
    assert created.status_code == 200
    created_payload = created.json()
    assert created_payload["name"] == "Brief"
    assert created_payload["prompt"] == "Reply briefly"

    updated = client.post(
        "/api/message-prefix-templates",
        json={"name": "Brief", "prompt": "Reply briefly and to the point"},
    )
    assert updated.status_code == 200
    updated_payload = updated.json()
    assert updated_payload["id"] == created_payload["id"]
    assert updated_payload["prompt"] == "Reply briefly and to the point"

    listed_after = client.get("/api/message-prefix-templates")
    assert listed_after.status_code == 200
    listed_data = listed_after.json()
    assert len(listed_data) == 1
    assert listed_data[0]["name"] == "Brief"

    deleted = client.delete(f"/api/message-prefix-templates/{created_payload['id']}")
    assert deleted.status_code == 200
    assert deleted.json()["ok"] is True

    missing = client.delete(f"/api/message-prefix-templates/{created_payload['id']}")
    assert missing.status_code == 404
