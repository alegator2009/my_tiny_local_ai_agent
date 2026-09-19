from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app


def test_typesafe_token_is_redacted_and_preserved(isolated_data_dir):
    client = TestClient(create_app())
    payload = client.get("/api/settings").json()
    payload["typesafe_config"] = {
        "enabled": True,
        "api_key": "secret-token",
        "model": "jev-latest",
        "timeout_sec": 3,
        "min_confidence": 0.7,
    }
    saved = client.put("/api/settings", json=payload)
    assert saved.status_code == 200
    assert saved.json()["typesafe_config"]["api_key"] == ""

    # The following save is the shape produced by the settings page after a
    # reload: the redacted token is blank, but it must not erase the secret.
    reloaded = client.get("/api/settings").json()
    assert reloaded["typesafe_config"]["api_key"] == ""
    reloaded["system_prompt"] = "A changed prompt should retain the token."
    assert client.put("/api/settings", json=reloaded).status_code == 200

    from app.config import load_app_config

    assert load_app_config().typesafe_config.api_key == "secret-token"
