"""End-to-end tests for the provider CRUD endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app import config as app_config_module
from app.config import AppConfig
from app.services import config_reloader as reloader_module
from app.services.config_reloader import ConfigReloader
from app.main import create_app


def _client_with_fresh_config(tmp_path):
    target = tmp_path / "config.json"
    target.write_text("{}", encoding="utf-8")
    original_settings_path = app_config_module.settings.app_config_path
    original_reloader_path = reloader_module.default_reloader._path
    app_config_module.settings.app_config_path = str(target)
    reloader_module.default_reloader = ConfigReloader(path=target)
    try:
        app = create_app()
        with TestClient(app) as client:
            yield client
    finally:
        app_config_module.settings.app_config_path = original_settings_path
        reloader_module.default_reloader = ConfigReloader(path=original_reloader_path)


def test_list_providers_initial_empty(tmp_path):
    for _ in _client_with_fresh_config(tmp_path):
        # First GET should produce an empty provider list (we did not
        # seed anything yet).
        resp = _.get("/api/providers")
        assert resp.status_code == 200
        body = resp.json()
        assert body["providers"] == []
        assert body["active_provider_id"] is None
        assert body["active_model_id"] is None
        break


def test_create_provider_with_models(tmp_path):
    for client in _client_with_fresh_config(tmp_path):
        payload = {
            "name": "Local",
            "base_url": "http://localhost:1234/v1",
            "models": [
                {"name": "qwen", "is_default": True},
                {"name": "phi"},
            ],
        }
        resp = client.post("/api/providers", json=payload)
        assert resp.status_code == 200, resp.text
        provider = resp.json()
        assert provider["name"] == "Local"
        assert len(provider["models"]) == 2
        assert provider["models"][0]["is_default"] is True

        # Active selection should auto-promote to the first provider.
        listing = client.get("/api/providers").json()
        assert listing["active_provider_id"] == provider["id"]
        assert listing["active_model_id"] is not None


def test_add_update_delete_model(tmp_path):
    for client in _client_with_fresh_config(tmp_path):
        created = client.post(
            "/api/providers",
            json={"name": "x", "base_url": "http://x", "models": [{"name": "m1"}]},
        ).json()
        provider_id = created["id"]

        # Add a model
        added = client.post(
            f"/api/providers/{provider_id}/models",
            json={"name": "m2", "is_default": True},
        ).json()
        assert added["name"] == "m2"

        # Patch the model
        patched = client.patch(
            f"/api/providers/{provider_id}/models/{added['id']}",
            json={"max_output_tokens": 4096},
        ).json()
        assert patched["max_output_tokens"] == 4096

        # Delete the model
        deleted = client.delete(
            f"/api/providers/{provider_id}/models/{added['id']}"
        )
        assert deleted.status_code == 200
        assert deleted.json()["deleted_model_id"] == added["id"]


def test_activate_model(tmp_path):
    for client in _client_with_fresh_config(tmp_path):
        provider = client.post(
            "/api/providers",
            json={
                "name": "x",
                "base_url": "http://x",
                "models": [{"name": "a"}, {"name": "b"}],
            },
        ).json()
        second = provider["models"][1]
        resp = client.post(
            f"/api/providers/{provider['id']}/models/{second['id']}/activate"
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["active_model_id"] == second["id"]


def test_delete_provider_clears_active(tmp_path):
    for client in _client_with_fresh_config(tmp_path):
        provider = client.post(
            "/api/providers",
            json={"name": "x", "base_url": "http://x", "models": [{"name": "a"}]},
        ).json()
        listing_before = client.get("/api/providers").json()
        assert listing_before["active_provider_id"] == provider["id"]

        resp = client.delete(f"/api/providers/{provider['id']}")
        assert resp.status_code == 200

        listing_after = client.get("/api/providers").json()
        assert listing_after["providers"] == []
        assert listing_after["active_provider_id"] is None