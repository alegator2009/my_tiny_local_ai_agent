from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import load_app_config, save_app_config
from app.db import fetch_all, fetch_one
from app.main import create_app
from app.services.orchestrator import _save_message
from app.services.runs import (
    _build_research_search_queries,
    _build_deterministic_final_answer,
    _build_step_search_query,
    _build_verification_commands,
    _extract_step_facts,
    _maybe_rollover_for_background_run,
    _record_run_event,
    _step_needs_web_tools,
    _write_run_text_artifact,
)
from app.services.sessions import get_last_window


def test_background_run_lifecycle_without_worker(isolated_data_dir):
    app = create_app()
    client = TestClient(app)

    created = client.post("/api/sessions", json={"title": "Runs Session", "workspace_path": "/tmp/workspace"})
    assert created.status_code == 200
    session_id = created.json()["id"]

    started = client.post(f"/api/runs/{session_id}", json={"content": "Build the project skeleton"})
    assert started.status_code == 200
    run = started.json()
    assert run["status"] == "queued"
    assert run["task_text"] == "Build the project skeleton"

    listed = client.get(f"/api/runs/{session_id}")
    assert listed.status_code == 200
    items = listed.json()
    assert len(items) >= 1
    assert items[0]["id"] == run["id"]

    canceled = client.post(f"/api/runs/{session_id}/{run['id']}/cancel")
    assert canceled.status_code == 200
    assert canceled.json()["status"] == "canceled"


def test_run_events_and_artifacts_are_queryable(isolated_data_dir):
    app = create_app()
    client = TestClient(app)

    created = client.post("/api/sessions", json={"title": "Workflow Session", "workspace_path": "/tmp/workspace"})
    assert created.status_code == 200
    session_id = created.json()["id"]

    started = client.post(f"/api/runs/{session_id}", json={"content": "Validate workflow audit trail"})
    assert started.status_code == 200
    run_id = started.json()["id"]

    event = _record_run_event(
        session_id=session_id,
        run_id=run_id,
        event_type="plan_ready",
        title="Plan ready",
        detail="Structured plan created.",
        payload={"workflow_type": "test"},
    )
    artifact = _write_run_text_artifact(
        session_id=session_id,
        run_id=run_id,
        stage="plan",
        title="Plan",
        relative_path="plan.json",
        content='{"steps":[]}',
    )

    events = client.get(f"/api/runs/{session_id}/{run_id}/events")
    assert events.status_code == 200
    assert events.json()[0]["id"] == event["id"]
    assert events.json()[0]["payload_json"]["workflow_type"] == "test"

    artifacts = client.get(f"/api/runs/{session_id}/{run_id}/artifacts")
    assert artifacts.status_code == 200
    assert artifacts.json()[0]["id"] == artifact["id"]
    assert artifacts.json()[0]["path"].endswith("/plan.json")


def test_research_step_searches_global_task_not_workflow_instruction():
    task = "Find the best cakes in Dnipro according to customer reviews"
    researcher_step = {
        "title": "Data Collection via Search Engine",
        "instruction": "Execute the formulated queries across major Ukrainian search engines.",
        "worker_type": "researcher",
    }
    synthesizer_step = {
        "title": "Final Recommendation Synthesis",
        "instruction": "Synthesize the findings into a user-friendly recommendation.",
        "worker_type": "synthesizer",
    }

    assert _step_needs_web_tools(researcher_step)
    assert not _step_needs_web_tools(synthesizer_step)
    assert _build_step_search_query(task, researcher_step) == task
    assert _build_step_search_query(task, researcher_step, attempt=2) == f"{task} rating reviews"


def test_research_retry_queries_include_local_review_sources():
    task = "Find the best cakes in Dnipro according to customer reviews"
    queries = _build_research_search_queries(task)

    assert queries[0] == task
    assert any(query.startswith("site:top20.ua ") for query in queries)
    assert any(query.startswith("site:ratelist.top ") for query in queries)
    assert any(query.startswith("site:tomato.ua ") for query in queries)


def test_verifier_uses_api_venv_python_for_monorepo(isolated_data_dir):
    workspace = isolated_data_dir / "workspace"
    (workspace / "apps" / "api" / ".venv" / "bin").mkdir(parents=True)
    (workspace / "apps" / "api" / ".venv" / "bin" / "python").write_text("", encoding="utf-8")
    (workspace / "apps" / "api" / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")

    commands = _build_verification_commands({"workspace_path": str(workspace)})

    assert commands[0]["command"] == "apps/api/.venv/bin/python -m pytest apps/api/tests"


def test_extract_step_facts_reads_web_search_results():
    output = """Search completed with 1 result:

**1. Pastry shops in Dnipro - 34 bakeries and 1400 reviews | TOP 20**
URL: https://top20.ua/dp/tag/konditerskaya/
"""

    extraction = _extract_step_facts(output, [])

    assert extraction["facts"][0]["source"] == "web_search_result"
    assert "Pastry shops in Dnipro" in extraction["facts"][0]["claim"]
    assert extraction["facts"][1]["claim"].startswith("Source URL:")


def test_deterministic_final_answer_uses_extracted_facts():
    final = _build_deterministic_final_answer(
        "Find cakes",
        [
            {
                "output": "Tool executed, but provider rejected the follow-up response request.",
                "extraction": {
                    "facts": [
                        {"claim": "TORTS.UA pastry chain", "source": "web_search_result"},
                    ]
                },
            }
        ],
        {"status": "passed", "results": [{"label": "api tests", "ok": True}]},
    )

    assert "TORTS.UA" in final
    assert "api tests: passed" in final


def test_background_rollover_creates_new_window_when_limit_exceeded(isolated_data_dir):
    app = create_app()
    client = TestClient(app)

    cfg = load_app_config()
    cfg.model_context_window_size_override = 40
    cfg.rollover_config.pre_rollover_threshold = 0.5
    cfg.rollover_config.hard_rollover_threshold = 0.6
    save_app_config(cfg)

    created = client.post("/api/sessions", json={"title": "Rollover Session", "workspace_path": "/tmp/workspace"})
    assert created.status_code == 200
    session_id = created.json()["id"]

    window = get_last_window(session_id)
    assert window is not None
    original_window_id = window["id"]

    _save_message(
        session_id=session_id,
        window_id=original_window_id,
        role="assistant",
        content_text="token " * 200,
        message_type="assistant",
        turn_id="test-rollover",
        source="run",
    )

    active_window_id = _maybe_rollover_for_background_run(
        session_id=session_id,
        window_id=original_window_id,
        run_id="test-run",
    )

    assert active_window_id != original_window_id
    latest_window = get_last_window(session_id)
    assert latest_window is not None
    assert latest_window["id"] == active_window_id

    closed_source_window = fetch_one("SELECT closed_at FROM windows WHERE id=?", (original_window_id,))
    assert closed_source_window is not None
    assert closed_source_window["closed_at"] is not None

    checkpoints = fetch_all("SELECT id FROM checkpoints WHERE session_id=?", (session_id,))
    assert len(checkpoints) >= 1
