from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.evolution import (
    activate_generation,
    apply_implementation_worker,
    assert_child_write_scope,
    build_operation_snippet_context,
    create_evolution_run,
    default_evolution_parent_repo,
    delete_generation,
    copy_generation_to_root,
    get_evolution_run,
    list_evolution_events,
    list_generations,
    parse_child_handoff_stdout,
    perform_generation,
    plan_generation_task,
    write_model_file_edits,
)


def make_minimal_repo(root: Path) -> None:
    (root / "apps" / "api" / "app" / "evolution").mkdir(parents=True)
    (root / "apps" / "api" / "tests").mkdir(parents=True)
    (root / "apps" / "api" / "app" / "evolution" / "__init__.py").write_text("", encoding="utf-8")
    (root / "apps" / "api" / "tests" / "test_placeholder.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    (root / "README.md").write_text("# Repo\n", encoding="utf-8")
    (root / "package.json").write_text('{"scripts":{}}\n', encoding="utf-8")
    (root / "data").mkdir()
    (root / "data" / "secret.db").write_text("ignore", encoding="utf-8")
    (root / "evolution").mkdir()
    (root / "evolution" / "old.txt").write_text("ignore", encoding="utf-8")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "dep.txt").write_text("ignore", encoding="utf-8")


def test_perform_generation_rejects_scaffold_only_child(tmp_path: Path, monkeypatch):
    import app.services.evolution as evolution

    monkeypatch.setattr(evolution, "complete_implementation_json", lambda messages: "")
    parent = tmp_path / "parent"
    lineage = tmp_path / "lineage"
    parent.mkdir()
    make_minimal_repo(parent)

    result = perform_generation(
        parent_repo=parent,
        lineage_root=lineage,
        prompt="Add better validation",
        mode="conservative",
        stop_on_failure=True,
        run_tests=False,
    )

    assert not result.ok
    assert result.generation == 1
    assert result.change_check["ok"] is False
    assert result.test_results["skipped"] is True
    assert result.child_repo.exists()
    assert (lineage / "agent-001" / "meta.json").exists()
    assert (lineage / "agent-001" / "prompt.md").read_text(encoding="utf-8") == "Add better validation"
    assert (lineage / "agent-001" / "final-report.md").exists()
    assert (result.child_repo / "EVOLUTION.md").exists()
    assert not (result.child_repo / "data").exists()
    assert not (result.child_repo / "evolution").exists()
    assert not (result.child_repo / "node_modules").exists()
    assert (lineage / "lineage.jsonl").exists()
    assert (lineage / "active.json").exists()


def test_write_scope_guard_rejects_parent_paths(tmp_path: Path):
    child = tmp_path / "child"
    child.mkdir()

    assert assert_child_write_scope(child / "ok.txt", child)
    with pytest.raises(ValueError):
        assert_child_write_scope(tmp_path / "outside.txt", child)


def test_generation_management_lists_activates_and_deletes(tmp_path: Path, monkeypatch):
    import app.services.evolution as evolution

    monkeypatch.setattr(evolution, "complete_implementation_json", lambda messages: "")
    parent = tmp_path / "parent"
    lineage = tmp_path / "lineage"
    parent.mkdir()
    make_minimal_repo(parent)

    first = perform_generation(
        parent_repo=parent,
        lineage_root=lineage,
        prompt="First",
        mode="tests-only",
        stop_on_failure=True,
        run_tests=False,
    )
    second = perform_generation(
        parent_repo=first.child_repo,
        lineage_root=lineage,
        prompt="Second",
        mode="tests-only",
        stop_on_failure=True,
        run_tests=False,
    )

    generations = list_generations(lineage)
    assert [item["generation"] for item in generations] == [1, 2]
    assert generations[0]["status"] == "failed"
    assert generations[1]["status"] == "failed"
    assert not generations[1]["active"]

    activated = activate_generation(1, lineage)
    assert activated["active"]
    with pytest.raises(ValueError):
        delete_generation(1, lineage_root=lineage)

    deleted = delete_generation(1, force=True, lineage_root=lineage)
    assert deleted["ok"]
    assert deleted["active_generation"] is None
    assert not (lineage / "agent-001").exists()
    assert (lineage / "agent-002").exists()
    assert second.generation == 2


def test_copy_generation_to_root_syncs_project_files_and_preserves_runtime(tmp_path: Path):
    root = tmp_path / "root"
    lineage = root / "evolution"
    source = lineage / "agent-001" / "repo"
    root.mkdir()
    source.mkdir(parents=True)

    (root / "apps" / "api" / ".venv" / "bin").mkdir(parents=True)
    (root / "apps" / "api" / ".venv" / "bin" / "python").write_text("runtime", encoding="utf-8")
    (root / "apps" / "web").mkdir(parents=True)
    (root / "apps" / "web" / "old.ts").write_text("old", encoding="utf-8")
    (root / "data").mkdir()
    (root / "data" / "state.db").write_text("keep", encoding="utf-8")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "dep.txt").write_text("keep", encoding="utf-8")
    (root / "README.md").write_text("# Old\n", encoding="utf-8")

    (source / "apps" / "web").mkdir(parents=True)
    (source / "apps" / "web" / "new.ts").write_text("new", encoding="utf-8")
    (source / "README.md").write_text("# New\n", encoding="utf-8")
    (source / "data").mkdir()
    (source / "data" / "wrong.db").write_text("do not copy", encoding="utf-8")

    result = copy_generation_to_root(1, lineage_root=lineage, root_repo=root)

    assert result["ok"] is True
    assert (root / "README.md").read_text(encoding="utf-8") == "# New\n"
    assert (root / "apps" / "web" / "new.ts").read_text(encoding="utf-8") == "new"
    assert not (root / "apps" / "web" / "old.ts").exists()
    assert (root / "apps" / "api" / ".venv" / "bin" / "python").read_text(encoding="utf-8") == "runtime"
    assert (root / "data" / "state.db").read_text(encoding="utf-8") == "keep"
    assert not (root / "data" / "wrong.db").exists()
    assert (root / "node_modules" / "dep.txt").read_text(encoding="utf-8") == "keep"
    assert '"type": "copy_to_root"' in (lineage / "lineage.jsonl").read_text(encoding="utf-8")


def test_create_evolution_run_defaults_to_active_generation_parent(tmp_path: Path, monkeypatch):
    import app.services.evolution as evolution

    root = tmp_path / "root"
    active_repo = tmp_path / "lineage" / "agent-007" / "repo"
    lineage = tmp_path / "lineage"
    root.mkdir()
    active_repo.mkdir(parents=True)
    make_minimal_repo(root)
    make_minimal_repo(active_repo)
    (lineage / "active.json").write_text(
        '{"active_generation": 7, "child_repo": "' + str(active_repo).replace("\\", "\\\\") + '"}',
        encoding="utf-8",
    )

    monkeypatch.setattr(evolution, "repo_root", lambda: root)

    assert default_evolution_parent_repo(lineage) == active_repo

    run = create_evolution_run(
        prompt="Next slice",
        max_generations=1,
        mode="conservative",
        stop_on_failure=True,
        lineage_root=lineage,
    )

    loaded = get_evolution_run(run["id"])
    assert Path(loaded["parent_repo_path"]) == active_repo
    assert loaded["parent_generation"] == 7


def test_model_project_worker_applies_generic_file_edits(tmp_path: Path, monkeypatch):
    import app.services.evolution as evolution

    child = tmp_path / "child"
    child.mkdir()
    (child / "README.md").write_text("# Repo\n", encoding="utf-8")

    responses = iter(
        [
            '{"selected_task":"Update README with one concise project note","reason":"small slice","remaining_tasks":["Improve layout"]}',
            (
                '{"summary":"Updated README","files":[{"path":"README.md",'
                '"content":"# Repo\\n\\nGeneric project edit.\\n"}],"notes":[]}'
            ),
        ]
    )
    monkeypatch.setattr(
        evolution,
        "complete_implementation_json",
        lambda messages: next(responses),
    )

    result = apply_implementation_worker(child_repo=child, prompt="Update the README", mode="conservative")

    assert result["ok"] is True
    assert result["worker"] == "model-project-agent"
    assert result["selected_task"] == "Update README with one concise project note"
    assert result["files_changed"] == ["README.md"]
    assert "Generic project edit" in (child / "README.md").read_text(encoding="utf-8")


def test_task_planner_turns_broad_prompt_into_selected_slice(monkeypatch):
    import app.services.evolution as evolution

    monkeypatch.setattr(
        evolution,
        "complete_implementation_json",
        lambda messages: (
            '{"selected_task":"Tighten spacing in the main chat layout using existing CSS variables",'
            '"reason":"small safe UI slice",'
            '"likely_files":["apps/web/app/globals.css"],'
            '"acceptance_checks":["web build passes"],'
            '"remaining_tasks":["Improve settings form grouping"]}'
        ),
    )

    plan = plan_generation_task(
        project_context="# Project File Tree\napps/web/app/globals.css",
        prompt="Make the UI compact and clearer",
        mode="conservative",
    )

    assert plan["selected_task"] == "Tighten spacing in the main chat layout using existing CSS variables"
    assert plan["remaining_tasks"] == ["Improve settings form grouping"]


def test_model_project_worker_retries_noop_with_operation_snippets(tmp_path: Path, monkeypatch):
    import app.services.evolution as evolution

    child = tmp_path / "child"
    css = child / "apps" / "web" / "app" / "globals.css"
    css.parent.mkdir(parents=True)
    css.write_text(".layout {\n  grid-template-columns: 280px 1fr 280px;\n  padding: 12px;\n}\n", encoding="utf-8")

    responses = iter(
        [
            (
                '{"selected_task":"Reduce sidebar widths from 280px to 240px",'
                '"likely_files":["apps/web/app/globals.css"],"remaining_tasks":[]}'
            ),
            '{"summary":"No changes","operations":[],"files":[],"notes":[]}',
            (
                '{"summary":"Reduced sidebar widths","operations":[{"path":"apps/web/app/globals.css",'
                '"find":"grid-template-columns: 280px 1fr 280px;",'
                '"replace":"grid-template-columns: 240px 1fr 240px;"}],"files":[],"notes":[]}'
            ),
        ]
    )
    monkeypatch.setattr(evolution, "complete_implementation_json", lambda messages: next(responses))

    result = apply_implementation_worker(
        child_repo=child,
        prompt="Make the UI more compact",
        mode="conservative",
    )

    assert result["ok"] is True
    assert result["files_changed"] == ["apps/web/app/globals.css"]
    assert "240px 1fr 240px" in css.read_text(encoding="utf-8")


def test_operation_snippet_context_focuses_likely_files(tmp_path: Path):
    child = tmp_path / "child"
    css = child / "apps" / "web" / "app" / "globals.css"
    css.parent.mkdir(parents=True)
    css.write_text(".layout {\n  grid-template-columns: 280px 1fr 280px;\n  padding: 12px;\n}\n", encoding="utf-8")

    context = build_operation_snippet_context(
        child_repo=child,
        task="Reduce 280px sidebars to 240px",
        task_plan={"likely_files": ["apps/web/app/globals.css"]},
    )

    assert "apps/web/app/globals.css" in context
    assert "grid-template-columns: 280px 1fr 280px;" in context


def test_model_project_worker_rejects_outside_paths(tmp_path: Path):
    child = tmp_path / "child"
    child.mkdir()
    with pytest.raises(ValueError):
        write_model_file_edits(
            child_repo=child,
            edit_plan={"files": [{"path": "../outside.txt", "content": "bad"}]},
        )


def test_model_project_worker_applies_replace_operations(tmp_path: Path):
    child = tmp_path / "child"
    child.mkdir()
    (child / "style.css").write_text(":root {\n  --accent: #36b7a8;\n}\n", encoding="utf-8")

    changed = write_model_file_edits(
        child_repo=child,
        edit_plan={
            "operations": [
                {
                    "path": "style.css",
                    "find": "--accent: #36b7a8;",
                    "replace": "--accent: #22c55e;",
                }
            ]
        },
    )

    assert changed == ["style.css"]
    assert "--accent: #22c55e;" in (child / "style.css").read_text(encoding="utf-8")


def test_parse_child_handoff_stdout_extracts_created_generations():
    parsed = parse_child_handoff_stdout(
        'noise\n{"ok": true, "created": [{"generation": 4, "child_repo": "/tmp/agent-004/repo", "ok": true}]}'
    )

    assert parsed["ok"] is True
    assert parsed["created"][0]["generation"] == 4


def test_evolution_routes_create_and_list_runs(isolated_data_dir, monkeypatch):
    import app.routes.evolution as evolution_routes

    def fake_run_evolution(run_id: str):
        return create_evolution_run(
            prompt="nested fake",
            max_generations=1,
            mode="tests-only",
            stop_on_failure=True,
            parent_repo=Path(isolated_data_dir),
            lineage_root=Path(isolated_data_dir) / "fake-lineage",
        )

    monkeypatch.setattr(evolution_routes, "run_evolution", fake_run_evolution)
    app = create_app()
    client = TestClient(app)

    started = client.post(
        "/api/evolution/start",
        json={"prompt": "Improve routing", "max_generations": 1, "mode": "tests-only"},
    )
    assert started.status_code == 200
    run = started.json()
    assert run["status"] == "queued"
    assert run["prompt"] == "Improve routing"

    listed = client.get("/api/evolution/runs")
    assert listed.status_code == 200
    assert any(item["id"] == run["id"] for item in listed.json())

    events = client.get(f"/api/evolution/runs/{run['id']}/events")
    assert events.status_code == 200
    assert events.json()[0]["event_type"] == "queued"
    assert list_evolution_events(run["id"])[0]["event_type"] == "queued"
