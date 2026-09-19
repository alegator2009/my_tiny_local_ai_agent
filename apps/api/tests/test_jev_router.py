import asyncio

from app.config import AppConfig, TypeSafeConfig
from app.services.jev_router import route_turn
from app.services.typesafe import ChoiceAnswer


def _config() -> AppConfig:
    return AppConfig(
        typesafe_config=TypeSafeConfig(enabled=True, api_key="test-token", min_confidence=0.7)
    )


def test_global_router_can_expose_terminal_for_computer_diagnostics(monkeypatch):
    seen: dict[str, object] = {}

    async def _choices(**kwargs):
        seen.update(kwargs)
        return {
            "terminal": ChoiceAnswer("terminal", 0.93, {"terminal": 0.93, "none": 0.07}),
            "file": ChoiceAnswer("none", 0.98, {"file": 0.02, "none": 0.98}),
            "mcp": ChoiceAnswer("none", 0.91, {"mcp": 0.09, "none": 0.91}),
            "web_search": ChoiceAnswer("skip", 0.81, {"search": 0.19, "skip": 0.81}),
            "skill": ChoiceAnswer("__no_matching_skill__", 0.88, {"__no_matching_skill__": 0.88}),
        }

    monkeypatch.setattr("app.services.jev_router.evaluate_choices", _choices)
    plan = asyncio.run(
        route_turn(
            "Що відомо про цей комп'ютер?",
            cfg=_config(),
            terminal_fallback=False,
            file_fallback=False,
            mcp_fallback=False,
            skill_criteria={"__no_matching_skill__": "No skill applies."},
            include_web_search=True,
        )
    )

    assert plan.terminal_tool_enabled is True
    assert plan.file_tool_enabled is False
    assert plan.mcp_tools_enabled is False
    assert plan.web_search_answer is not None
    assert plan.web_search_answer.choice == "skip"
    assert "terminal" in seen["questions"]
    assert "web_search" in seen["questions"]


def test_global_router_preserves_fallbacks_for_low_confidence(monkeypatch):
    async def _uncertain(**_kwargs):
        return {
            "terminal": ChoiceAnswer("none", 0.51, {"terminal": 0.49, "none": 0.51}),
            "file": ChoiceAnswer("none", 0.51, {"file": 0.49, "none": 0.51}),
            "mcp": ChoiceAnswer("none", 0.51, {"mcp": 0.49, "none": 0.51}),
        }

    monkeypatch.setattr("app.services.jev_router.evaluate_choices", _uncertain)
    plan = asyncio.run(
        route_turn(
            "run diagnostics",
            cfg=_config(),
            terminal_fallback=True,
            file_fallback=True,
            mcp_fallback=True,
        )
    )

    assert plan.terminal_tool_enabled is True
    assert plan.file_tool_enabled is True
    assert plan.mcp_tools_enabled is True
