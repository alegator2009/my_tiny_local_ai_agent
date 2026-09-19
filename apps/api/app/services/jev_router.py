"""Confidence-gated, pre-turn orchestration with TypeSafe Jev.

Jev supplies semantic judgments; deterministic application code still owns
tool discovery, execution, permissions and fallbacks.  All independent
pre-turn routing questions are sent in one System One request to avoid a
sequence of disconnected classifiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import AppConfig
from .typesafe import ChoiceAnswer, evaluate_choices


@dataclass(frozen=True)
class GlobalRoutePlan:
    terminal_tool_enabled: bool
    file_tool_enabled: bool
    mcp_tools_enabled: bool
    selected_skill: str | None
    skill_decided: bool = False
    web_search_answer: ChoiceAnswer | None = None
    jev_evaluated: bool = False
    decisions: dict[str, ChoiceAnswer] = field(default_factory=dict)


def _accepted(answer: ChoiceAnswer | None, cfg: AppConfig) -> bool:
    return answer is not None and answer.confidence >= cfg.typesafe_config.min_confidence


async def route_turn(
    user_message: str,
    *,
    cfg: AppConfig,
    terminal_fallback: bool,
    file_fallback: bool,
    mcp_fallback: bool,
    skill_fallback: str | None = None,
    skill_criteria: dict[str, str | None] | None = None,
    include_web_search: bool = False,
) -> GlobalRoutePlan:
    """Make the independent tool/skill routing decisions for one chat turn.

    A missing token, network failure, incomplete result or low-confidence
    answer preserves every pre-existing heuristic decision independently.
    """

    fallback = GlobalRoutePlan(
        terminal_tool_enabled=terminal_fallback,
        file_tool_enabled=file_fallback,
        mcp_tools_enabled=mcp_fallback,
        selected_skill=skill_fallback,
    )
    if not cfg.typesafe_config.is_ready():
        return fallback

    questions: dict[str, dict[str, Any]] = {
        "terminal": {
            "instructions": (
                "Decide whether this turn should expose a shell terminal to the chat model. "
                "Choose terminal when answering needs observation or action in the available "
                "container/workspace: computer or OS diagnostics, hardware or resource inspection, "
                "files, processes, network state, commands, builds, tests, or source inspection. "
                "For example, a request asking what is known about this computer needs terminal. "
                "The terminal observes the Docker container and mounted workspace, not the host OS."
            ),
            "criteria": {
                "terminal": "The terminal is materially useful before the assistant answers.",
                "none": "The request can be answered without shell access.",
            },
        },
        "file": {
            "instructions": (
                "Decide whether this turn should expose the file-artifact tool to the chat model. "
                "Choose file only when the user asks to create, edit, save, export, or provide a "
                "downloadable file or artifact."
            ),
            "criteria": {
                "file": "The response needs a file operation or downloadable artifact.",
                "none": "No file operation is needed.",
            },
        },
        "mcp": {
            "instructions": (
                "Decide whether this turn should discover and expose registered MCP tools. "
                "Choose mcp when a registered external capability could directly help, such as web "
                "lookup, a requested skill, or a connected tool. Do not choose it for ordinary chat "
                "or requests already handled by the terminal or file tool alone."
            ),
            "criteria": {
                "mcp": "A registered MCP capability is materially useful for this turn.",
                "none": "No registered MCP capability is needed.",
            },
        },
    }
    if include_web_search:
        questions["web_search"] = {
            "instructions": (
                "Before the chat model responds, decide whether this user message needs a web search. "
                "Choose search only for information that is current, externally verifiable, local, "
                "or otherwise needs sources. Choose skip for casual conversation, writing, reasoning, "
                "coding from supplied context, and questions answerable without current web facts."
            ),
            "criteria": {
                "search": "A web lookup is needed before answering.",
                "skip": "A web lookup is not needed before answering.",
            },
        }
    if skill_criteria and len(skill_criteria) > 1:
        questions["skill"] = {
            "instructions": (
                "Select the one registered SKILL.state skill that should manage this user request. "
                "Choose no matching skill unless a skill's stated purpose clearly applies."
            ),
            "criteria": skill_criteria,
        }

    answers = await evaluate_choices(
        state={
            "user_message": user_message,
            "environment": "Docker API container with a mounted session workspace; no host OS access.",
        },
        questions=questions,
        config=cfg.typesafe_config,
    )
    if answers is None:
        return fallback

    terminal = answers.get("terminal")
    file_tool = answers.get("file")
    mcp = answers.get("mcp")
    skill = answers.get("skill")
    web_search = answers.get("web_search")

    selected_skill = skill_fallback
    skill_decided = False
    if _accepted(skill, cfg):
        skill_decided = True
        selected_skill = None if skill.choice == "__no_matching_skill__" else skill.choice

    return GlobalRoutePlan(
        terminal_tool_enabled=(terminal.choice == "terminal") if _accepted(terminal, cfg) else terminal_fallback,
        file_tool_enabled=(file_tool.choice == "file") if _accepted(file_tool, cfg) else file_fallback,
        mcp_tools_enabled=(mcp.choice == "mcp") if _accepted(mcp, cfg) else mcp_fallback,
        selected_skill=selected_skill,
        skill_decided=skill_decided,
        web_search_answer=web_search if _accepted(web_search, cfg) else None,
        jev_evaluated=True,
        decisions=answers,
    )
