from __future__ import annotations

import json
from typing import Any

from ..config import AppConfig
from ..db import fetch_all, fetch_one
from ..storage import read_memory_file
from .prompt_cache import (
    PromptCache,
    default_cache as _prompt_cache,
    fingerprint_config,
    fingerprint_tool_set,
    make_cache_key,
)
from .skill_state import list_states as _list_skill_states, build_prompt_bundle

THINKING_GUIDANCE = {
    "off": "Reason minimally and answer directly with concise output.",
    "low": "Use lightweight reasoning and keep the answer compact.",
    "medium": "Use balanced reasoning depth before final answer.",
    "high": "Use deeper reasoning and check assumptions before final answer.",
}


def assemble_static_prefix(
    cfg: AppConfig,
    *,
    thinking_mode: str,
    terminal_tool_enabled: bool = False,
    tool_instruction_lines: list[str] | None = None,
) -> list[str]:
    """Return the *static* sections of the system prompt.

    These sections depend only on the configuration, the thinking mode and
    the set of tool instructions in effect.  They are identical for every
    turn in a session as long as those inputs do not change, which makes
    them a natural target for caching (see :mod:`prompt_cache`).
    """
    sections: list[str] = [
        f"System prompt: {cfg.system_prompt}",
        f"Session profile: {cfg.session_memory_profile}",
        f"Thinking mode for this turn: {thinking_mode}. {THINKING_GUIDANCE.get(thinking_mode, THINKING_GUIDANCE['medium'])}",
        "MCP/tools policy: call tools only when needed, do not invent tool output.",
        "Grounded facts policy: when a 'Grounded web search results' section is present, "
        "treat its citations as the only source of truth for the question and weave the "
        "answer around them with explicit [n] references. Never invent numbers, dates or "
        "names that are not in the grounded block.",
        "Memory carry-over policy: if the user asks you to list, name, repeat, or "
        "expand on something that was already covered in a previous turn of this "
        "session, FIRST check the 'Known entities from this session' / 'Durable facts' "
        "sections of the prompt before re-asking the user. Never reply with 'please "
        "provide a list' when the list is already in the prompt above.",
    ]
    if tool_instruction_lines:
        sections.append("Available tools:\n" + "\n".join(tool_instruction_lines))
        sections.append(
            "If a web-search MCP tool is available and the user asks about current information, "
            "local businesses, reviews, ratings, prices, news, releases, or recommendations, "
            "call the web-search tool before answering. Do not tell the user to search manually "
            "when a relevant search tool is available."
        )
    if terminal_tool_enabled:
        sections.append(
            "Terminal tool is available: run_terminal_command(command, timeout_sec). "
            "Use it when user asks for real environment data (files, commands, network state). "
            "Do not invent command output."
        )
    return sections


def assemble_prompt(
    session_id: str,
    user_message: str,
    cfg: AppConfig,
    recall_pack: dict[str, Any] | None,
    thinking_mode: str,
    terminal_tool_enabled: bool = False,
    tool_instruction_lines: list[str] | None = None,
    grounded_block: str | None = None,
    grounded_citations: list[dict[str, Any]] | None = None,
    active_skill: str | None = None,
) -> list[dict[str, str]]:
    pinned_rows = fetch_all(
        """
        SELECT content_text FROM messages
        WHERE session_id=? AND role='user' AND is_pinned=1
        ORDER BY timestamp ASC
        """,
        (session_id,),
    )
    pinned_text = "\n".join(r["content_text"] for r in pinned_rows)

    latest_user_row = fetch_one(
        "SELECT content_text FROM messages WHERE session_id=? AND role='user' ORDER BY timestamp DESC LIMIT 1",
        (session_id,),
    )
    latest_user = latest_user_row["content_text"] if latest_user_row else ""

    durable_facts = read_memory_file(session_id, "durable_facts.json")
    working_set = read_memory_file(session_id, "working_set.json")

    checkpoint_row = fetch_one(
        "SELECT * FROM checkpoints WHERE session_id=? ORDER BY checkpoint_index DESC LIMIT 1",
        (session_id,),
    )

    # Per-turn sections (pinned, durable, recall, checkpoint, grounded
    # search) -- these change every turn and are *not* eligible for
    # caching.  ``grounded_block`` carries the auto-search router output
    # when the "google where I don't know" policy fired for this turn.
    per_turn_sections: list[str] = [
        f"Pinned user instructions:\n{pinned_text or '(none)'}",
        f"Latest explicit user message:\n{latest_user or '(none)'}",
        f"Durable facts:\n{json.dumps(durable_facts, ensure_ascii=False)}",
        f"Working set:\n{json.dumps(working_set, ensure_ascii=False)}",
        "Retrieved historical chunks:\n"
        + (json.dumps(recall_pack, ensure_ascii=False) if recall_pack else "(none)"),
    ]

    # SKILL.state carry-over (legacy full mode).  When the session
    # is in ``skill_state`` mode but no skill is currently active
    # (no ``active_skill`` directive and the bag-of-words auto-router
    # did not match a registered skill), the runtime falls through
    # to the legacy prompt path.  In that path we still want the
    # model to see the durable facts the chat path has accumulated
    # — otherwise a follow-up "give me the answer with those
    # services" cannot use them.  We surface them as a dedicated
    # section so the model can quote them by name.
    if isinstance(durable_facts, list) and durable_facts:
        claim_lines = [
            f"- {f.get('claim')}"
            for f in durable_facts
            if isinstance(f, dict) and f.get("claim")
        ]
        if claim_lines:
            per_turn_sections.append(
                "Known entities from this session (carry-over from previous turns / runs):\n"
                + "\n".join(claim_lines[-30:])
            )
    if checkpoint_row:
        per_turn_sections.append(f"Latest checkpoint summary:\n{checkpoint_row['summary_text']}")
    if grounded_block:
        per_turn_sections.append(grounded_block)
    if grounded_citations:
        per_turn_sections.append(
            "Grounded citations (use the bracketed numbers in your answer):\n"
            + json.dumps(grounded_citations, ensure_ascii=False)
        )

    # SKILL.state: when an active skill is in scope the model receives
    # only the (spec, state, observation) bundle. The append-only
    # ``recent_rows`` history is dropped from the prompt per the paper —
    # this is what prevents context poisoning on long horizons.
    skill_bundle = None
    if active_skill:
        try:
            skill_bundle = build_prompt_bundle(session_id, active_skill)
        except Exception:
            skill_bundle = None

    static_sections = assemble_static_prefix_cached(
        cfg,
        thinking_mode=thinking_mode,
        terminal_tool_enabled=terminal_tool_enabled,
        tool_instruction_lines=tool_instruction_lines,
    )

    messages: list[dict[str, str]] = []

    # When a SKILL.state skill is active we do NOT replay any of the
    # append-only chat history (no pinned rows, no latest user msg,
    # no durable facts, no working set, no recall, no checkpoint, no
    # recent_rows). The model's view of the past is the bounded ring
    # inside ``skill_bundle.history``. This is the core claim of the
    # paper: the prompt does not grow with execution history, and
    # context poisoning is structurally impossible.
    if skill_bundle is not None:
        # SKILL.state carry-over: when the runtime has accumulated
        # ``known_entities`` in this session (via the foreground chat
        # extractor or the background run extractor), surface them as
        # a separate section so the model does not have to re-search
        # for facts it already has.  The structured ``known_entities``
        # field is also embedded in the JSON bundle below for
        # completeness.
        known_entities = list(skill_bundle.get("known_entities") or [])
        carryover_sections: list[str] = []
        if known_entities:
            claim_lines = [
                f"- {ent.get('claim')}" for ent in known_entities if ent.get("claim")
            ]
            if claim_lines:
                carryover_sections.append(
                    "Known entities from this session (carry-over from previous turns / runs):\n"
                    + "\n".join(claim_lines)
                )
        bundle_text = (
            "SKILL.state active skill bundle:\n"
            + json.dumps(skill_bundle, ensure_ascii=False, indent=2)
        )
        messages.append(
            {
                "role": "system",
                "content": "\n\n".join(static_sections + carryover_sections + [bundle_text]),
            }
        )
        # Surface the current user turn as the only conversational
        # message the provider needs to see.
        messages.append({"role": "user", "content": user_message})
        return messages

    # Legacy code path — unchanged for backward compatibility. Only
    # reached when no SKILL.state skill is active.
    prompt_sections = static_sections + per_turn_sections
    messages = [{"role": "system", "content": "\n\n".join(prompt_sections)}]

    recent_rows = fetch_all(
        """
        SELECT role, content_text FROM messages
        WHERE session_id=?
        ORDER BY timestamp DESC
        LIMIT 12
        """,
        (session_id,),
    )
    recent_rows.reverse()
    if recent_rows and recent_rows[-1]["role"] == "user":
        recent_rows[-1] = {"role": "user", "content_text": user_message}
    else:
        recent_rows.append({"role": "user", "content_text": user_message})

    # OpenAI-compatible providers such as LM Studio require a system message
    # to be the first message in a conversation. Internal events are stored as
    # `system` messages for the UI, but must not be replayed as later system
    # turns to the provider. Only conversational roles belong in chat history.
    for row in recent_rows:
        if row["role"] not in {"user", "assistant"}:
            continue
        messages.append({"role": row["role"], "content": row["content_text"]})
    return messages


def assemble_static_prefix_cached(
    cfg: AppConfig,
    *,
    thinking_mode: str,
    terminal_tool_enabled: bool = False,
    tool_instruction_lines: list[str] | None = None,
    cache: PromptCache | None = None,
) -> list[str]:
    """Return the static prefix, consulting a cache first.

    The cache key is derived from the configuration fingerprint, the
    thinking mode and a hash of the tool instruction lines.  When the
    cache hits, the prefix is rendered from the cached fragment; on a
    miss the full prefix is assembled and stored for next time.
    """
    cache = cache or _prompt_cache
    key = make_cache_key(
        config_hash=fingerprint_config(cfg),
        thinking_mode=str(thinking_mode),
        tool_set_hash=fingerprint_tool_set(tool_instruction_lines),
    )
    cached = cache.get(key)
    if cached is not None:
        # Cache stores a single joined string for compactness; split back
        # into the canonical section list.
        return cached.split("\n\n--SECTION--\n\n")
    sections = assemble_static_prefix(
        cfg,
        thinking_mode=thinking_mode,
        terminal_tool_enabled=terminal_tool_enabled,
        tool_instruction_lines=tool_instruction_lines,
    )
    cache.put(key, "\n\n--SECTION--\n\n".join(sections))
    return sections
