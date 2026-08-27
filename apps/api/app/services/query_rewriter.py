"""Search query rewriter — uses a separate, fast LLM call to convert the
last user turn (plus the recent chat context) into a self-contained web
search query.

Why this exists
---------------

Ultra-fast local models frequently drop pronouns ("the one we talked
about"), refer to prior context ("the artist" → which performer?), or
type meta-commands ("Force web search") instead of stating a search
topic.  When that message is fed verbatim to the search backend the
results are useless (the Bing RSS endpoint returns the FORCE bicycle
brand for the query "Force", and the meaning-of-"Performer" article
for "the artist").

A dedicated, lightweight model instance gets the whole session as
context plus a tight instruction:

    "Given this conversation, write a single web search query that
    captures what the user actually wants to find.  Output ONLY the
    query — no quotes, no explanation."

That query is what ``run_auto_search`` hands to the native-web-search
MCP backend.  The main session model then sees the citations, quotes
them, and answers the user — exactly the flow the orchestrator
already implements, just with a much better query upstream.

When the rewriter fails (network error, provider not configured, the
model returns junk) we fall back to ``_resolve_search_query``'s cheap
prior-message concatenation so the chat still works.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import httpx

from ..config import load_app_config
from .orchestrator import build_payload, provider_timeout_seconds, resolve_provider_model

logger = logging.getLogger(__name__)


# Hard cap on the rewritten query — Bing/Brave/etc. handle ~10 words
# best; anything longer dilutes the ranking.
_MAX_QUERY_CHARS = 200
# Hard cap on the context we forward to the rewriter so we don't blow
# up its prompt when the session has been going for hours.
_MAX_CONTEXT_CHARS = 4000
# Per-call timeout.  The rewriter is best-effort; a slow call must
# never delay the main chat response noticeably.
_REWRITER_TIMEOUT_SEC = 15


_REWRITER_SYSTEM_PROMPT = (
    "You are a search query rewriter. The user is in a conversation "
    "with another assistant. Your job is to read the conversation and "
    "produce ONE concise web search query that captures what the user "
    "actually wants to find right now.\n\n"
    "Rules:\n"
    "1. Output ONLY the search query. No quotes, no explanation, no "
    "preamble, no JSON, no code fences.\n"
    "2. Keep the query in the same language as the user's message "
    "(Ukrainian, Russian, English, etc.) unless the subject is a "
    "well-known foreign proper noun.\n"
    "3. Resolve pronouns, follow-ups, and meta-commands ('Force web "
    "search', 'find that artist', 'the performer') against the prior "
    "context — write the query the user MEANT, not the words they "
    "typed.\n"
    "4. Keep the query under 12 words. Use the most distinctive "
    "nouns/proper names from the conversation.\n"
    "5. Never invent facts. If the conversation truly gives no "
    "searchable subject, output exactly: NONE\n"
)


def _format_conversation_for_rewriter(
    last_user_message: str,
    recent_user_messages: list[str] | None,
) -> str:
    """Render the recent chat as a compact transcript the rewriter can
    consume.  We only forward user turns — the assistant turns are not
    needed to derive a search query and would dilute the signal."""

    lines: list[str] = []
    if recent_user_messages:
        for prior in recent_user_messages[-6:]:
            text = (prior or "").strip()
            if not text:
                continue
            lines.append(f"User: {text}")
    lines.append(f"User: {last_user_message}")
    blob = "\n".join(lines)
    if len(blob) > _MAX_CONTEXT_CHARS:
        blob = blob[-_MAX_CONTEXT_CHARS:]
        # Re-align to the start of a "User:" line so the rewriter
        # doesn't see a half-line at the top.
        idx = blob.find("User:")
        if idx > 0:
            blob = blob[idx:]
    return blob


def _sanitize_rewriter_output(raw: str) -> str | None:
    """Strip common LLM artefacts (``<think>`` blocks, code fences,
    bullet points, leading quotes) and return a clean query.  Returns
    ``None`` if the rewriter explicitly said it couldn't derive a
    query or the result is empty/garbage."""

    if not raw:
        return None
    text = raw.strip()

    # Strip <think>...</think> blocks some local models emit.
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)

    # Strip code fences: extract the *inner* content of a fenced block
    # when it's the only thing the model emitted, otherwise just drop
    # the fence delimiters.
    fence_match = re.search(r"```(?:[a-zA-Z0-9_+\-]*\n)?([\s\S]*?)```", text)
    if fence_match and fence_match.group(1).strip():
        text = fence_match.group(1).strip()
    else:
        text = re.sub(r"```", "", text).strip()

    # Take the first non-empty line — models occasionally add a
    # trailing "Here is the query:" preamble.
    _PREAMBLE_LINE_RE = re.compile(
        r"^(here (is|are)|"
        r"the (search )?query (is|:)|"
        r"query:|"
        r"search query:|"
        r"output:|"
        r"answer:|"
        r"result:|"
        r"\s*[-*]?\s*sure[,!.]?|"
        r"certainly[,!.]?)\s*",
        re.IGNORECASE,
    )
    candidate: str | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Drop wrapping quotes first so the bullet-marker regex can
        # see the actual first non-quote character.
        line = line.strip("\"'`«»“”‘’")
        # Drop leading bullet/number markers.
        line = re.sub(r"^[\-\*\u2022\d.\)\s]+", "", line).strip()
        # Drop wrapping quotes once more (the bullet strip may have
        # exposed a leading quote on a line like `"- foo`).
        line = line.strip("\"'`«»“”‘’")
        if not line:
            continue
        # Skip common preamble lines ("Here is the query:", etc.).
        if _PREAMBLE_LINE_RE.match(line):
            continue
        candidate = line
        break

    if not candidate:
        return None
    text = candidate

    lowered = text.lower().strip()
    if lowered in {"none", "n/a", "no query", "no search", "-"}:
        return None
    if len(text) > _MAX_QUERY_CHARS:
        text = text[:_MAX_QUERY_CHARS].rstrip()
    return text


async def rewrite_search_query(
    last_user_message: str,
    *,
    recent_user_messages: list[str] | None = None,
    provider_id: str | None = None,
    model_id: str | None = None,
) -> str | None:
    """Return a single web-search query that captures what the user
    wants to find.  Returns ``None`` when the rewriter is unavailable
    or fails — callers MUST fall back to ``_resolve_search_query``.

    ``provider_id`` / ``model_id`` default to the session's active
    pair so the rewriter uses the same model the chat uses.  Callers
    can override (e.g. a tiny dedicated model) via Settings in the
    future.
    """

    cfg = load_app_config()
    provider, model_obj = resolve_provider_model(
        provider_id=provider_id,
        model_id=model_id,
    )
    if provider is None or model_obj is None or not provider.base_url:
        logger.debug("query_rewriter: no provider/model configured; skipping")
        return None

    conversation_blob = _format_conversation_for_rewriter(
        last_user_message, recent_user_messages
    )
    prompt_messages = [
        {"role": "system", "content": _REWRITER_SYSTEM_PROMPT},
        {"role": "user", "content": conversation_blob},
    ]

    try:
        url, headers, payload = build_payload(
            provider=provider,
            model=model_obj,
            messages=prompt_messages,
            stream=False,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("query_rewriter: build_payload failed: %s", exc)
        return None

    timeout_sec = min(provider_timeout_seconds(provider), _REWRITER_TIMEOUT_SEC)
    try:
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            obj = resp.json()
    except (httpx.HTTPError, asyncio.TimeoutError, json.JSONDecodeError) as exc:
        logger.debug("query_rewriter: provider call failed: %s", exc)
        return None
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("query_rewriter: unexpected error: %s", exc)
        return None

    try:
        choices = obj.get("choices") or []
        if not choices:
            return None
        message = choices[0].get("message") or {}
        raw = str(message.get("content") or "").strip()
    except (AttributeError, TypeError, IndexError):
        return None
    if not raw:
        return None
    return _sanitize_rewriter_output(raw)


async def rewrite_search_query_safe(
    last_user_message: str,
    *,
    recent_user_messages: list[str] | None = None,
    provider_id: str | None = None,
    model_id: str | None = None,
    fallback_resolver: Any = None,
) -> str | None:
    """Same as :func:`rewrite_search_query` but guarantees a result by
    falling back to ``fallback_resolver`` (typically
    ``_resolve_search_query``) on any failure.  Returns the raw user
    message as a last resort so the search backend always gets
    *something*."""

    rewritten: str | None = None
    try:
        rewritten = await rewrite_search_query(
            last_user_message,
            recent_user_messages=recent_user_messages,
            provider_id=provider_id,
            model_id=model_id,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("query_rewriter: raised %s", exc)

    if rewritten:
        return rewritten
    if fallback_resolver is not None:
        try:
            fallback = fallback_resolver(last_user_message, recent_user_messages)
        except Exception:
            fallback = None
        if fallback and fallback.strip():
            return fallback.strip()
    base = (last_user_message or "").strip()
    return base or None


__all__ = [
    "rewrite_search_query",
    "rewrite_search_query_safe",
]