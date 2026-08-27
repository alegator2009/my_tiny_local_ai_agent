"""Auto web search — the orchestrator's "google where I don't know" router.

Why this module exists
----------------------

The bundled ``mcp/web-search-mcp`` server exposes three tools the local
model *could* call: ``full-web-search``, ``get-web-search-summaries`` and
``get-single-web-page-content``.  In practice ultra-fast, non-frontier
models forget to call them, or call them with a vague query, or call
them and then ignore the result.  The result is a confident hallucination.

This module takes the *decision* to search out of the model's hands.  The
orchestrator runs it *before* the model is asked to answer:

1. ``should_search`` — a cheap regex-based heuristic decides whether the
   user is asking for fresh, local or fact-grounded information.  The
   ``policy`` from ``AppConfig`` (``off`` / ``auto`` / ``always``) and
   a per-turn ``force_search`` flag from the UI further gate the call.
2. ``AutoSearchCache`` — a SQLite-backed LRU keeps a normalised,
   hash-keyed view of recent answers so the same question is never
   re-fetched inside the configured TTL.
3. ``run_auto_search`` — when no cache hit exists, it spins up a tiny
   ``MCPToolRegistry`` for the native-web-search backend, calls
   ``get-web-search-summaries`` (the lightweight tool) and assembles a
   deterministic ``{answer, citations}`` block.  No extra LLM is invoked
   to "summarise" the result — the snippets themselves are short enough
   to paste into the prompt verbatim.
4. ``build_grounded_block`` — formats the cache hit / fresh result into
   the per-turn prompt section the model sees.

The block the model receives is intentionally small and structured:

    Grounded web search results (engine: bing, 3 sources, 412 chars):
    [1] Title — example.com
        "quoted snippet line one. quoted snippet line two."
    [2] ...
    Answer to weave into the response:
    "First sentence from the top snippet. Second sentence from the
    same source."

That shape is what ultra-fast local models can use without reasoning
about which source said what.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable

from ..config import AppConfig, AutoSearchConfig
from ..db import execute, fetch_one, init_db, utcnow_iso
from .mcp import MCPError, MCPToolRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

# A query is "freshness" / "local" / "factual" when it carries one of these
# cues.  Defaults are intentionally narrow — the orchestrator already
# has its own ``_needs_fresh_or_local_info`` heuristic which we share via
# the same import path, but the auto-search router keeps its own copy so
# the policy is decoupled from the orchestrator.
_DEFAULT_FRESHNESS_HINTS = [
    # English
    "today", "latest", "current", "fresh", "recent", "now",
    "this week", "this month", "this year", "right now",
    "as of", "updated",
    # Russian
    "сегодня", "сейчас", "актуальн", "свеж", "новост", "текущ",
    "в этом году", "в этом месяце", "на этой неделе",
    # Ukrainian
    "сьогодні", "зараз", "актуальн", "свіж", "новин", "поточн",
    # Spanish / Portuguese
    "hoy", "actual", "reciente", "ahora", "último",
    # French / Italian
    "aujourd'hui", "récent", "maintenant", "oggi", "recente",
    # German / Dutch
    "heute", "aktuell", "kürzlich", "jetzt", "vandaag", "recent",
    # Chinese / Japanese / Korean
    "今天", "最新", "现在", "今日", "最新", "現在", "최근",
    # Numbers and years — "in 2024", "since 2023"
    r"\b(19|20)\d{2}\b",
    r"\b\d{1,2}[./-]\d{1,2}[./-](19|20)\d{2}\b",
]

_DEFAULT_FACTUAL_HINTS = [
    # "who", "what", "when", "where", "how many / much / long"
    r"\bwho\b", r"\bwhat\b", r"\bwhen\b", r"\bwhere\b",
    r"\bhow (many|much|long|old|far|fast|tall|heavy)\b",
    r"\bwhich\b",
    # price / cost / score / rating
    r"\b(price|cost|rate|score|rating|stock|exchange)\b",
    r"\b(buy|sell|cheaper|cheapest|expensive)\b",
    # people / orgs
    r"\b(president|ceo|founder|author|company|organization|organisation)\b",
    r"\b(founded|established|released|launched|published)\b",
    # definitions
    r"\b(define|definition|meaning of|stands for)\b",
    r"\b(difference between|compare|comparison)\b",
    # place-specific
    r"\bin [A-Z][a-zA-Z]+",  # crude but catches "in Paris", "in Kyiv"
    # currency / measurement
    r"\$\d", r"€\d", r"₴\d", r"₽\d", r"\b\d+\s*(usd|eur|uah|rub|gbp)\b",
]

_DEFAULT_OPINION_HINTS = [
    # "what do you think", "how do you feel", "your opinion"
    r"\b(what do you think|how do you feel|your opinion|do you like)\b",
    r"\b(opinion|thoughts|feelings|impressions? about)\b",
    # greetings / pleasantries
    r"^(hi|hello|hey|good (morning|afternoon|evening))[\s\.\!]?$",
    r"^(привіт|здоров|добр(ый|ого) (день|вечір|ранок))",
]

# Strip surrounding punctuation, lower-case, collapse whitespace — this is
# what we feed both to the cache key and to the search backend so two
# near-identical user phrasings hit the same row.
_QUERY_NORMALIZE_RE = re.compile(r"[\s\u00A0]+", re.UNICODE)
_QUERY_STRIP_RE = re.compile(r"^[\s\"'`«»“”‘’(\[]+|[\s\"'`«»“”‘’.,;:!?)\]]+$", re.UNICODE)


def _normalize_query(raw: str) -> str:
    if not raw:
        return ""
    text = raw.strip().lower()
    text = _QUERY_NORMALIZE_RE.sub(" ", text)
    text = _QUERY_STRIP_RE.sub("", text)
    return text.strip()


# Queries shorter than this are treated as follow-ups and get prefixed
# with the most recent prior user messages so the search backend can
# see the conversation context.  Keep the threshold high enough that
# standalone questions ("What is X?") are NOT inflated.
_FOLLOWUP_MAX_CHARS = 40
# How many prior user messages to fold into a short follow-up.
_FOLLOWUP_CONTEXT_COUNT = 3
# Hard cap on the joined query so we don't accidentally create a giant
# blob if the user has been chatting for a while.
_FOLLOWUP_MAX_QUERY_CHARS = 300


def _looks_like_followup(user_message: str) -> bool:
    """Heuristic: a single-word reply, a clarification, or a short
    pronoun-heavy fragment is treated as a follow-up.  ``no_signal``
    queries (``policy=auto`` triggers) rarely reach here because the
    router only fires on the actual user message, but we still want the
    search backend to see the surrounding context."""

    text = (user_message or "").strip()
    if not text:
        return False
    if len(text) >= _FOLLOWUP_MAX_CHARS:
        return False
    # One or two words → definitely a follow-up.
    if len(text.split()) <= 2:
        return True
    # Pronoun-heavy: "the one from", "that guy", "her", "him" — almost
    # always refers back to something earlier.
    if re.search(r"\b(the one|the other|that (one|guy|person|band|song)|he|she|they|him|her|his|hers|their|its|this|этот|это|этого|этой|той|тот|того|той|того|то|этих|тех|цей|ця|це|цей|той)\b", text, re.IGNORECASE):
        return True
    return False


def _resolve_search_query(
    user_message: str,
    recent_user_messages: list[str] | None,
) -> str:
    """Compose the actual search-backend query.

    Long, self-contained messages are returned untouched.  Short or
    pronoun-heavy follow-ups get prefixed with the most recent prior
    user messages so the search engine has the surrounding context
    (without us spending an LLM call to rewrite it).

    This is the *fallback* used when the LLM-based query rewriter
    (see ``query_rewriter.py``) is unavailable or fails.  The orchestrator
    prefers the rewriter because it understands pronouns, follow-ups and
    meta-commands ("Force web search", "the artist") much better than a
    plain concatenation can."""

    base = (user_message or "").strip()
    if not recent_user_messages or not _looks_like_followup(base):
        return base

    # Walk backwards through the prior messages, oldest → newest, and
    # concatenate up to the cap.
    parts: list[str] = []
    total = len(base)
    for prior in recent_user_messages[-(_FOLLOWUP_CONTEXT_COUNT * 2):]:
        prior_text = (prior or "").strip()
        if not prior_text:
            continue
        # Stop once we've accumulated enough context.
        if total + len(prior_text) + 1 > _FOLLOWUP_MAX_QUERY_CHARS:
            break
        if prior_text in parts:
            continue
        parts.append(prior_text)
        total += len(prior_text) + 1
        if len(parts) >= _FOLLOWUP_CONTEXT_COUNT and total >= _FOLLOWUP_MAX_CHARS:
            break

    if not parts:
        return base

    parts.append(base)
    return " | ".join(parts)


def _hash_query(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Decision: should we actually run a search?
# ---------------------------------------------------------------------------


@dataclass
class SearchDecision:
    should_search: bool
    reason: str
    policy: str
    query: str
    normalized_query: str


def should_search(
    user_message: str,
    *,
    policy: str,
    enabled: bool,
    force: bool = False,
    freshness_hints: list[str] | None = None,
    factual_hints: list[str] | None = None,
    opinion_hints: list[str] | None = None,
) -> SearchDecision:
    """Decide whether to fire an automatic web search for this turn.

    ``force`` (set when the UI's per-message "Force web search" toggle is
    on) wins over every other check, including ``policy=off``.
    """

    query = (user_message or "").strip()
    normalized = _normalize_query(query)

    if not normalized:
        return SearchDecision(
            should_search=False,
            reason="empty_query",
            policy=policy,
            query=query,
            normalized_query=normalized,
        )

    if force:
        return SearchDecision(
            should_search=True,
            reason="forced_by_user",
            policy=policy,
            query=query,
            normalized_query=normalized,
        )

    if not enabled or policy == "off":
        return SearchDecision(
            should_search=False,
            reason="policy_off",
            policy=policy,
            query=query,
            normalized_query=normalized,
        )

    if policy == "always":
        return SearchDecision(
            should_search=True,
            reason="policy_always",
            policy=policy,
            query=query,
            normalized_query=normalized,
        )

    # policy == "auto": cheap regex scan
    f_hints = freshness_hints if freshness_hints is not None else _DEFAULT_FRESHNESS_HINTS
    fa_hints = factual_hints if factual_hints is not None else _DEFAULT_FACTUAL_HINTS
    o_hints = opinion_hints if opinion_hints is not None else _DEFAULT_OPINION_HINTS

    for pattern in o_hints:
        if re.search(pattern, normalized, flags=re.IGNORECASE | re.UNICODE):
            return SearchDecision(
                should_search=False,
                reason="opinion_or_chitchat",
                policy=policy,
                query=query,
                normalized_query=normalized,
            )

    for pattern in f_hints:
        if re.search(pattern, normalized, flags=re.IGNORECASE | re.UNICODE):
            return SearchDecision(
                should_search=True,
                reason="freshness_hint",
                policy=policy,
                query=query,
                normalized_query=normalized,
            )

    for pattern in fa_hints:
        if re.search(pattern, normalized, flags=re.IGNORECASE | re.UNICODE):
            return SearchDecision(
                should_search=True,
                reason="factual_hint",
                policy=policy,
                query=query,
                normalized_query=normalized,
            )

    # As a last-resort signal, a question mark / 5-gram length bias helps
    # catch "What's the weather in Lisbon?" which our factual hints
    # already cover; if we got here the query is short, declarative and
    # doesn't look like a knowledge gap.
    return SearchDecision(
        should_search=False,
        reason="no_signal",
        policy=policy,
        query=query,
        normalized_query=normalized,
    )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class AutoSearchCache:
    """Tiny SQLite-backed cache for grounded search results.

    The cache key is a sha256 of the *normalised* query (see
    :func:`_normalize_query`).  Each hit bumps ``last_used_at`` and
    ``hits`` so the LRU-like ``purge`` below can trim cold rows.
    """

    def __init__(self, ttl_sec: int):
        self.ttl_sec = max(0, int(ttl_sec))

    def get(self, query: str) -> dict[str, Any] | None:
        key = _hash_query(_normalize_query(query))
        if not key:
            return None
        row = fetch_one(
            """
            SELECT answer_text, citations_json, engine, source, expires_at, last_used_at, hits
            FROM auto_search_cache
            WHERE cache_key = ?
            """,
            (key,),
        )
        if row is None:
            return None
        # Lazy expiry — the row is removed on access if it has aged out.
        if self.ttl_sec > 0:
            try:
                from datetime import datetime, timezone

                expires = datetime.fromisoformat(row["expires_at"])
                if expires <= datetime.now(timezone.utc):
                    execute("DELETE FROM auto_search_cache WHERE cache_key = ?", (key,))
                    return None
            except Exception:
                # Corrupt timestamp — drop the row.
                execute("DELETE FROM auto_search_cache WHERE cache_key = ?", (key,))
                return None
        # Touch.
        now_iso = utcnow_iso()
        execute(
            "UPDATE auto_search_cache SET last_used_at = ?, hits = hits + 1 WHERE cache_key = ?",
            (now_iso, key),
        )
        try:
            citations = json.loads(row["citations_json"])
        except Exception:
            citations = []
        return {
            "answer": row["answer_text"],
            "citations": citations if isinstance(citations, list) else [],
            "engine": row["engine"] or "",
            "source": row["source"] or "auto_search",
            "cache_hit": True,
            "hits": int(row["hits"] or 0) + 1,
        }

    def put(
        self,
        query: str,
        *,
        answer: str,
        citations: list[dict[str, Any]],
        engine: str = "",
        source: str = "auto_search",
    ) -> None:
        if not query.strip():
            return
        key = _hash_query(_normalize_query(query))
        if not key:
            return
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        expires = now + timedelta(seconds=self.ttl_sec) if self.ttl_sec > 0 else now
        now_iso = now.isoformat()
        expires_iso = expires.isoformat()
        execute(
            """
            INSERT INTO auto_search_cache (
                cache_key, query_text, answer_text, citations_json,
                engine, source, created_at, expires_at, last_used_at, hits
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            ON CONFLICT(cache_key) DO UPDATE SET
                answer_text = excluded.answer_text,
                citations_json = excluded.citations_json,
                engine = excluded.engine,
                source = excluded.source,
                created_at = excluded.created_at,
                expires_at = excluded.expires_at,
                last_used_at = excluded.last_used_at
            """,
            (
                key,
                _normalize_query(query),
                answer,
                json.dumps(citations, ensure_ascii=False),
                engine,
                source,
                now_iso,
                expires_iso,
                now_iso,
            ),
        )

    def purge(self) -> int:
        """Delete all expired rows.  Returns the number of removed rows."""

        from datetime import datetime, timezone

        if self.ttl_sec <= 0:
            return 0
        now_iso = datetime.now(timezone.utc).isoformat()
        # SQLite RETURNING is unavailable on older builds; we just count.
        rows = fetch_all(
            "SELECT cache_key FROM auto_search_cache WHERE expires_at <= ?",
            (now_iso,),
        )
        for row in rows:
            execute("DELETE FROM auto_search_cache WHERE cache_key = ?", (row["cache_key"],))
        return len(rows)


# ---------------------------------------------------------------------------
# Calling the native-web-search MCP backend
# ---------------------------------------------------------------------------


def _extract_text(result: Any) -> str:
    """Best-effort flatten of an MCP tool result into a single string."""

    if not isinstance(result, dict):
        return json.dumps(result, ensure_ascii=False)
    content = result.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if text:
                    parts.append(str(text))
        if parts:
            return "\n".join(parts)
    return json.dumps(result, ensure_ascii=False)


def _parse_citations_from_summary_text(
    text: str, max_citations: int
) -> list[dict[str, Any]]:
    """Parse ``get-web-search-summaries`` / ``full-web-search`` output into
    structured citations.

    The MCP server formats its output as::

        **1. Title**
        URL: https://...
        Description: snippet
        **Full Content:**
        <extracted body>
        ---
        **2. Title**
        ...

    We extract the URL, title, description *and* any Full Content the
    extractor was able to scrape.  When the description is empty (the
    RSS snippet for YouTube / Apple Music / Instagram is often "No
    description available"), we fall back to the first useful chunk of
    Full Content so the model still has something to quote.

    This parser is permissive — it falls back to URL-only extraction when
    the title cannot be read.
    """

    citations: list[dict[str, Any]] = []
    if not text:
        return citations
    # Split on the dashed separator the MCP server uses between entries.
    blocks = re.split(r"\n\s*---\s*\n", text)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        title_match = re.search(r"\*\*\d+\.\s*([^\*\n]+)\*\*", block)
        url_match = re.search(r"URL:\s*(\S+)", block)
        desc_match = re.search(r"Description:\s*([\s\S]+?)(?:\n\n|\Z)", block)
        # ``Full Content:`` section is optional and only present when the
        # ``full-web-search`` tool ran content extraction.  Capture up to
        # the next blank line / end-of-block.
        content_match = re.search(
            r"\*\*Full Content:\*\*\s*\n([\s\S]+?)(?:\n\n|\Z)",
            block,
        )
        if not url_match:
            continue
        url = url_match.group(1).strip().rstrip(".,;")
        title = (title_match.group(1).strip() if title_match else url)
        description = (desc_match.group(1).strip() if desc_match else "")
        full_content = (content_match.group(1).strip() if content_match else "")
        if description and len(description) > 800:
            description = description[:800].rstrip() + "…"
        if full_content and len(full_content) > 1500:
            full_content = full_content[:1500].rstrip() + "…"
        # Fallback: if the RSS description is missing (either truly empty
        # or the placeholder the MCP server emits when the RSS snippet
        # was blank — common for YouTube / Apple Music / Instagram), use
        # a slice of the extracted page body so the model still has
        # something concrete to quote.
        if full_content and (
            not description
            or description.lower() == "no description available"
        ):
            description = full_content[:300].rstrip() + (
                "…" if len(full_content) > 300 else ""
            )
        citations.append(
            {
                "title": title,
                "url": url,
                "description": description,
                "snippet": description,
                "full_content": full_content,
                "engine": "",
            }
        )
        if len(citations) >= max_citations:
            break
    return citations


def _extract_engine_from_text(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"Search engine:\s*([A-Za-z0-9_]+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def _pick_summaries_tool(registry: MCPToolRegistry) -> str | None:
    """Return the namespaced tool name we should call for auto-search.

    Preference order:
      1. ``full_web_search`` (heavier — actually extracts page content via
         Playwright; necessary for sites like YouTube/Apple Music/Instagram
         whose RSS description is empty so the model gets nothing useful)
      2. ``get_web_search_summaries`` (lightweight, RSS snippets only —
         only useful as a fast-path when full extraction is disabled)
    """

    candidates = list(registry.tools_by_name.keys())
    for name in candidates:
        if name.endswith("__full_web_search"):
            return name
    for name in candidates:
        if name.endswith("__get_web_search_summaries"):
            return name
    return None


async def _call_native_web_search(
    query: str,
    *,
    cfg: AutoSearchConfig,
    # Generous default — YouTube, Instagram and other JS-heavy sites
    # can take 20–40s to render before content extraction can scrape
    # their description, and we may have to fan that out across up to
    # ``max_citations`` URLs in a single call.
    request_timeout_sec: int = 120,
) -> dict[str, Any]:
    """Spin up a one-shot MCP registry, fire the search, and tear it down.

    We use a fresh registry per call because the auto-search router runs
    once per user turn and the orchestrator may not have any MCP clients
    active (e.g. the user disabled MCP tools globally but still wants the
    auto-search to work).
    """

    from ..config import load_app_config, effective_mcp_servers

    app_cfg = load_app_config()
    if not app_cfg.mcp_config.enabled or not app_cfg.mcp_config.native_web_search_enabled:
        return {"ok": False, "error": "native_web_search_disabled", "answer": "", "citations": []}

    servers = effective_mcp_servers(app_cfg)
    if not servers:
        return {"ok": False, "error": "no_mcp_servers", "answer": "", "citations": []}

    registry = MCPToolRegistry()
    try:
        registry = await MCPToolRegistry.from_server_configs(servers)
    except Exception as exc:  # pragma: no cover - defensive
        return {"ok": False, "error": f"registry_init_failed: {exc}", "answer": "", "citations": []}

    try:
        tool_name = _pick_summaries_tool(registry)
        if not tool_name:
            return {
                "ok": False,
                "error": "no_search_tool_registered",
                "answer": "",
                "citations": [],
            }

        # Sanitize: the tool may not be the lightweight one, in which case
        # the registry's ``tool_schemas`` already dropped a sanitised version
        # — but the *call* uses the raw schema, so we let ``call_tool``
        # normalise arguments for us.
        try:
            raw = await asyncio.wait_for(
                registry.call_tool(
                    tool_name,
                    {
                        "query": query,
                        "limit": cfg.max_citations,
                        # Always ask for full-page content extraction so
                        # the model gets descriptions from sites whose
                        # RSS snippet is empty (YouTube, Apple Music,
                        # Instagram, etc.). The heavy tool (full-web-search)
                        # uses this flag to decide whether to spin up
                        # Playwright; the lightweight summaries tool
                        # ignores it.
                        "includeContent": True,
                    },
                ),
                timeout=request_timeout_sec,
            )
        except asyncio.TimeoutError:
            return {"ok": False, "error": "search_timeout", "answer": "", "citations": []}
        except MCPError as exc:
            return {"ok": False, "error": f"mcp_error: {exc}", "answer": "", "citations": []}

        text = _extract_text(raw)
        citations = _parse_citations_from_summary_text(text, cfg.max_citations)
        engine = _extract_engine_from_text(text)
        # Fallback: the lightweight tool does not always emit the
        # "Search engine:" line, so we look at the structured MCP result
        # for ``engine``/``search_engine`` keys (the heavy tool puts the
        # value there) before giving up.
        if not engine and isinstance(raw, dict):
            for key in ("engine", "search_engine"):
                value = raw.get(key)
                if isinstance(value, str) and value.strip():
                    engine = value.strip()
                    break
        return {
            "ok": True,
            "answer": text,
            "citations": citations,
            "engine": engine,
            "raw": raw,
        }
    finally:
        try:
            await registry.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Assemble the answer from raw search output
# ---------------------------------------------------------------------------


def _format_answer(
    citations: list[dict[str, Any]],
    *,
    max_chars: int,
    snippet_chars: int,
    include_snippets: bool,
    include_full_content: bool,
) -> str:
    """Turn the citation list into a tight prose answer.

    The strategy: take the first citation's description, prepend a lead
    from the title, then glue the next one or two descriptions together
    with explicit "also:" markers.  We avoid the temptation to invoke
    another LLM here — the snippets are short enough to quote verbatim.
    """

    if not citations:
        return ""

    parts: list[str] = []
    used = 0
    for idx, c in enumerate(citations):
        title = str(c.get("title") or c.get("url") or "").strip()
        snippet = str(c.get("snippet") or c.get("description") or "").strip()
        if snippet_chars and len(snippet) > snippet_chars:
            snippet = snippet[:snippet_chars].rstrip() + "…"
        # ``full_content`` is populated by ``_parse_citations_from_summary_text``
        # when ``full-web-search`` ran its content extractor.  When the
        # RSS description is empty (YouTube / Apple Music / Instagram)
        # this is the only way the model gets a concrete quote.
        if include_full_content and c.get("full_content"):
            extra = str(c.get("full_content") or "").strip()
            if snippet_chars and len(extra) > snippet_chars * 3:
                extra = extra[: snippet_chars * 3].rstrip() + "…"
            if extra:
                snippet = (snippet + "\n" + extra).strip() if snippet else extra
        if not include_snippets:
            snippet = ""
        line_bits: list[str] = []
        if title:
            line_bits.append(f"[{idx + 1}] {title}")
        if snippet:
            line_bits.append(snippet)
        block = "\n".join(line_bits).strip()
        if not block:
            continue
        sep = "\n\n" if parts else ""
        tentative = sep.join(parts) + sep + block
        if max_chars and len(tentative) > max_chars:
            break
        parts.append(block)
        used += 1
        if used >= 5:
            break
    return "\n\n".join(parts).strip()


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


@dataclass
class AutoSearchResult:
    """The shape the orchestrator consumes.

    ``answer`` is a short prose block; ``citations`` is the structured
    source list; ``grounded_block`` is the verbatim text we paste into
    the per-turn prompt section.  ``cache_hit`` distinguishes fresh
    searches from cached ones (useful in the UI timeline and for
    telemetry).
    """

    query: str
    normalized_query: str
    answer: str
    citations: list[dict[str, Any]]
    engine: str
    source: str
    cache_hit: bool
    took_ms: int
    error: str = ""
    grounded_block: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "normalized_query": self.normalized_query,
            "answer": self.answer,
            "citations": list(self.citations),
            "engine": self.engine,
            "source": self.source,
            "cache_hit": self.cache_hit,
            "took_ms": self.took_ms,
            "error": self.error,
            "grounded_block": self.grounded_block,
        }


def build_grounded_block(result: AutoSearchResult) -> str:
    """Render the result as a single prompt-insertable text block."""

    if result.error and not result.citations:
        return ""

    lines: list[str] = []
    label_bits: list[str] = []
    if result.engine:
        label_bits.append(f"engine: {result.engine}")
    if result.citations:
        label_bits.append(f"{len(result.citations)} sources")
    if result.cache_hit:
        label_bits.append("cache hit")
    label = ", ".join(label_bits) if label_bits else "no sources"
    lines.append(f"Grounded web search results ({label}):")

    if result.answer:
        lines.append(result.answer)
    else:
        lines.append("(no snippets returned by the search backend)")

    if result.citations:
        lines.append("")
        lines.append("Sources:")
        for idx, c in enumerate(result.citations):
            title = str(c.get("title") or "").strip()
            url = str(c.get("url") or "").strip()
            if title and url:
                lines.append(f"  [{idx + 1}] {title} — {url}")
            elif url:
                lines.append(f"  [{idx + 1}] {url}")
    return "\n".join(lines).strip()


async def run_auto_search(
    user_message: str,
    *,
    cfg: AppConfig,
    force: bool = False,
    bypass_cache: bool = False,
    request_timeout_sec: int | None = None,
    recent_user_messages: list[str] | None = None,
    rewrite_query: bool = True,
    provider_id: str | None = None,
    model_id: str | None = None,
) -> AutoSearchResult:
    """Resolve a user message to a grounded ``AutoSearchResult``.

    The router runs the heuristic, the cache, the MCP backend and the
    final ``build_grounded_block`` step.  When the policy is off and the
    user did not force a search, the function returns an empty result
    so callers can distinguish "did nothing" from "tried but failed".

    ``recent_user_messages`` is the chat history (most recent last) used
    to expand short follow-up queries ("the artist") into a
    context-rich search query ("tell me about Aposolix |
    the artist") so the backend isn't left guessing what the user
    meant.

    When ``rewrite_query`` is true (the default) and a provider/model is
    configured, a fast, dedicated LLM call (``query_rewriter``) is made
    to turn the raw user message + context into a *self-contained*
    search query.  This is what fixes "Force web search" being fed
    verbatim to the search backend: the rewriter sees the prior
    "Aposolix" mention and rewrites the query to "Aposolix artist".
    The cheap ``_resolve_search_query`` heuristic remains as a fallback
    when the rewriter fails or is disabled.
    """

    auto_cfg: AutoSearchConfig = cfg.mcp_config.auto_search
    decision = should_search(
        user_message,
        policy=auto_cfg.policy,
        enabled=auto_cfg.enabled,
        force=force,
        freshness_hints=auto_cfg.freshness_hints or None,
        factual_hints=auto_cfg.factual_hints or None,
        opinion_hints=auto_cfg.opinion_hints or None,
    )

    # Try the LLM-based rewriter first.  It understands pronouns,
    # meta-commands ("Force web search") and one-word follow-ups much
    # better than a regex concatenation can.  The cheap fallback runs
    # automatically inside ``rewrite_search_query_safe`` when the LLM
    # call is unavailable or returns nothing useful.
    resolved_query: str | None = None
    if rewrite_query and decision.should_search:
        try:
            from .query_rewriter import rewrite_search_query_safe

            resolved_query = await rewrite_search_query_safe(
                user_message,
                recent_user_messages=recent_user_messages,
                provider_id=provider_id,
                model_id=model_id,
                fallback_resolver=_resolve_search_query,
            )
        except Exception as exc:
            # Never let the rewriter break auto-search.
            logger.debug("run_auto_search: rewriter raised %s", exc)
            resolved_query = None

    if not resolved_query:
        resolved_query = _resolve_search_query(user_message, recent_user_messages)

    if resolved_query and resolved_query != user_message:
        decision = SearchDecision(
            should_search=decision.should_search,
            reason=decision.reason,
            policy=decision.policy,
            query=resolved_query,
            normalized_query=_normalize_query(resolved_query),
        )

    if not decision.should_search:
        empty = AutoSearchResult(
            query=decision.query,
            normalized_query=decision.normalized_query,
            answer="",
            citations=[],
            engine="",
            source="auto_search",
            cache_hit=False,
            took_ms=0,
            error="",
        )
        empty.grounded_block = ""
        return empty

    started = time.perf_counter()
    init_db()
    cache = AutoSearchCache(ttl_sec=auto_cfg.cache_ttl_sec)
    if not bypass_cache:
        hit = cache.get(decision.query)
        if hit is not None:
            took = int((time.perf_counter() - started) * 1000)
            answer = _format_answer(
                hit["citations"],
                max_chars=auto_cfg.summary_max_chars,
                snippet_chars=auto_cfg.snippet_per_source_chars,
                include_snippets=auto_cfg.include_snippets,
                include_full_content=auto_cfg.include_full_content,
            )
            res = AutoSearchResult(
                query=decision.query,
                normalized_query=decision.normalized_query,
                answer=answer,
                citations=hit["citations"],
                engine=hit.get("engine") or "",
                source=hit.get("source") or "auto_search",
                cache_hit=True,
                took_ms=took,
            )
            res.grounded_block = build_grounded_block(res)
            return res

    timeout = request_timeout_sec or max(15, cfg.mcp_config.native_web_search_timeout_sec * 2)
    raw = await _call_native_web_search(
        decision.query,
        cfg=auto_cfg,
        request_timeout_sec=timeout,
    )
    took = int((time.perf_counter() - started) * 1000)
    if not raw.get("ok"):
        # Tier-2 fallback: in-process direct multi-engine search
        # (Bing HTML, Wikipedia REST, arXiv, Startpage). Runs in
        # the same process and is independent of MCP plumbing.
        logger.info(
            "run_auto_search: MCP returned error=%s; falling back to direct_search",
            raw.get("error"),
        )
        fallback = await _fallback_direct_search(decision.query)
        if fallback.get("ok"):
            raw = fallback
        else:
            res = AutoSearchResult(
                query=decision.query,
                normalized_query=decision.normalized_query,
                answer="",
                citations=[],
                engine="",
                source="auto_search",
                cache_hit=False,
                took_ms=took,
                error=str(raw.get("error") or "unknown")
                + f";fallback:{fallback.get('error', '')}",
            )
            res.grounded_block = ""
            return res

    citations: list[dict[str, Any]] = list(raw.get("citations") or [])
    engine = str(raw.get("engine") or "")

    # Tier-2b: if the MCP path returned ``engine == "None"`` (the
    # known failure mode of the bundled web-search-mcp) or an empty
    # citation list, fall through to the in-process direct search
    # before giving up.  This is what stopped the orchestrator from
    # silently answering "I don't know" when the upstream MCP
    # returns 0 results.
    if not citations or engine.strip().lower() in {"", "none", "null"}:
        logger.info(
            "run_auto_search: MCP returned empty citations / engine=%r; falling back to direct_search",
            engine,
        )
        fallback = await _fallback_direct_search(decision.query)
        if fallback.get("ok"):
            raw = fallback
            citations = list(raw.get("citations") or [])
            engine = str(raw.get("engine") or "")

    for c in citations:
        if engine and not c.get("engine"):
            c["engine"] = engine

    answer = _format_answer(
        citations,
        max_chars=auto_cfg.summary_max_chars,
        snippet_chars=auto_cfg.snippet_per_source_chars,
        include_snippets=auto_cfg.include_snippets,
        include_full_content=auto_cfg.include_full_content,
    )

    if citations:
        try:
            cache.put(
                decision.query,
                answer=answer,
                citations=citations,
                engine=engine,
                source="auto_search",
            )
        except Exception:
            # Cache failures must never break the chat.
            pass

    res = AutoSearchResult(
        query=decision.query,
        normalized_query=decision.normalized_query,
        answer=answer,
        citations=citations,
        engine=engine,
        source="auto_search",
        cache_hit=False,
        took_ms=took,
    )
    res.grounded_block = build_grounded_block(res)
    return res


# ---------------------------------------------------------------------------
# Telemetry / audit log
# ---------------------------------------------------------------------------


def record_run(
    *,
    session_id: str,
    window_id: str,
    turn_id: str,
    result: AutoSearchResult,
    policy: str,
    trigger_reason: str,
) -> None:
    """Persist an audit row for the search we just performed (or skipped)."""

    from uuid import uuid4

    execute(
        """
        INSERT OR REPLACE INTO auto_search_runs (
            id, session_id, window_id, turn_id, query_text, policy,
            triggered, trigger_reason, cache_hit, engine, citations_json,
            answer_chars, took_ms, error_text, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid4()),
            session_id,
            window_id,
            turn_id,
            result.query,
            policy,
            1 if (result.citations or result.error) else 0,
            trigger_reason,
            1 if result.cache_hit else 0,
            result.engine,
            json.dumps(result.citations, ensure_ascii=False),
            len(result.answer or ""),
            int(result.took_ms or 0),
            result.error or "",
            utcnow_iso(),
        ),
    )


__all__ = [
    "AutoSearchCache",
    "AutoSearchConfig",
    "AutoSearchResult",
    "SearchDecision",
    "build_grounded_block",
    "record_run",
    "run_auto_search",
    "should_search",
]


# ---------------------------------------------------------------------------
# Tier-2 fallback: in-process direct web search
# ---------------------------------------------------------------------------
#
# The bundled ``mcp/web-search-mcp`` server has been observed to
# return ``engine: "None"`` / zero results reliably on the local
# Docker host: the upstream DuckDuckGo / Yahoo / Brave endpoints
# rate-limit the IP, the Playwright / Chromium binary is not
# bundled in the Docker image, and the Bing path is gated by a
# network policy that the operator cannot change.  Rather than
# ride the outage, ``run_auto_search`` falls through to
# ``direct_search.DirectSearchClient`` — a Python-only multi-engine
# client (Bing HTML, Wikipedia REST, arXiv, Startpage) that runs
# in the same process.  This block is intentionally async and
# short-circuits as soon as a non-empty result is found.


async def _fallback_direct_search(
    query: str,
    *,
    max_results: int = 5,
    timeout_sec: float = 12.0,
) -> dict[str, Any]:
    """Run the in-process multi-engine client and return a
    shape compatible with ``_call_native_web_search``.

    The shape mirrors the MCP contract: ``ok``, ``engine``,
    ``citations`` (list of dicts with title / url / snippet /
    source), and ``error`` (string)."""
    try:
        from .direct_search import DirectSearchClient
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "engine": "", "citations": [], "error": f"import:{exc}"}
    try:
        client = DirectSearchClient(timeout_sec=timeout_sec)
        resp = await client.search(query, max_results=max_results)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "engine": "", "citations": [], "error": str(exc)}
    if not resp.results:
        return {
            "ok": False,
            "engine": ",".join(resp.engines_used) if resp.engines_used else "",
            "citations": [],
            "error": f"no_results:{','.join(resp.errors) or 'empty'}",
        }
    citations = resp.to_citation_dicts()
    return {
        "ok": True,
        "engine": ",".join(resp.engines_used),
        "citations": citations,
        "error": "",
    }
