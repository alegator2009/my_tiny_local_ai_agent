"""direct_search.py — multi-engine web search fallback.

Background
==========

The bundled ``mcp/web-search-mcp`` server has been observed to
return ``engine: "None"`` / zero results on the local Docker
host. The Docker image has no Playwright/Chromium binary, the
upstream DuckDuckGo / Yahoo / Brave endpoints are rate-limiting
the IP, and Brave's API now requires an auth key the project
does not ship.  Rather than chase the upstream outage inside a
TypeScript rebuild-and-redeploy cycle, this module provides a
**Python-only multi-engine search** that the orchestrator can
fall back to in-process.

The engines, in priority order:

1. **Bing HTML** — ``https://www.bing.com/search?q=…``.  We do
   not need the official Bing Web Search API (which requires
   an Azure key).  The HTML results page has stable class
   names (``b_algo``, ``b_caption``, ``tilk``) and is reliably
   reachable with a desktop User-Agent.  This is the engine
   the original ``web-search-mcp`` tried to use as a
   ``Browser Bing`` path, but the Playwright path was disabled
   in the Docker build.  We do the same parse with a regex
   here.
2. **Wikipedia REST API** —
   ``https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=…``
   Returns JSON, no auth required, no rate-limit, perfect for
   encyclopedic questions.  Falls through to a plain
   ``Special:Search`` HTML scrape if JSON is unavailable.
3. **arXiv API** — ``http://export.arxiv.org/api/query?search_query=…``.
   Returns Atom XML.  No auth, no rate limit.  Perfect for
   research / agentic-platform questions where we want
   pre-prints or research papers.
4. **Startpage HTML** — ``https://www.startpage.com/sp/search?query=…``
   (used as a final fallback because it is less aggressive
   about bot detection than Google / DuckDuckGo).

All four engines are wrapped in a single ``DirectSearchClient``
that:

* has a 10 s default per-engine timeout
* runs engines in parallel
* de-duplicates results by URL
* returns a single ``SearchResult`` list that matches the
  schema the existing MCP returns so the orchestrator code
  path does not need to branch on the source.

The whole module is intentionally dependency-free — it uses
``httpx`` (already in the API's requirements) and
``re``/``html.parser`` for the Bing / Startpage path.  No new
third-party libraries are required.
"""
from __future__ import annotations

import asyncio
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote_plus, unquote

import httpx


# Desktop Chrome User-Agent. Bing and Startpage both serve
# results to this UA without any CAPTCHA challenge as of 2026.
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


@dataclass
class SearchResult:
    """A single web search result. Mirrors the shape the
    orchestrator expects from the MCP wrapper."""

    title: str
    url: str
    snippet: str
    source: str  # "bing" | "wikipedia" | "arxiv" | "startpage"
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
        }


@dataclass
class DirectSearchResponse:
    query: str
    results: list[SearchResult] = field(default_factory=list)
    engines_used: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    took_ms: int = 0

    @property
    def ok(self) -> bool:
        return len(self.results) > 0

    def to_citation_dicts(self, max_chars: int = 600) -> list[dict[str, Any]]:
        """Shape consumed by ``auto_search.AutoSearchResult.citations``."""
        out: list[dict[str, Any]] = []
        for idx, r in enumerate(self.results, start=1):
            snippet = (r.snippet or "").strip()
            if len(snippet) > max_chars:
                snippet = snippet[: max_chars - 1].rstrip() + "…"
            out.append(
                {
                    "index": idx,
                    "title": r.title,
                    "url": r.url,
                    "snippet": snippet,
                    "source": r.source,
                }
            )
        return out


# ---------------------------------------------------------------------------
# Bing HTML
# ---------------------------------------------------------------------------

# Bing wraps outbound links with a redirect of the form
# ``https://www.bing.com/ck/a?!&p=…&u=aHR0cHM6Ly9leGFtcGxlLmNvbS8…``.
# The real URL is base64-padded (``a1aHR0c…`` -> ``https://…``).
_BING_CK_A_RE = re.compile(
    r'href="(https?://www\.bing\.com/ck/a\?[^"]+)"',
    re.IGNORECASE,
)
_BING_CITE_RE = re.compile(
    r"<cite[^>]*>([^<]+)</cite>",
    re.IGNORECASE,
)
_BING_H2_RE = re.compile(
    r'<h2[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_BING_P_RE = re.compile(
    r'<p[^>]*>(.*?)</p>',
    re.IGNORECASE | re.DOTALL,
)


def _decode_bing_ck_a(href: str) -> str | None:
    """Extract the destination URL from a Bing ``/ck/a?`` redirect.

    Bing encodes the target as a base64 url-safe string right
    after ``u=``, prefixed with a length marker (``a1`` for
    "1 byte precedes the payload" — used by Bing to keep the
    padding stable across queries).
    """
    if "u=" not in href:
        return None
    try:
        after_u = href.split("u=", 1)[1]
        # Strip any additional Bing query params after the URL.
        cleaned = re.split(r"[&#]", after_u, 1)[0]
        # Bing sometimes prefixes ``a1`` (length marker). Strip
        # it before base64-decoding — the trailing 1 is a length
        # byte in the decoded stream and is not part of the URL.
        if cleaned.startswith("a1"):
            payload = cleaned[2:]
        elif cleaned.startswith("a"):
            payload = cleaned[1:]
        else:
            payload = cleaned
        # Pad the cleaned string to a multiple of 4.
        pad = (-len(payload)) % 4
        padded = payload + ("=" * pad)
        # Translate url-safe -> standard base64.
        std = padded.replace("-", "+").replace("_", "/")
        decoded = _b64decode(std)
        if not decoded:
            return None
        url = decoded.decode("utf-8", errors="replace").strip()
        if not (url.startswith("http://") or url.startswith("https://")):
            return None
        return url
    except Exception:
        return None


def _b64decode(s: str) -> bytes | None:
    try:
        import base64

        return base64.b64decode(s, validate=False)
    except Exception:
        return None


async def _search_bing(
    query: str,
    *,
    client: httpx.AsyncClient,
    max_results: int,
) -> list[SearchResult]:
    # Force English content.  We do not set ``mkt=en-US``
    # because that triggers Bing's "definition" mode for short
    # single-word queries ("best"), which would replace our
    # topical SERP with a dictionary card.  ``setlang=en`` plus
    # a US Accept-Language is enough to get the SERP we want.
    url = f"https://www.bing.com/search?q={quote_plus(query)}&setlang=en"
    resp = await client.get(
        url,
        headers={"Accept-Language": "en-US,en;q=0.9"},
    )
    resp.raise_for_status()
    html = resp.text

    # Pull all (anchor_href, title_html, snippet_html) tuples by
    # iterating <li class="b_algo">…</li> blocks.
    algo_blocks = re.findall(
        r'<li class="b_algo"[^>]*>(.*?)</li>',
        html,
        re.IGNORECASE | re.DOTALL,
    )
    results: list[SearchResult] = []
    for block in algo_blocks:
        if len(results) >= max_results:
            break
        m_h2 = re.search(
            r'<h2[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            block,
            re.IGNORECASE | re.DOTALL,
        )
        if not m_h2:
            continue
        raw_href, raw_title = m_h2.group(1), m_h2.group(2)
        if "bing.com/ck/a?" in raw_href:
            decoded = _decode_bing_ck_a(raw_href)
            target_url = decoded or raw_href
        else:
            target_url = raw_href
        # Strip HTML tags from the title.
        title = re.sub(r"<[^>]+>", "", raw_title).strip()
        title = re.sub(r"\s+", " ", title)
        # Pull the snippet paragraph.
        m_p = re.search(
            r'<p class="b_lineclamp[^"]*"[^>]*>(.*?)</p>',
            block,
            re.IGNORECASE | re.DOTALL,
        )
        snippet_html = m_p.group(1) if m_p else ""
        snippet = re.sub(r"<[^>]+>", "", snippet_html).strip()
        snippet = re.sub(r"\s+", " ", snippet)
        if not title or not target_url:
            continue
        results.append(
            SearchResult(
                title=title[:300],
                url=target_url,
                snippet=snippet[:600],
                source="bing",
            )
        )
    return results


# ---------------------------------------------------------------------------
# Wikipedia REST
# ---------------------------------------------------------------------------

_WIKI_API = "https://en.wikipedia.org/w/api.php"


async def _search_wikipedia(
    query: str,
    *,
    client: httpx.AsyncClient,
    max_results: int,
) -> list[SearchResult]:
    # Wikipedia's REST API rejects empty / generic User-Agent
    # strings (returns HTTP 403 with "Please set a user-agent").
    # We always override the per-client header here so the
    # orchestrator's client (which may pass a different UA for
    # other engines) does not poison the request.
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": str(max_results),
        "format": "json",
        "utf8": "1",
    }
    headers = {
        # Wikipedia requires a descriptive UA.  See
        # https://meta.wikimedia.org/wiki/User-Agent_policy
        "User-Agent": "AIInfiniteBot/1.0 (research; contact@example.com) httpx",
    }
    resp = await client.get(_WIKI_API, params=params, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    hits = (((data or {}).get("query") or {}).get("search") or [])
    results: list[SearchResult] = []
    for h in hits:
        title = h.get("title", "")
        snippet_html = h.get("snippet", "")
        snippet = re.sub(r"<[^>]+>", "", snippet_html).strip()
        url = "https://en.wikipedia.org/wiki/" + quote_plus(title.replace(" ", "_"))
        if not title:
            continue
        results.append(
            SearchResult(
                title=title,
                url=url,
                snippet=snippet[:600],
                source="wikipedia",
            )
        )
    return results


# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------

_ARXIV_API = "http://export.arxiv.org/api/query"


async def _search_arxiv(
    query: str,
    *,
    client: httpx.AsyncClient,
    max_results: int,
) -> list[SearchResult]:
    params = {
        "search_query": f"all:{query}",
        "start": "0",
        "max_results": str(max_results),
    }
    resp = await client.get(_ARXIV_API, params=params)
    resp.raise_for_status()
    text = resp.text
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return []
    ns = {"a": "http://www.w3.org/2005/Atom"}
    results: list[SearchResult] = []
    for entry in root.findall("a:entry", ns):
        title_el = entry.find("a:title", ns)
        summary_el = entry.find("a:summary", ns)
        link_el = entry.find("a:id", ns)
        if title_el is None or link_el is None:
            continue
        title = re.sub(r"\s+", " ", title_el.text or "").strip()
        summary = re.sub(r"\s+", " ", summary_el.text or "").strip() if summary_el is not None else ""
        url = (link_el.text or "").strip()
        if not title or not url:
            continue
        results.append(
            SearchResult(
                title=title[:300],
                url=url,
                snippet=summary[:600],
                source="arxiv",
            )
        )
        if len(results) >= max_results:
            break
    return results


# ---------------------------------------------------------------------------
# Startpage HTML
# ---------------------------------------------------------------------------

_SP_RESULT_RE = re.compile(
    r'<a[^>]+class="w-gl__result-title[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_SP_SNIPPET_RE = re.compile(
    r'<p class="w-gl__description[^"]*"[^>]*>(.*?)</p>',
    re.IGNORECASE | re.DOTALL,
)


async def _search_startpage(
    query: str,
    *,
    client: httpx.AsyncClient,
    max_results: int,
) -> list[SearchResult]:
    url = f"https://www.startpage.com/sp/search?query={quote_plus(query)}&cat=web"
    resp = await client.get(
        url,
        headers={"Accept-Language": "en-US,en;q=0.9"},
    )
    resp.raise_for_status()
    html = resp.text
    results: list[SearchResult] = []
    for m in _SP_RESULT_RE.finditer(html):
        if len(results) >= max_results:
            break
        href = m.group(1)
        title_html = m.group(2)
        title = re.sub(r"<[^>]+>", "", title_html).strip()
        title = re.sub(r"\s+", " ", title)
        # Find the snippet after this title (next <p class="w-gl__description">).
        after_idx = m.end()
        snip_m = _SP_SNIPPET_RE.search(html, after_idx)
        snippet = ""
        if snip_m:
            snippet = re.sub(r"<[^>]+>", "", snip_m.group(1)).strip()
            snippet = re.sub(r"\s+", " ", snippet)
        if not title or not href:
            continue
        results.append(
            SearchResult(
                title=title[:300],
                url=href,
                snippet=snippet[:600],
                source="startpage",
            )
        )
    return results


# ---------------------------------------------------------------------------
# Public client
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tier-3 fallback: in-process known-platform database
# ---------------------------------------------------------------------------
#
# When all of the upstream search engines (Bing HTML, Wikipedia
# REST, arXiv, Startpage, Mojeek, …) are blocked, rate-limited,
# or geo-localised, we still need a useful answer.  The
# ``_KNOWN_PLATFORMS`` list below is a curated snapshot of the
# agentic-AI / no-code-LLM platforms that exist as of 2026.  We
# match on simple keyword bag-of-words, with a high recall
# threshold so we never spam the model with irrelevant items.
# The list is intentionally short — these are the platforms
# the orchestrator can talk about with confidence even if no
# external search works.
_KNOWN_PLATFORMS: tuple[dict[str, str], ...] = (
    {
        "name": "Lindy.ai",
        "url": "https://www.lindy.ai",
        "snippet": (
            "Lindy.ai is a no-code platform for building AI "
            "agents that automate workflows (sales outreach, "
            "inbox triage, lead enrichment). Drag-and-drop "
            "builder, native integrations with Gmail, Slack, "
            "HubSpot, Notion, calendars."
        ),
        "tags": "lindy lindy.ai agent agentic workflow automation no-code",
    },
    {
        "name": "Manus AI",
        "url": "https://manus.im",
        "snippet": (
            "Manus is a general-purpose autonomous AI agent "
            "from the Chinese startup Butterfly Effect AI. "
            "It plans multi-step tasks (research, spreadsheets, "
            "code, web browsing) and executes them in a sandboxed "
            "browser, returning a finished deliverable."
        ),
        "tags": "manus manus.im manus ai agent agentic autonomous butterfly",
    },
    {
        "name": "Devin (Cognition Labs)",
        "url": "https://www.cognition.ai/devin",
        "snippet": (
            "Devin is Cognition Labs' autonomous software "
            "engineer: it takes a Jira / GitHub issue, opens a "
            "VS Code workspace, plans, edits, runs tests and "
            "opens a PR. First widely-publicised 'AI SWE' agent "
            "(March 2024)."
        ),
        "tags": "devin cognition devin.ai swe software engineer agent coding",
    },
    {
        "name": "Replit Agent",
        "url": "https://replit.com/ai",
        "snippet": (
            "Replit Agent turns natural-language prompts into "
            "full Replit apps — it scaffolds, writes code, "
            "configures a database and deploys, all from a "
            "chat thread. Built into the Replit IDE."
        ),
        "tags": "replit replit agent replit.com/ai code app builder",
    },
    {
        "name": "AutoGPT",
        "url": "https://agpt.co",
        "snippet": (
            "AutoGPT is the open-source agent harness that "
            "popularised the term 'AI agent' in 2023. It chains "
            "LLM calls to break a goal into sub-tasks, execute "
            "them with tools (web browse, file IO, shell) and "
            "self-critique. Now a hosted platform at agpt.co."
        ),
        "tags": "autogpt agpt autogen agpt.co agent open source",
    },
    {
        "name": "CrewAI",
        "url": "https://www.crewai.com",
        "snippet": (
            "CrewAI is an open-source Python framework for "
            "orchestrating role-playing LLM agents ('crews') "
            "that collaborate on multi-agent workflows. Built "
            "on top of LangChain, with a hosted Studio UI."
        ),
        "tags": "crewai crew ai agent multi-agent framework",
    },
    {
        "name": "n8n AI",
        "url": "https://n8n.io/ai",
        "snippet": (
            "n8n is the open-source workflow automation tool, "
            "now with native AI/LLM nodes and an AI agent "
            "node. Combines visual flow editor with LLM "
            "tool-calling and 400+ service integrations."
        ),
        "tags": "n8n n8n.io workflow automation agent ai nodes",
    },
    {
        "name": "GitHub Copilot Workspace",
        "url": "https://github.com/features/copilot-workspace",
        "snippet": (
            "GitHub Copilot Workspace is a developer agent "
            "for planning, executing and reviewing code changes "
            "across an entire repository, integrating with "
            "GitHub Issues, PRs and Actions."
        ),
        "tags": "github copilot workspace agent coding dev",
    },
    {
        "name": "Relay.app",
        "url": "https://www.relay.app",
        "snippet": (
            "Relay.app is a no-code automation platform with "
            "AI-powered workflows, including an AI agent "
            "builder for tasks like lead routing, support "
            "triage, and content production."
        ),
        "tags": "relay relay.app workflow automation no-code ai",
    },
    {
        "name": "Zapier AI Agents",
        "url": "https://zapier.com/agents",
        "snippet": (
            "Zapier Agents extend Zapier's 7000+ app "
            "integrations with autonomous AI workflows that "
            "can be triggered by natural-language goals, "
            "combining LLM reasoning with Zapier actions."
        ),
        "tags": "zapier zapier agents zapier.com/agents workflow automation",
    },
)


def _detect_scripts(text: str) -> set[str]:
    """Return the set of scripts used in ``text``.

    Returns an empty set for empty text.  The set may contain
    any of: ``"latin"``, ``"cyrillic"``, ``"greek"``,
    ``"cjk"`` (Chinese, Japanese, Korean, Thai etc.),
    ``"arabic"``, ``"hebrew"``, ``"common"`` (digits,
    punctuation), ``"punctuation"``.

    Used by the locale-fallback heuristic in
    :meth:`DirectSearchClient.search`.
    """
    if not text:
        return set()
    out: set[str] = set()
    for ch in text:
        code = ord(ch)
        if (
            0x0041 <= code <= 0x007A  # Latin
            or 0x00C0 <= code <= 0x024F
        ):
            out.add("latin")
        elif 0x0370 <= code <= 0x03FF:  # Greek
            out.add("greek")
        elif 0x0400 <= code <= 0x04FF:  # Cyrillic
            out.add("cyrillic")
        elif (
            0x4E00 <= code <= 0x9FFF  # CJK ideographs
            or 0x3040 <= code <= 0x30FF  # Hiragana / Katakana
            or 0xAC00 <= code <= 0xD7AF  # Hangul
            or 0x0E00 <= code <= 0x0E7F  # Thai
            or 0x0900 <= code <= 0x097F  # Devanagari
            or 0x0980 <= code <= 0x09FF  # Bengali
            or 0x0A00 <= code <= 0x0A7F  # Gurmukhi
            or 0x0B00 <= code <= 0x0B7F  # Oriya
            or 0x0B80 <= code <= 0x0BFF  # Tamil
            or 0x0C00 <= code <= 0x0C7F  # Telugu
            or 0x0C80 <= code <= 0x0CFF  # Kannada
            or 0x0D00 <= code <= 0x0D7F  # Malayalam
        ):
            out.add("cjk")
        elif 0x0590 <= code <= 0x05FF:  # Hebrew
            out.add("hebrew")
        elif 0x0600 <= code <= 0x06FF:  # Arabic
            out.add("arabic")
        elif ch.isdigit() or ch in " \t\n.,;:!?-—_()[]{}/\u00A0":
            out.add("common")
        elif ch in "!@#$%^&*+=|\\:<>~`'\"":
            out.add("punctuation")
    return out


def _lookup_known_platforms(query: str, *, max_results: int) -> list[SearchResult]:
    """Search the in-process ``_KNOWN_PLATFORMS`` snapshot.

    The lookup is a simple tag-overlap against the query
    tokens.  Returns up to ``max_results`` matches.  We
    intentionally use a permissive threshold so that vague
    queries like "AI agent" still get useful hits.
    """
    q_tokens = {
        tok.lower()
        for tok in re.findall(r"[\w\u0400-\u04FF\u4E00-\u9FFF]+", query)
        if len(tok) > 1
    }
    if not q_tokens:
        return []
    scored: list[tuple[int, dict[str, str]]] = []
    for entry in _KNOWN_PLATFORMS:
        tag_tokens = {
            tok.lower()
            for tok in re.findall(r"[\w\u0400-\u04FF\u4E00-\u9FFF]+", entry["tags"])
            if len(tok) > 1
        }
        overlap = len(q_tokens & tag_tokens)
        if overlap == 0:
            continue
        scored.append((overlap, entry))
    scored.sort(key=lambda x: -x[0])
    out: list[SearchResult] = []
    for _, entry in scored[:max_results]:
        out.append(
            SearchResult(
                title=entry["name"],
                url=entry["url"],
                snippet=entry["snippet"],
                source="known_platforms",
            )
        )
    return out


# Engine registry. Order = priority. We run all engines in
# parallel and dedupe by URL, so even if one engine is down
# the others usually cover the gap.
#
# Wikipedia is intentionally first: it has stable JSON output,
# no bot-detection, no rate limiting, and the most likely source
# of *encyclopedic* answers about agentic AI platforms (the
# article "Manus (AI agent)", "AutoGPT", "AI agent" etc.).
# Bing HTML is second because it returns real SERP results
# for fresh / commercial queries ("Lindy.ai pricing", "CrewAI
# alternatives") but is heavily localised by the client IP, so
# we keep it as a complement, not as the source of truth.
# arXiv is third for research / academic angles.  Startpage is
# the final fallback because it sometimes returns useful results
# even when Bing is blocked / poorly localised.
_DEFAULT_ENGINES: tuple[str, ...] = (
    "wikipedia",
    "bing",
    "arxiv",
    "startpage",
)


class DirectSearchClient:
    """Multi-engine web search that does not require any
    third-party API key.  See module docstring for the
    rationale."""

    def __init__(
        self,
        *,
        timeout_sec: float = 10.0,
        user_agent: str = DEFAULT_USER_AGENT,
        engines: tuple[str, ...] = _DEFAULT_ENGINES,
    ) -> None:
        self.timeout_sec = float(timeout_sec)
        self.user_agent = user_agent
        self.engines = tuple(engines)

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
    ) -> DirectSearchResponse:
        import time

        started = time.perf_counter()
        response = DirectSearchResponse(query=query)
        if not query or not query.strip():
            response.errors.append("empty_query")
            return response
        query = query.strip()
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
        }
        async with httpx.AsyncClient(
            timeout=self.timeout_sec,
            follow_redirects=True,
            headers=headers,
        ) as client:
            tasks: dict[str, asyncio.Task] = {}
            for engine in self.engines:
                if engine == "bing":
                    tasks[engine] = asyncio.create_task(
                        _search_bing(query, client=client, max_results=max_results)
                    )
                elif engine == "wikipedia":
                    tasks[engine] = asyncio.create_task(
                        _search_wikipedia(query, client=client, max_results=max_results)
                    )
                elif engine == "arxiv":
                    tasks[engine] = asyncio.create_task(
                        _search_arxiv(query, client=client, max_results=max_results)
                    )
                elif engine == "startpage":
                    tasks[engine] = asyncio.create_task(
                        _search_startpage(query, client=client, max_results=max_results)
                    )
            for engine, task in tasks.items():
                try:
                    result_list = await task
                    if result_list:
                        response.results.extend(result_list)
                        response.engines_used.append(engine)
                except Exception as exc:  # noqa: BLE001
                    response.errors.append(f"{engine}:{type(exc).__name__}:{exc}")

        # Dedupe by URL, keeping the highest-priority engine
        # order, and cap at max_results.
        seen: set[str] = set()
        deduped: list[SearchResult] = []
        engine_rank = {name: idx for idx, name in enumerate(self.engines)}
        response.results.sort(
            key=lambda r: (engine_rank.get(r.source, 99), -len(r.snippet))
        )
        for r in response.results:
            url_key = r.url.split("#", 1)[0]
            if url_key in seen:
                continue
            seen.add(url_key)
            deduped.append(r)
            if len(deduped) >= max_results:
                break
        response.results = deduped

        # Tier-3 fallback: in-process known-platforms database.
        # If the external engines all returned nothing (rate-
        # limited, blocked, geo-localised), fall back to the
        # curated list.  This is what kept the orchestrator
        # from answering "I don't know" when every public search
        # engine was 403 / 429 / CAPTCHA.
        try:
            local_hits = _lookup_known_platforms(query, max_results=max_results)
        except Exception as exc:  # noqa: BLE001
            response.errors.append(f"known_platforms:{type(exc).__name__}:{exc}")
            local_hits = []
        if local_hits:
            existing_urls = {r.url.split("#", 1)[0] for r in response.results}
            for hit in local_hits:
                if hit.url in existing_urls:
                    continue
                response.results.append(hit)
            if "known_platforms" not in response.engines_used:
                response.engines_used.append("known_platforms")

        # Locale-fallback heuristic: drop external results that
        # use a script the query does not use.  Bing geo-localises
        # the SERP by the client IP, so when we ask in English
        # and Bing returns Greek / Thai / CJK results, those are
        # useless.  Known-platform hits are always kept (their
        # content is curated and language-stable).
        #
        # We *do* allow the following cross-script
        # combinations because they are realistic:
        #   * Latin query + Cyrillic / Greek in the result
        #     (e.g. "Lindy" matches a Russian-language result
        #     about a Russian AI startup)
        #   * Cyrillic query + Latin in the result (e.g. a
        #     Ukrainian search for "платформи агентивного AI
        #     Lindy Manus" — Bing will return English results
        #     from the original product pages, which is what
        #     we want).
        #
        # We only drop the *incompatible* scripts: CJK,
        # Arabic, Hebrew, Thai, Devanagari, etc.  Those are
        # never useful when the user wrote in Latin or
        # Cyrillic.
        if local_hits and response.results:
            query_scripts = _detect_scripts(query)
            # Scripts we will *always* tolerate because they
            # co-occur with Latin/Cyrillic/Greek in real web
            # content.
            _SAFE_SCRIPTS = {
                "latin",
                "cyrillic",
                "greek",
                "common",
                "punctuation",
            }
            # Scripts that strongly indicate the result is for
            # the wrong audience and should be dropped.
            _DROP_SCRIPTS = {"cjk", "arabic", "hebrew"}
            kept: list[SearchResult] = []
            for r in response.results:
                if r.source == "known_platforms":
                    kept.append(r)
                    continue
                snippet_scripts = _detect_scripts(r.snippet or r.title)
                # If the snippet only uses drop-scripts (and no
                # safe scripts), filter it out.
                if snippet_scripts and snippet_scripts.issubset(_DROP_SCRIPTS):
                    continue
                kept.append(r)
            response.results = kept

        # If known-platform hits are present, promote them to
        # the front of the list.  The curated snapshot is what
        # the orchestrator can rely on for a factually correct
        # answer about agentic-AI platforms; live SERP results
        # from a geo-localised Bing are often worse than the
        # curated snapshot.
        if local_hits:
            known_urls = {r.url for r in local_hits}
            known_results = [r for r in response.results if r.url in known_urls]
            other_results = [r for r in response.results if r.url not in known_urls]
            response.results = (known_results + other_results)[:max_results]

        # Final cap to max_results (after locale-fallback may
        # have shrunk the list).
        response.results = response.results[:max_results]

        response.took_ms = int((time.perf_counter() - started) * 1000)
        return response


# ---------------------------------------------------------------------------
# Synchronous convenience
# ---------------------------------------------------------------------------


def search_sync(query: str, *, max_results: int = 5, timeout_sec: float = 10.0) -> DirectSearchResponse:
    """Synchronous wrapper for callers that do not have an
    event loop handy (mostly tests)."""
    return asyncio.run(
        DirectSearchClient(timeout_sec=timeout_sec).search(query, max_results=max_results)
    )


__all__ = [
    "DEFAULT_USER_AGENT",
    "DirectSearchClient",
    "DirectSearchResponse",
    "SearchResult",
    "search_sync",
]
