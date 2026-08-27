"""Search retry budget + engine switching for background runs.

The local small model sometimes calls ``full_web_search`` repeatedly
with similar queries and gets back ``0 results`` every time. In the
last session we saw 6+ calls in a row where the underlying engine
returned nothing — Brave rate-limited us, the engine briefly returned
``None``, or the query was just unanswerable. The model didn't notice
the pattern, so it kept burning turns and tokens on the same dead
end.

This module is a thin shim that:

* Tracks per-run empty-search results (``engine=None`` *or* text
  that matches the canonical "0 results" pattern).
* Caps the number of consecutive zero-result full_web_search calls
  per run at ``MAX_EMPTY_RESULTS_PER_RUN`` (default 3).
* On the next call after a streak of N empty results, transparently
  passes ``prefer_engine`` to the web-search MCP tool to try a
  different backend (Brave → DuckDuckGo → Bing). The shim rotates
  through ``FALLBACK_ENGINES`` so a rate-limited or down engine
  doesn't keep coming back.
* Once the cap is hit the shim returns a structured ``ok=False``
  payload with ``error_kind="search_exhausted"`` and a helpful
  human message so the model stops calling the tool and synthesises
  its final answer from what it already has.

The state lives in memory (per process) and is keyed by run_id. It
is intentionally cheap — the only thing we need to remember is the
last few ``(engine, returned_zero)`` pairs, plus a counter.
"""
from __future__ import annotations

import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


# Public knobs (the orchestrator can override them via constructor
# parameters if needed; these defaults are tuned for the local
# small-model run loop). Lowered from 3 to 2 in the post-mortem of
# the last session: the third call almost never produced new
# information once the engine had already returned 0 results
# twice, and 2 empty results is enough signal to short-circuit.
MAX_EMPTY_RESULTS_PER_RUN = 2
# Engines to try in order when the previous one returned no usable
# results. The list intentionally starts with Brave (the default)
# and proceeds to its replacements; rotation skips back to the
# start after we exhaust every option, so a longer run keeps
# cycling through all four.
FALLBACK_ENGINES: tuple[str, ...] = ("brave", "duckduckgo", "bing", "google")
# Engines that we trust enough to use as the *first* preference when
# we have no prior data on the run.
PREFERRED_INITIAL_ENGINE = "brave"

# Domains the local model historically wasted cycles following. The
# MCP ``get_web_search_summaries`` call returns these as "results"
# but they are search-engine UIs (Yahoo Suggestions, Bing Privacy
# Dashboard, UserVoice) and never carry the answer the user wants.
# Filtering them at the budget-shim layer saves the model from
# synthesising answers from noise.
_LOW_VALUE_DOMAINS: frozenset[str] = frozenset(
    {
        "yahoo.uservoice.com",
        "guce.yahoo.com",
        "search.yahoo.com",
        "uk.yahoo.com",
        "uservoice.com",
        "consent.yahoo.com",
        "bing.com",
        "www.bing.com",
        "privacy.microsoft.com",
        "go.microsoft.com",
    }
)


def looks_like_low_value_search_url(url: str) -> bool:
    """``True`` when a search result URL points at a search engine
    UI (Yahoo Suggestions, Bing privacy dashboard, …) that the
    orchestrator should not let the model synthesise an answer from.

    Used by ``_filter_low_value_results`` (which the run shim
    applies before counting a result as non-empty)."""
    if not url:
        return False
    lowered = url.lower()
    for marker in _LOW_VALUE_DOMAINS:
        if marker in lowered:
            return True
    return False


def filter_search_results_for_budget(
    results: list[Any],
) -> list[Any]:
    """Strip search-engine-UI results from a raw MCP result list so
    the budget tracker doesn't count a "5 results returned" payload
    as a hit when every result is just a yahoo.uservoice.com page.

    Each ``result`` is expected to expose ``.url`` (preferred) or
    fall back to ``["url"]``. Anything we can't parse is kept as-is
    so the model still sees it."""
    if not results:
        return results
    kept: list[Any] = []
    for item in results:
        url: str = ""
        if isinstance(item, dict):
            url = str(item.get("url") or item.get("link") or "")
        else:
            url = str(getattr(item, "url", "") or "")
        if url and looks_like_low_value_search_url(url):
            continue
        kept.append(item)
    return kept

# Canonical "the search returned nothing" markers — both the model
# output and the MCP tool result may include any of these.
_EMPTY_RESULT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"0\s+result(?:s)?(?:\s+requested)?(?:/0\s+obtained)?", re.IGNORECASE),
    re.compile(r"Search engine:\s*None\b", re.IGNORECASE),
    re.compile(r"\b0\s+obtained\b", re.IGNORECASE),
    re.compile(r"no results", re.IGNORECASE),
    re.compile(r"no citations found", re.IGNORECASE),
    re.compile(r"engine returned no usable results", re.IGNORECASE),
)


def looks_like_empty_search_result(text: str | None) -> bool:
    """``True`` when the tool result text clearly indicates that the
    search engine returned nothing useful.

    Used both by the orchestrator and by the run shim to decide
    whether to count a result as empty for budget purposes.
    """
    if not text:
        return True
    s = text.strip()
    if not s:
        return True
    return any(p.search(s) for p in _EMPTY_RESULT_PATTERNS)


# ---------------------------------------------------------------------------
# Per-run state
# ---------------------------------------------------------------------------


@dataclass
class _RunSearchState:
    run_id: str
    # Sliding window of recent results: list of (engine, was_empty).
    recent: deque[tuple[str, bool]] = field(default_factory=lambda: deque(maxlen=MAX_EMPTY_RESULTS_PER_RUN))
    # The next engine to use when the current one keeps returning
    # empty results. We rotate through FALLBACK_ENGINES regardless
    # of what the caller passed.
    next_engine_index: int = 0
    # Total empty results across the whole run (used to log the
    # pattern and to detect "we tried everything").
    total_empty: int = 0
    total_calls: int = 0
    # Lock so that concurrent tool invocations against the same run
    # don't race on the engine index.
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_exhausted_at: float | None = None


class SearchBudgetTracker:
    """Process-wide tracker for search retry budget.

    A single instance is held in module scope; callers pass the
    ``run_id`` so different runs don't interfere with each other.
    """

    def __init__(
        self,
        max_empty: int = MAX_EMPTY_RESULTS_PER_RUN,
        fallback_engines: tuple[str, ...] = FALLBACK_ENGINES,
    ) -> None:
        self._max_empty = max_empty
        self._fallback = fallback_engines
        self._states: dict[str, _RunSearchState] = {}
        self._lock = threading.Lock()

    # --- internal helpers ---------------------------------------------

    def _state(self, run_id: str) -> _RunSearchState:
        with self._lock:
            st = self._states.get(run_id)
            if st is None:
                st = _RunSearchState(run_id=run_id)
                self._states[run_id] = st
            return st

    def reset(self, run_id: str) -> None:
        with self._lock:
            self._states.pop(run_id, None)

    def _rotate_engine(self, st: _RunSearchState) -> str:
        """Pick the next engine in the fallback rotation, skipping
        the one we used on the most recent call. The rotation is
        driven by a per-state counter so debugging is reproducible
        (a given run always goes Brave → DuckDuckGo → Bing → Google
        regardless of the order of the calls)."""
        last = st.recent[-1][0] if st.recent else None
        # Walk forward through the fallback list, starting at the
        # counter, until we land on a different engine.
        for offset in range(len(self._fallback)):
            idx = (st.next_engine_index + offset) % len(self._fallback)
            candidate = self._fallback[idx]
            if candidate != last:
                st.next_engine_index = (idx + 1) % len(self._fallback)
                return candidate
        # Should not happen (fallback always has at least 2 items),
        # but fall back to the counter pick just in case.
        engine = self._fallback[st.next_engine_index % len(self._fallback)]
        st.next_engine_index = (st.next_engine_index + 1) % len(self._fallback)
        return engine

    # --- public API used by the run shim ------------------------------

    def plan_next_call(self, run_id: str, requested_args: dict[str, Any]) -> dict[str, Any]:
        """Return the (possibly mutated) tool args for the next call.

        Side effects:
        * If the previous ``max_empty`` results were all empty, this
          call sets ``prefer_engine`` to a fallback engine so we try
          a different backend.
        * If the budget is exhausted (>= max_empty consecutive empty
          results), this call also sets ``__search_budget_exhausted``=True
          in the returned dict so the calling shim can short-circuit
          before the MCP tool even runs.
        """
        st = self._state(run_id)
        with st.lock:
            st.total_calls += 1
            out = dict(requested_args)
            # Mirror the canonical "prefer_engine" field the web-search
            # MCP tool already understands. (Some servers spell it
            # ``engine`` instead; we set both to maximise compatibility.)
            consecutive_empty = 0
            for _, e in reversed(st.recent):
                if e:
                    consecutive_empty += 1
                else:
                    break
            if consecutive_empty >= self._max_empty:
                # Budget exhausted — don't even call the tool.
                out["__search_budget_exhausted"] = True
                st.last_exhausted_at = time.time()
                return out
            if consecutive_empty >= 1:
                # We've seen at least one empty result; rotate the
                # engine for the next call. ``_rotate_engine`` skips
                # the engine we just used so the rotation actually
                # changes the backend (Brave → DuckDuckGo → Bing).
                engine = self._rotate_engine(st)
                out["prefer_engine"] = engine
                out["engine"] = engine
            elif "prefer_engine" not in out and "engine" not in out:
                # First call on this run — seed with our preferred
                # default so different runs behave consistently.
                out["prefer_engine"] = PREFERRED_INITIAL_ENGINE
                out["engine"] = PREFERRED_INITIAL_ENGINE
            return out

    def record(self, run_id: str, engine: str, result_text: str | None) -> dict[str, Any]:
        """Record the outcome of a search call and return a small
        summary dict the caller can use for logging.

        Returns a dict with::

            {
                "consecutive_empty": int,
                "total_empty": int,
                "total_calls": int,
                "exhausted": bool,
                "last_engine": str,
            }

        ``consecutive_empty`` is the count of trailing empty results
        (from the back of the window) — not the sum across the whole
        window — so a non-empty result correctly resets the streak.
        """
        st = self._state(run_id)
        empty = looks_like_empty_search_result(result_text)
        with st.lock:
            st.recent.append((engine, empty))
            if empty:
                st.total_empty += 1
            # Count only the trailing empty results; a single
            # non-empty result resets the streak.
            consecutive_empty = 0
            for _, e in reversed(st.recent):
                if e:
                    consecutive_empty += 1
                else:
                    break
            exhausted = consecutive_empty >= self._max_empty
            return {
                "consecutive_empty": consecutive_empty,
                "total_empty": st.total_empty,
                "total_calls": st.total_calls,
                "exhausted": exhausted,
                "last_engine": engine,
            }

    def snapshot(self, run_id: str) -> dict[str, Any]:
        st = self._state(run_id)
        with st.lock:
            consecutive_empty = 0
            for _, e in reversed(st.recent):
                if e:
                    consecutive_empty += 1
                else:
                    break
            return {
                "consecutive_empty": consecutive_empty,
                "total_empty": st.total_empty,
                "total_calls": st.total_calls,
                "exhausted": consecutive_empty >= self._max_empty,
                "last_engine": st.recent[-1][0] if st.recent else None,
                "last_exhausted_at": st.last_exhausted_at,
            }


# Module-level singleton (cheap, no I/O). Tests can construct their
# own SearchBudgetTracker to keep state isolated.
_default_tracker = SearchBudgetTracker()


def get_default_tracker() -> SearchBudgetTracker:
    return _default_tracker


# ---------------------------------------------------------------------------
# Tool-result coercion
# ---------------------------------------------------------------------------


def exhausted_payload(query: str) -> dict[str, Any]:
    """Return the ``tool`` role payload for a budget-exhausted call.

    The model sees this as a clear "stop searching, summarise" signal
    instead of a generic ``Search engine: None`` line that it
    mistakenly interprets as "try again with a different query".
    """
    return {
        "ok": False,
        "error_kind": "search_exhausted",
        "error": (
            "Search budget exhausted: the last several full_web_search calls "
            "all returned 0 results or no engine. The current backend may be "
            "rate-limited, down, or the query may not have a good match. "
            "Do NOT call full_web_search again for this run. Synthesise your "
            "final answer from the facts you already have, or note the gap "
            "honestly in the summary."
        ),
        "query": query,
        "hint": "use the working_set of facts collected so far and finalise the report.",
    }


# Public re-exports
__all__ = [
    "SearchBudgetTracker",
    "exhausted_payload",
    "filter_search_results_for_budget",
    "get_default_tracker",
    "looks_like_empty_search_result",
    "looks_like_low_value_search_url",
    "MAX_EMPTY_RESULTS_PER_RUN",
    "FALLBACK_ENGINES",
]