# Changelog

All notable changes to AI Infinite Session are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **SKILL.state runtime** — per-session, in-process skill execution that
  replaces the append-only chat history with a bounded `(spec, state,
  observation)` bundle.  Re-uses the same `prompt_cache` so the model
  gets a stable per-skill system prefix.  Inspired by
  [arXiv:2608.26263](https://arxiv.org/abs/2608.26263).
- **Memory carry-over** — `known_entities` and `durable_facts` are
  surfaced to the model on every turn (both in SKILL.state and in the
  legacy `full` mode), so follow-up questions like *"give me the answer
  with those services"* no longer trigger a clarification loop.
- **`direct_search` Tier-2 fallback** — a Python-only multi-engine
  search client (Bing HTML, Wikipedia REST, arXiv, Startpage) that
  runs in-process when the bundled `mcp/web-search-mcp` returns zero
  results.  A curated `known_platforms` snapshot is appended as a
  Tier-3 fallback so the agent always has grounded citations for
  common agentic-AI / no-code queries, even with no internet.
- **Stalled-session detector** — `_maybe_mark_session_stalled()` in
  `orchestrator.py` flips `sessions.status` to `stalled` after three
  identical clarifications and back to `active` on the next user
  message.  The UI surfaces this state so users can intervene.
- **Memory-write hardening** — `working_set.last_completed_step` no
  longer records clarification text, and the durable-facts extractor
  pulls bullets, citation `[n]` lines, and bare URLs (with TLD
  whitelist) from both the assistant text and `grounded_citations`.
- **`mcp_request_timeout_sec` raised** for the native-web-search
  server (45 s default) to stop the orchestrator from aborting slow
  Bing/Brave responses mid-flight.
- **Bing URL parser** — properly decodes the `/ck/a?u=…` redirect
  format Bing uses for outbound links, so the model sees real
  destinations (agentic.ai, IBM, geeksforgeeks.org, …) instead of
  `bing.com/ck/a?…` stubs.
- **Locale-fallback heuristic** — drops external search hits that
  are written in a script the query does not use (CJK, Arabic,
  Hebrew, Thai, Devanagari).  Bing's geo-localisation is no longer
  able to flood a Cyrillic / English query with Greek / Chinese
  results.
- **Docker / build fixes** — `apps/web/Dockerfile` now uses
  `pnpm install --frozen-lockfile` (the repo only ships
  `pnpm-lock.yaml`); the compose file runs `pnpm run dev`.

### Changed
- `assemble_prompt()` now also surfaces durable facts in
  `context_mode = "full"` (not only in SKILL.state), so the legacy
  path benefits from the same carry-over behaviour.
- `_maybe_mark_session_stalled()` queries assistant turns only and
  takes them in chronological order, so trailing clarifications are
  detected even when the user typed in between.

### Fixed
- `_pick_web_search_tool_schema` no longer over-allocates the
  in-flight `search_budget` when the bundled MCP returns
  `engine: "None"`.
- `mcp/web-search-mcp`'s `low-value` URL filter excludes
  `yahoo.uservoice.com`, `privacy.microsoft.com`, etc. so the model
  stops citing feedback pages.

## [0.1.0] — 2025-08-27

### Added
- First public MVP.  Multi-window context, hybrid retrieval
  (BM25 + LanceDB), MCP registry, terminal/file tools, lineage
  evolution runner.
