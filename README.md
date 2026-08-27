# AI Infinite Session

> A local-first chat agent that lets you work **comfortably and indefinitely**
> with language models that have a narrow context window — without losing
> track of the original prompt, the goal of the conversation, or important
> details from earlier turns.

[![Status](https://img.shields.io/badge/status-active%20development-yellow)](#status)
[![License](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](apps/api/pyproject.toml)
[![Next.js](https://img.shields.io/badge/next-14-black)](apps/web/package.json)
[![Docker](https://img.shields.io/badge/docker-compose-blue)](docker-compose.yml)

The full design rationale lives in [`IDEA.md`](./IDEA.md).  This README
focuses on **what the code does today** and **how to run it**.

---

## The problem

Small-context models forget the beginning of a long chat.  The standard
workaround — periodically asking the model to summarise the conversation —
eats details.  AI Infinite Session keeps the full transcript on disk in a
structured store and gives the model fast, hybrid retrieval over it, so
short-term summarisation and long-term memory work together.

## Highlights

- **Multi-window context with rollovers.**  A long-lived *session* is split
  into many short *context windows*.  When a window nears its token budget,
  the orchestrator creates a checkpoint, opens a fresh window, and seeds it
  with the previous checkpoint plus the most recent raw messages — keeping
  the handoff close to invisible to the user.
- **SKILL.state runtime.**  Per-session, in-process skill execution that
  replaces the append-only chat history with a bounded
  `(spec, state, observation)` bundle, anchored to a JSON memory file.
  Long-running tasks (research → write-up → review) get a *finite* token
  budget the model can actually reason about.  Inspired by
  [*SKILL.state: a bounded-context execution model for long-running
  LLM skills*](https://arxiv.org/abs/2608.26263) (arXiv:2608.26263).
- **Memory carry-over across turns.**  `durable_facts.json` is updated
  after every turn with bullet points, citations and bare URLs extracted
  from the assistant's answer.  The next turn's prompt bundle includes
  the same facts as `known_entities`, so follow-ups like *"give me the
  answer with those services"* do not loop back to a clarification
  request.
- **Three-tier auto web search.**  The orchestrator runs a cheap
  "google where I don't know" router *before* the model is asked to
  answer.  When the bundled MCP `web-search-mcp` returns zero results
  (rate-limited, blocked, geo-localised), the API falls through to a
  Python-only multi-engine client (Bing HTML → Wikipedia REST → arXiv →
  Startpage) and, if every external engine is down, to an in-process
  curated snapshot of common agentic-AI / no-code platforms.  The agent
  always has grounded citations to quote.
- **Stalled-session detector.**  When the assistant replies with three
  identical clarification requests in a row, the session is demoted to
  `status: "stalled"`; the very next user message flips it back to
  `active`.  The UI surfaces this state so the user can intervene
  before the model gets stuck in a loop.
- **Live MCP tool streaming.**  Connect any Model Context Protocol server
  over stdio and the agent will stream `tool_call` / `tool_result` cards
  to the UI as they happen — no buffering.  A bundled `web-search-mcp`
  is included for fresh / local information.
- **Hybrid retrieval over the full transcript.**  Every message is
  chunked and indexed in SQLite FTS5 (BM25) **and** LanceDB (semantic).
  Queries are fused, reranked by recency and importance, and injected
  into the prompt only when relevant.  If LanceDB isn't available, the
  orchestrator degrades gracefully to keyword-only search.
- **Built-in terminal and file tools.**  The agent can run shell
  commands in the session workspace and write downloadable file
  artifacts, both surfaced in the chat timeline.
- **Self-evolving repo (lineage).**  A bounded agent-evolution runner
  can iterate on the project itself: each generation lives under
  `evolution/agent-NNN/`, and `evolution/active.json` picks the one
  currently in use.  `scripts/active-web-dev.mjs` watches that file
  and hot-restarts the web app when it changes.

## Architecture at a glance

```
┌──────────────────────────────────────────────────────────────────┐
│                          apps/web  (Next.js 14)                  │
│   SessionList │ ChatPanel (SSE) │ StatusPanel │ LiveSessionGraph │
└─────────────────────────────┬────────────────────────────────────┘
                              │  SSE chat stream  + REST
┌─────────────────────────────▼────────────────────────────────────┐
│                       apps/api  (FastAPI)                        │
│                                                                  │
│   stream_chat (orchestrator)                                     │
│      │                                                            │
│      ├─► retrieval.run_retrieval   (BM25 + vectors + rerank)      │
│      ├─► memory.update_working_set + checkpoint on rollover       │
│      │   └─► record_turn_entities → durable_facts.json            │
│      ├─► mcp.MCPToolRegistry       (stdio MCP, schema cache)      │
│      ├─► auto_search.run_auto_search                              │
│      │   ├─► Tier 1: native-web-search MCP                        │
│      │   ├─► Tier 2: direct_search (Bing / Wiki / arXiv / SP)    │
│      │   └─► Tier 3: known_platforms snapshot                     │
│      ├─► skill_state.build_prompt_bundle   (per-skill `(s,s,o)`)  │
│      ├─► prompt.assemble_prompt    (with prompt cache)            │
│      │   └─► injects known_entities from durable_facts            │
│      └─► provider                  (OpenAI-compatible, stream)    │
│                                                                  │
│   Storage: SQLite + FTS5, LanceDB, ./data transcripts             │
└──────────────────────────────────────────────────────────────────┘
```

Key entry points:

| Path | What lives there |
| --- | --- |
| `apps/api/app/main.py` | FastAPI app, routes, startup hooks |
| `apps/api/app/services/orchestrator.py` | `stream_chat()` — the main turn loop, including the stalled-session detector |
| `apps/api/app/services/retrieval.py` | Hybrid search + rerank |
| `apps/api/app/services/memory.py` | Working set, durable facts, checkpoints, turn-entity extractor |
| `apps/api/app/services/skill_state.py` | SKILL.state runtime (`(spec, state, observation)` bundle) |
| `apps/api/app/services/mcp.py` | MCP registry, schema sanitiser, telemetry |
| `apps/api/app/services/auto_search.py` | Auto-search router: heuristic + cache + grounded block + Tier-2 fallback |
| `apps/api/app/services/direct_search.py` | In-process Bing / Wikipedia / arXiv / Startpage + curated snapshot |
| `apps/api/app/services/prompt.py` + `prompt_cache.py` | Prompt assembly with `known_entities` injection |
| `apps/api/app/services/vector_store.py` | LanceDB wrapper with safe fallback |
| `apps/api/app/evolution/` | Lineage runner |
| `apps/web/app/` | Next.js app router pages |
| `apps/web/components/ChatPanel.tsx` | Chat timeline + composer (incl. auto-search card) |
| `apps/web/components/LiveSessionGraph.tsx` | Session graph visualisation |
| `scripts/active-web-dev.mjs` | Dev launcher that follows `evolution/active.json` |
| `mcp/web-search-mcp/` | Bundled web-search MCP server (Tier-1 auto-search backend) |

## Quick start

### With Docker Compose (recommended)

```bash
git clone https://github.com/<your-org>/my_ai_agent.git
cd my_ai_agent
cp .env.example .env             # then edit, if needed
docker compose up --build
```

- Web UI: <http://localhost:3000>
- API docs: <http://localhost:8000/docs>

Data is persisted under `./data` (SQLite, LanceDB, transcripts).
Open the web UI → **Settings** and pick an OpenAI-compatible provider
(local `llama.cpp`, LM Studio, OpenAI, Groq, etc.).

### Local development (no Docker)

#### API

```bash
cd apps/api
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
uvicorn app.main:app --reload --port 8000
```

Add `.[vector]` to install LanceDB; without it the orchestrator falls
back to keyword-only retrieval.

#### Web

```bash
cd apps/web
pnpm install --frozen-lockfile
pnpm run dev
```

By default the web app talks to `http://localhost:8000` (override with
`NEXT_PUBLIC_API_URL`).

#### Viewing the active evolution generation

To run the web app against the currently active generation tracked by
`evolution/active.json`, use the root-level helper instead of
`apps/web`'s `dev` script:

```bash
pnpm run dev:web
```

It polls `evolution/active.json` and restarts Next.js automatically
when the active generation changes.

## Feature tour

### Auto web search — "Google where I don't know"

The orchestrator ships a tiny router that decides, *before* the model is
asked to answer, whether the user is asking for fresh, local or
fact-grounded information.  When the policy fires, the search backend
is consulted and the result is dropped into the per-turn prompt as a
**grounded block** — the model is told to quote it instead of guessing.

The router is three tiers deep:

1. **Tier 1 — `mcp/web-search-mcp`.**  Scrape Bing / Brave / DuckDuckGo
   with no API keys.  This is the path documented below; it's the one
   most users touch.
2. **Tier 2 — `direct_search.DirectSearchClient`.**  A Python-only
   multi-engine client (Bing HTML, Wikipedia REST, arXiv, Startpage)
   that runs in the same process.  Used when Tier 1 returns
   `engine: "None"` (rate-limited, blocked, geo-localised).  A
   script-based filter drops results that are in a script the query
   does not use, so Greek / Chinese SERPs do not flood Cyrillic /
   English queries.
3. **Tier 3 — `known_platforms` snapshot.**  An in-process curated list
   of common agentic-AI / no-code platforms (Lindy.ai, Manus, Devin,
   Replit Agent, AutoGPT, CrewAI, n8n AI, GitHub Copilot Workspace,
   Relay.app, Zapier Agents).  Always present in the response so the
   model has grounded citations even with no internet.

Toggle and tune the feature in **Settings → Auto web search**:

- **Гугли, где не знаешь** — the headline checkbox that turns the
  router on.
- **Policy** — `off` (never, but the model still has the raw MCP tool),
  `auto` (cheap regex-based decision per turn, recommended), or
  `always` (every user turn).
- **Summary max chars / Max citations / Snippet per source** — bound
  the prompt footprint.
- **Cache TTL (sec)** — repeat questions within the TTL are answered
  from the local cache without touching the search backend.
- **Prefer engine** — `auto` / `bing` / `brave` / `duckduckgo` if the
  bundled MCP supports the choice.
- **Test the router with a sample query** — a collapsible panel that
  hits `POST /api/settings/auto-search/test` and shows the decision
  plus the parsed citations.

Per-message overrides live in the chat composer:

- **Force web search** — overrides the global policy for the next
  message.
- **Bypass cache** — skip the local cache for the next message.

Auto-search events stream to the UI as a
`🔎 Auto-search: N source(s) (cache hit|fresh)` mini-status in the
chat header and as a `🔎 Auto-search (…)` system card listing each
citation with its URL.  Audit rows live in the `auto_search_runs`
SQLite table.

### SKILL.state — bounded context for long-running tasks

For tasks that don't fit in a single chat (multi-step research, code
review across many files, long-running agents), switch the session's
`context_mode` to `skill_state` (Settings → Session, or the per-session
dropdown).  The bundle shape — a stable *spec*, a mutable *state* and a
bounded ring of *observations* — is our implementation of the
SKILL.state runtime described in
[arXiv:2608.26263](https://arxiv.org/abs/2608.26263).  The orchestrator
will:

1. Load the skill's `spec` (instruction text) into a stable system
   section cached across turns.
2. Maintain a `state` (current step, working set, artefacts) in a
   per-session JSON file.
3. Append the latest `observation` (model output, tool results,
   user reply) to a bounded ring, evicting older turns as the ring
   fills.

The model never sees the full chat history; it sees a finite
`(spec, state, observation)` bundle that updates as the skill runs.
This is what makes the orchestrator *actually* work with sub-2B
quantised models — see [`IDEA.md`](./IDEA.md) for the full rationale.

### Memory carry-over across turns

Every assistant turn is post-processed by
`memory.record_turn_entities()`, which:

- Strips clarification text from `working_set.last_completed_step`
  (the model would otherwise poison the next turn's `current_subtask`).
- Extracts bullet points, `[N]` citation lines and bare URLs from
  both the assistant text *and* the search citations.
- Appends them to `durable_facts.json`, deduped against the existing
  list.

The next turn's prompt bundle (`prompt.assemble_prompt` for
`context_mode = "full"`, or `skill_state.build_prompt_bundle` for
SKILL.state) reads `durable_facts.json` and surfaces the top-N
entries as `known_entities`, so the model can quote them by name.

The exact format:

```json
[
  {
    "claim": "Lindy.ai is a no-code platform for building AI agents that automate workflows…",
    "source": "turn_text",
    "confidence": "high"
  },
  {
    "claim": "https://www.lindy.ai",
    "source": "turn_text",
    "confidence": "medium"
  }
]
```

### Stalled-session detector

When `orchestrator._maybe_mark_session_stalled()` notices that the
last three assistant turns are all clarifications, the session row in
SQLite is updated to `status = "stalled"` and a system message is
appended so the user sees what happened.  The very next user message
calls `_restore_stalled_session_if_needed()` and flips the status
back to `active`.  The UI honours the `stalled` status with a small
banner.

### Configuring an LLM provider

Open the web UI → **Settings** and fill in an OpenAI-compatible
endpoint:

- `base_url`, `endpoint`, `api_key`, `model_name`
- `context_window_size`, `temperature`, `top_p`, `max_output_tokens`
- Optional `extra_params_json` merged into every request

A separate summarisation model can be configured in the same panel.
Edits to `data/config.json` can be picked up live via
`POST /admin/reload`.

### MCP tools

In **Settings**:

1. Enable **MCP tools**.
2. Toggle **native web-search MCP** as needed.
3. Add servers JSON, for example:

   ```json
   [
     {
       "name": "filesystem",
       "command": "npx",
       "args": ["-y", "@modelcontextprotocol/server-filesystem", "/absolute/path"]
     }
   ]
   ```

4. Click **Discover MCP Tools** to list the registered tools.

The bundled web-search MCP server lives at `mcp/web-search-mcp/`.  Its
default path is `mcp/web-search-mcp/codex-wrapper.mjs`, which wraps
the server and redirects noisy logs to stderr.

## Repository layout

```
.
├── apps/
│   ├── api/             FastAPI backend
│   └── web/             Next.js frontend
├── packages/
│   └── schemas/         Shared frontend/backend TypeScript contracts
├── mcp/
│   └── web-search-mcp/  Bundled MCP web-search server (Bing / Brave / DDG scraper)
├── skills/              JS skill registry used during evolution
├── evolution/           Lineage: agent-001 … agent-NNN, plus active.json
├── scripts/             Dev helpers (active-web-dev.mjs, etc.)
├── data/                Runtime data (gitignored): SQLite, LanceDB, transcripts
├── docker-compose.yml
├── .env.example
├── IDEA.md              Design rationale and product spec
├── CHANGELOG.md
└── LICENSE
```

## Configuration reference

| Variable | Default | Purpose |
| --- | --- | --- |
| `AUTO_SEARCH_POLICY` | `auto` | `off` / `auto` / `always` |
| `AUTO_SEARCH_PREFER_DIRECT` | `1` | Skip the bundled MCP and call `direct_search.py` directly |
| `AUTO_SEARCH_CACHE_TTL_SEC` | `900` | Repeat questions within the TTL are answered from the cache |
| `NATIVE_WEB_SEARCH_TIMEOUT_SEC` | `45` | Per-engine timeout (s) for the `mcp/web-search-mcp` server |
| `LLM_BASE_URL` | — | OpenAI-compatible base URL |
| `LLM_MODEL` | — | Default model id |
| `LLM_CONTEXT_WINDOW` | — | Context window size (tokens) for the model above |
| `APP_ENABLE_VECTOR` | `1` | Set to `0` to disable LanceDB and use BM25-only retrieval |
| `APP_EVOLUTION_ENABLED` | `0` | Enable the self-evolution runner (research-grade) |

## Status

This is an actively-developed MVP.  Things that work end-to-end today:

- Sessions, context windows, soft and hard rollovers with checkpoints
- Hybrid retrieval with rerank and a graceful LanceDB-less fallback
- SKILL.state runtime with `(spec, state, observation)` bundles
- Three-tier auto web search with curated snapshot fallback
- Memory carry-over across turns (durable_facts → known_entities)
- Stalled-session detector with auto-recovery
- MCP registry with live tool streaming in the UI
- Terminal and file-write tools inside the session workspace
- Per-session thinking mode and message-prefix prompts
- Hot-reload of `data/config.json`
- Bounded lineage evolution runner

Known gaps / rough edges:

- No CI, no code coverage reports yet
- Embeddings are local heuristics; quality is intentionally simple
- The lineage runner is research-grade; expect to read
  `evolution/*/final-report.md` for context
- The bundled `mcp/web-search-mcp` is best-effort: it depends on
  upstream scrapers that occasionally change their markup or rate-limit
  the IP.  In those cases the API automatically falls through to the
  in-process `direct_search` client and the curated snapshot.

See [`CHANGELOG.md`](./CHANGELOG.md) for the running list of changes.

## Contributing

Pull requests are welcome.  For substantial changes, please open an
issue first to discuss the design.  Bug reports should include:

- A reproducible script or session id
- The model id and provider you were using
- The output of `POST /api/health` (it dumps a system snapshot
  without exposing any user data)

## Acknowledgements

The SKILL.state runtime shape (a stable *spec*, a mutable *state*, and a
bounded ring of *observations*) is our implementation of the
[*SKILL.state: a bounded-context execution model for long-running LLM
skills*](https://arxiv.org/abs/2608.26263) paper.  See
[`IDEA.md`](./IDEA.md) for the full design rationale.

The bundled `mcp/web-search-mcp` is a TypeScript port that scrapes
Bing / Brave / DuckDuckGo without API keys.

## License

[MIT](./LICENSE) — see `LICENSE` for the full text.
