# Project Idea

The goal is an agent that lets you work comfortably and indefinitely with models that have a
very narrow context window (for example ~130K tokens) **without** losing the chat context,
abandoning the task, or hallucinating.

## The standard problem with a narrow-window agent

The first thing to go is the context from the start of the chat — most importantly the original
prompt and the goal of the conversation.

## The standard workaround: summarising the context

When the context window is almost full, the entire chat is handed to a *separate* model session
with the task of compressing and summarising everything that was said. The model reduces the
information. The original session — now sitting at ~100% of its window — is closed; a new,
clean session is opened and given the summary as its initial prompt. The model continues working
from there.

**The core problem:** the compression eats detail.

## The proposed solution

Take the summarisation approach, but — in addition to the compressed context — store the
**entire previous chat** in a separate store, and give the model a way to **quickly search** it
when it needs to "remember" something in more detail.

### Trade-offs

The user has to be OK with large files on disk and possibly some extra latency when search
across the chat history is slow.

### How it works in detail

1. Start with a small model that has a small context window. Give it access to MCP so it can
   extend its knowledge when needed (for example, fresh web search results).
2. To maximise the chance that the context fits in the window, each long session should focus
   on **one specific topic** (e.g. engineering, cooking, philosophy). The user creates a
   session — say about engineering — and a folder is allocated for it. Everything the system
   does for that session lives inside that folder.
3. The user writes the initial prompt and the chat with the model begins (**session 1**).
4. When a configurable percentage of the context window is filled, work is paused. The entire
   chat is handed to the **same model** (but with an empty context — a service session) and
   asked to **summarise** the conversation (**summarised context**).
5. The full chat is also saved into the store (raw `.md` files, SQLite/Postgres for metadata,
   Qdrant for retrieval) and indexed for fast lookup (TBD: how to index so the model can find
   the right slice quickly). If this is not the first context exhaustion in the session, the
   new data is **appended** to what is already there — the store holds the entire session
   from the very beginning.
6. In session 1, a **new agent** is launched with an empty context window. As its initial
   prompt it receives:
   - the **summarised context**,
   - the most recent **raw** messages from the previous window, so the agent knows where it
     stopped and can keep going (this is **critical** — the user must barely feel that the
     context was summarised; the chat should continue almost seamlessly),
   - information that the agent has the full chat in the store and how to search it on demand,
   - information about the connected MCP servers.
7. The chat continues seamlessly. When the fill threshold is reached again, jump back to
   step 4 — and so on, indefinitely.

### Options

- Give every session a system-level initial prompt describing how it should behave, work, and
  answer (system-supplied).
- Across all sessions, share **one memory file** that says who they are, who the user is, and so
  on (user-supplied; the agents themselves can extend it).

### Open problems: storage and search

The agent has to know what information is in the store and navigate it quickly. We still have to
research the available solutions — graphs, tags, etc.

## Interface

A single web page with a chat and a settings panel.

### Settings

- **Primary model** — host, route (OpenAI format), API key if needed, model name, context window
  size (auto-detected from the model if possible, otherwise set manually).
- **Summarisation model** — same as the primary by default, but configurable independently
  (host, route, API key, model name, context window size).
- **Storage and indexing settings**, if needed (to be refined once the storage layer is fixed).
- **Context fill percentage** that triggers the rollover into a new window.
- **Session management** — delete, clear, etc.
- **System prompt** at the start of a session.
- **MCP connections**.

### Chat

- Switch between sessions.
- Create a new session (name it and switch to it).
- Visualise what the system is doing so I can see whether it is "thinking" or has crashed.
- Visualise when context summarisation for a new window begins.
- Visualise what the model is doing — thinking, using MCP, reading from the session-wide memory.

The model has to be able to work with the filesystem (a folder is provided at the start of a
session) — the implementation should support more than just chatting; it should also support
running commands in the terminal, coding, generating images, and so on.