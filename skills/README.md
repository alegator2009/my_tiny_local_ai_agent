# Skills System

A skills system for AI Infinite Session. Skills are reusable
instructions/prompts the agent can apply to tasks.

## Structure

```
skills/
├── wrapper.mjs              # MCP server (stdio, JSON-RPC) — registers each skill as an MCP tool
├── registry.json            # Skill storage (single source of truth)
├── registry.js              # CRUD operations over skills (Node API)
├── loader.js                # Skill activation: returns displayMessage + instructions
├── executor.js              # Skill execution: parse + plan + simulate
├── chat-display.js          # Chat message formatting
├── meta-skill-creator.js    # Create a new skill from a natural-language request
├── agent-integration.js     # Intent recognition: create skill / show list / execute
├── index.js                 # Public API (re-exports)
├── types.js                 # JSDoc typedefs
├── test-all.js              # 20 baseline tests
├── test-integration.js      # 20 integration tests
└── run-all-tests.js         # Runs both test suites
```

## How it works

1. **MCP wrapper (`wrapper.mjs`)** — a separate process that reads
   `registry.json` and registers every skill as an MCP tool under the
   name `mcp__skills_mcp__<skill-name>`. This hides the underlying
   code from the LLM; it only sees the description, instructions, and
   examples.

2. **Skills in the chat** — when the agent invokes
   `mcp__skills_mcp__weather_dnipro`, the backend sends an SSE event
   `tool_call_display` with the skill name, and the frontend renders
   it as `🔧 Using skill: weather-dnipro`.

3. **Creating a skill** — the agent can create a new skill in
   response to a user request via `meta-skill-creator.js` (parses the
   request, extracts name / whenToUse / instructions / examples, adds
   it to `registry.json`).

## Running tests

```bash
cd skills
node run-all-tests.js
```

## Adding a new skill manually

Edit `registry.json` and add an object under the `skills` field:

```json
{
  "name": "my-skill",
  "description": "Short description (the LLM sees this as the tool description)",
  "whenToUse": "When to apply this skill",
  "instructions": [
    "Step 1",
    "Step 2"
  ],
  "examples": [
    { "prompt": "example request", "action": "expected action" }
  ],
  "createdAt": "2026-08-28T00:00:00.000Z",
  "updatedAt": "2026-08-28T00:00:00.000Z"
}
```

After editing, restart the API container so the MCP server re-reads
the registry.

## List of current skills

See `registry.json` (the `skills` field) or run:

```bash
node -e "const r=require('./registry.js'); console.log(r.listSkills().map(s=>s.name).join(', '))"
```