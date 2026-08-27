"""Inspect one auto_search result to see the raw MCP response shape."""

import json
import httpx

r = httpx.post(
    "http://localhost:8000/api/settings/auto-search/test",
    json={"query": "Who is the CEO of Anthropic?", "force": True},
    timeout=120,
)
print(r.status_code)
print(json.dumps(r.json(), ensure_ascii=False, indent=2)[:2000])
