"""Smoke test for the auto-search router running inside Docker.

Loads the live config, flips ``mcp_config.auto_search.enabled`` to True,
PUTs it back, then exercises ``POST /api/settings/auto-search/test``
with three sample queries (one freshness, one factual, one chitchat)
so we can confirm the heuristic + the search backend are wired up.
"""

from __future__ import annotations

import json
import sys
import time

import httpx

API = "http://localhost:8000"


def get_config() -> dict:
    r = httpx.get(f"{API}/api/settings", timeout=15)
    r.raise_for_status()
    return r.json()


def put_config(cfg: dict) -> dict:
    r = httpx.put(f"{API}/api/settings", json=cfg, timeout=15)
    r.raise_for_status()
    return r.json()


def call_test(query: str, force: bool = False, bypass_cache: bool = False) -> dict:
    r = httpx.post(
        f"{API}/api/settings/auto-search/test",
        json={"query": query, "force": force, "bypass_cache": bypass_cache},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def main() -> int:
    cfg = get_config()
    auto = cfg.setdefault("mcp_config", {}).setdefault("auto_search", {})
    was_enabled = auto.get("enabled", False)
    was_policy = auto.get("policy", "auto")
    auto["enabled"] = True
    auto["policy"] = "auto"
    put_config(cfg)
    print(f"auto_search.enabled: {was_enabled} -> True")
    print(f"auto_search.policy:  {was_policy} -> auto")
    print()

    samples = [
        ("Who is the current president of France?", False, False),  # factual
        ("Latest iPhone 16 Pro price in USD", False, False),  # freshness
        ("Hello, how are you?", False, False),  # chitchat (should be skipped)
        ("Tell me a joke about cats", True, False),  # forced even though chitchat
    ]

    for query, force, bypass in samples:
        print(f"--- query: {query!r} (force={force}, bypass_cache={bypass}) ---")
        t0 = time.time()
        try:
            res = call_test(query, force=force, bypass_cache=bypass)
        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            continue
        dt = (time.time() - t0) * 1000
        decision = res.get("decision", {})
        result = res.get("result", {})
        print(f"  decision: search={decision.get('should_search')} reason={decision.get('reason')} policy={decision.get('policy')}")
        if result:
            err = result.get("error") or ""
            cite_n = len(result.get("citations") or [])
            print(
                f"  result: cache_hit={result.get('cache_hit')} engine={result.get('engine') or '-'}"
                f" took_ms={result.get('took_ms')} citations={cite_n} error={err!r}"
            )
            for i, c in enumerate((result.get("citations") or [])[:3], start=1):
                print(f"    [{i}] {c.get('title','')[:60]!s:60s} | {c.get('url','')[:60]!s}")
        else:
            print("  result: <skipped>")
        print(f"  wall: {dt:.0f}ms")
        print()

    # Restore
    cfg = get_config()
    cfg["mcp_config"]["auto_search"]["enabled"] = was_enabled
    cfg["mcp_config"]["auto_search"]["policy"] = was_policy
    put_config(cfg)
    print(f"Restored auto_search to enabled={was_enabled}, policy={was_policy}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
