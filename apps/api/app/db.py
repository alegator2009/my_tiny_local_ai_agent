from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import settings


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db() -> None:
    Path(settings.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(settings.sqlite_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.executescript(SCHEMA_SQL)
        conn.commit()


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(settings.sqlite_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def fetch_all(query: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    with get_conn() as conn:
        cur = conn.execute(query, params)
        return cur.fetchall()


def fetch_one(query: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
    with get_conn() as conn:
        cur = conn.execute(query, params)
        return cur.fetchone()


def execute(query: str, params: tuple[Any, ...] = ()) -> None:
    with get_conn() as conn:
        conn.execute(query, params)


def execute_many(query: str, values: list[tuple[Any, ...]]) -> None:
    if not values:
        return
    with get_conn() as conn:
        conn.executemany(query, values)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  description TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  status TEXT NOT NULL,
  primary_model_config_id TEXT,
  summary_model_config_id TEXT,
  workspace_path TEXT,
  last_window_id TEXT,
  total_message_count INTEGER NOT NULL DEFAULT 0,
  total_token_count INTEGER NOT NULL DEFAULT 0,
  settings_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS windows (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  window_index INTEGER NOT NULL,
  started_at TEXT NOT NULL,
  closed_at TEXT,
  token_limit INTEGER NOT NULL,
  rollover_trigger_percent REAL NOT NULL,
  pre_rollover_started_at TEXT,
  hard_rollover_started_at TEXT,
  checkpoint_id TEXT,
  closing_reason TEXT,
  FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS messages (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  window_id TEXT NOT NULL,
  turn_id TEXT NOT NULL,
  role TEXT NOT NULL,
  timestamp TEXT NOT NULL,
  content_text TEXT NOT NULL,
  content_json TEXT NOT NULL DEFAULT '{}',
  token_count INTEGER NOT NULL DEFAULT 0,
  message_type TEXT NOT NULL,
  source TEXT NOT NULL DEFAULT 'chat',
  reply_to_message_id TEXT,
  tool_name TEXT,
  tool_call_id TEXT,
  status TEXT NOT NULL DEFAULT 'ok',
  is_pinned INTEGER NOT NULL DEFAULT 0,
  is_anchor INTEGER NOT NULL DEFAULT 0,
  FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
  FOREIGN KEY(window_id) REFERENCES windows(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chunks (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  window_id TEXT NOT NULL,
  chunk_index INTEGER NOT NULL,
  chunk_type TEXT NOT NULL,
  text TEXT NOT NULL,
  token_count INTEGER NOT NULL DEFAULT 0,
  embedding_ref TEXT,
  fts_doc_id INTEGER,
  start_message_id TEXT,
  end_message_id TEXT,
  recency_score REAL NOT NULL DEFAULT 0,
  importance_score REAL NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
  FOREIGN KEY(window_id) REFERENCES windows(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS checkpoints (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  source_window_id TEXT NOT NULL,
  checkpoint_index INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  summary_text TEXT NOT NULL,
  working_set_json TEXT NOT NULL,
  decisions_json TEXT NOT NULL,
  open_questions_json TEXT NOT NULL,
  constraints_json TEXT NOT NULL,
  artifacts_json TEXT NOT NULL,
  files_touched_json TEXT NOT NULL,
  retrieval_anchors_json TEXT NOT NULL,
  FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS facts (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  fact_type TEXT NOT NULL,
  subject TEXT NOT NULL,
  predicate TEXT NOT NULL,
  object TEXT NOT NULL,
  confidence REAL NOT NULL DEFAULT 0.5,
  source_chunk_id TEXT,
  is_durable INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS claim_sources (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  fact_id TEXT NOT NULL,
  source_type TEXT NOT NULL,
  source_ref_id TEXT NOT NULL,
  excerpt TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
  FOREIGN KEY(fact_id) REFERENCES facts(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS fact_conflicts (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  subject TEXT NOT NULL,
  predicate TEXT NOT NULL,
  fact_a_id TEXT NOT NULL,
  fact_b_id TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'open',
  explanation TEXT NOT NULL,
  first_detected_at TEXT NOT NULL,
  last_seen_at TEXT NOT NULL,
  FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
  FOREIGN KEY(fact_a_id) REFERENCES facts(id) ON DELETE CASCADE,
  FOREIGN KEY(fact_b_id) REFERENCES facts(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS decisions (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  title TEXT NOT NULL,
  decision_text TEXT NOT NULL,
  rationale TEXT,
  status TEXT NOT NULL,
  source_chunk_id TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  title TEXT NOT NULL,
  description TEXT,
  status TEXT NOT NULL,
  priority INTEGER NOT NULL DEFAULT 2,
  source_chunk_id TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS artifacts (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  path TEXT NOT NULL,
  artifact_type TEXT NOT NULL,
  title TEXT,
  description TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  source_message_id TEXT,
  FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS retrieval_logs (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  window_id TEXT NOT NULL,
  trigger_reason TEXT NOT NULL,
  query_text TEXT NOT NULL,
  query_type TEXT NOT NULL,
  filters_json TEXT NOT NULL,
  results_json TEXT NOT NULL,
  reranked_results_json TEXT NOT NULL,
  final_pack_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS lint_runs (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  timestamp TEXT NOT NULL,
  reason TEXT NOT NULL,
  summary_json TEXT NOT NULL,
  FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS workspace_events (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  window_id TEXT NOT NULL,
  event_type TEXT NOT NULL,
  timestamp TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  summary_text TEXT NOT NULL,
  FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  window_id TEXT,
  user_message_id TEXT,
  result_message_id TEXT,
  task_text TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  started_at TEXT,
  finished_at TEXT,
  progress_json TEXT NOT NULL DEFAULT '{}',
  error_text TEXT,
  FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
  FOREIGN KEY(window_id) REFERENCES windows(id) ON DELETE SET NULL,
  FOREIGN KEY(user_message_id) REFERENCES messages(id) ON DELETE SET NULL,
  FOREIGN KEY(result_message_id) REFERENCES messages(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS run_events (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  run_id TEXT NOT NULL,
  step_index INTEGER,
  event_type TEXT NOT NULL,
  title TEXT NOT NULL,
  detail TEXT NOT NULL,
  payload_json TEXT NOT NULL DEFAULT '{}',
  timestamp TEXT NOT NULL,
  FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
  FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_run_events_run_timestamp
ON run_events(run_id, timestamp ASC);

CREATE TABLE IF NOT EXISTS run_artifacts (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  run_id TEXT NOT NULL,
  artifact_id TEXT,
  step_index INTEGER,
  stage TEXT NOT NULL,
  title TEXT NOT NULL,
  path TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
  FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE,
  FOREIGN KEY(artifact_id) REFERENCES artifacts(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_run_artifacts_run_created
ON run_artifacts(run_id, created_at ASC);

CREATE INDEX IF NOT EXISTS idx_runs_session_status_created
ON runs(session_id, status, created_at DESC);

CREATE TABLE IF NOT EXISTS evolution_runs (
  id TEXT PRIMARY KEY,
  prompt TEXT NOT NULL,
  status TEXT NOT NULL,
  mode TEXT NOT NULL,
  max_generations INTEGER NOT NULL,
  stop_on_failure INTEGER NOT NULL DEFAULT 1,
  current_generation INTEGER NOT NULL DEFAULT 0,
  parent_generation INTEGER,
  child_generation INTEGER,
  lineage_root_path TEXT NOT NULL,
  parent_repo_path TEXT NOT NULL,
  child_repo_path TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  started_at TEXT,
  finished_at TEXT,
  progress_json TEXT NOT NULL DEFAULT '{}',
  score_json TEXT NOT NULL DEFAULT '{}',
  error_text TEXT
);

CREATE TABLE IF NOT EXISTS evolution_events (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  generation INTEGER,
  event_type TEXT NOT NULL,
  title TEXT NOT NULL,
  detail TEXT NOT NULL,
  payload_json TEXT NOT NULL DEFAULT '{}',
  timestamp TEXT NOT NULL,
  FOREIGN KEY(run_id) REFERENCES evolution_runs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_evolution_events_run_timestamp
ON evolution_events(run_id, timestamp ASC);

CREATE INDEX IF NOT EXISTS idx_evolution_runs_status_created
ON evolution_runs(status, created_at DESC);

-- Cache of "grounded web search" results produced by the auto-search
-- router.  Rows live for ``auto_search_cache_ttl_sec`` and are keyed by a
-- deterministic hash of the (normalised) query, so the same question
-- asked twice does not re-hit the MCP search backend.
CREATE TABLE IF NOT EXISTS auto_search_cache (
  cache_key TEXT PRIMARY KEY,
  query_text TEXT NOT NULL,
  answer_text TEXT NOT NULL,
  citations_json TEXT NOT NULL,
  engine TEXT NOT NULL DEFAULT '',
  source TEXT NOT NULL DEFAULT 'auto_search',
  created_at TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  last_used_at TEXT NOT NULL,
  hits INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_auto_search_cache_expires
ON auto_search_cache(expires_at);

-- Per-turn record of "the router decided to search" so the UI timeline
-- and post-hoc debugging can see when and why a search fired.
CREATE TABLE IF NOT EXISTS auto_search_runs (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  window_id TEXT NOT NULL,
  turn_id TEXT NOT NULL,
  query_text TEXT NOT NULL,
  policy TEXT NOT NULL,
  triggered INTEGER NOT NULL,
  trigger_reason TEXT NOT NULL,
  cache_hit INTEGER NOT NULL DEFAULT 0,
  engine TEXT NOT NULL DEFAULT '',
  citations_json TEXT NOT NULL DEFAULT '[]',
  answer_chars INTEGER NOT NULL DEFAULT 0,
  took_ms INTEGER NOT NULL DEFAULT 0,
  error_text TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_auto_search_runs_session_created
ON auto_search_runs(session_id, created_at DESC);

CREATE TABLE IF NOT EXISTS message_prefix_templates (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  prompt TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
  message_id UNINDEXED,
  session_id UNINDEXED,
  window_id UNINDEXED,
  role,
  text,
  timestamp UNINDEXED,
  tags
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  chunk_id UNINDEXED,
  session_id UNINDEXED,
  window_id UNINDEXED,
  chunk_type,
  text,
  created_at UNINDEXED
);
"""
