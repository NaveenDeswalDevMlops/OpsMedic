# llmops/cache.py
"""Response cache for OpsMedic (LLMOps cost-optimisation / bonus criterion).

Identical requests to the same sub-task+model return the cached response
instead of re-running the model or re-calling the paid API. Backed by
SQLite so hits survive app restarts; stdlib only, unit-testable offline.

Key = sha256(subtask | model | canonicalised input). TTL-expired entries
are treated as misses and overwritten on the next set().
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    key        TEXT PRIMARY KEY,
    subtask    TEXT NOT NULL,
    model      TEXT NOT NULL,
    value      TEXT NOT NULL,          -- JSON-encoded response
    created_ts REAL NOT NULL,
    hits       INTEGER NOT NULL DEFAULT 0
);
"""


def _make_key(subtask: str, model: str, payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str)
    raw = f"{subtask}|{model}|{canonical}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class ResponseCache:
    def __init__(self, db_path: str | None = None, ttl_seconds: int = 3600) -> None:
        self.db_path = db_path or os.getenv("CACHE_DB_PATH", "./llmops_cache.db")
        self.ttl_seconds = ttl_seconds
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def get(self, subtask: str, model: str, payload: Any) -> Any | None:
        """Return the cached response or None. Increments the hit counter."""
        key = _make_key(subtask, model, payload)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value, created_ts FROM cache WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            if time.time() - row["created_ts"] > self.ttl_seconds:
                return None  # expired -> miss; overwritten by next set()
            conn.execute("UPDATE cache SET hits = hits + 1 WHERE key = ?", (key,))
        return json.loads(row["value"])

    def set(self, subtask: str, model: str, payload: Any, response: Any) -> None:
        key = _make_key(subtask, model, payload)
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO cache (key, subtask, model, value, created_ts, hits)
                   VALUES (?, ?, ?, ?, ?, 0)
                   ON CONFLICT(key) DO UPDATE SET
                     value = excluded.value, created_ts = excluded.created_ts""",
                (key, subtask, model, json.dumps(response, default=str), time.time()),
            )

    def stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS entries, COALESCE(SUM(hits), 0) AS total_hits"
                " FROM cache"
            ).fetchone()
        return {"entries": row["entries"], "total_hits": row["total_hits"]}

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM cache")
