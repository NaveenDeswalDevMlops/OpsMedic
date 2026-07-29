# llmops/conversations.py
"""Conversation persistence for the OpsMedic chat UI.

Stores chat threads and their messages in SQLite so the sidebar can show
a ChatGPT-style history ("New chat" + past incidents grouped by day).
Stdlib only -> fully unit-testable offline.

Schema:
  conversations(id, title, created_ts, updated_ts)
  messages(id, conversation_id, role, content, meta_json, ts)
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from datetime import datetime, timedelta
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    created_ts  REAL NOT NULL,
    updated_ts  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role            TEXT NOT NULL,          -- 'user' | 'assistant'
    content         TEXT NOT NULL,
    meta_json       TEXT,                   -- assistant metadata (JSON)
    ts              REAL NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);
CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages (conversation_id, id);
"""


def make_title(text: str, max_len: int = 48) -> str:
    """Derive a short thread title from the first user message."""
    clean = " ".join(str(text).split())
    if len(clean) <= max_len:
        return clean or "New incident"
    return clean[:max_len].rsplit(" ", 1)[0] + "…"


def day_bucket(ts: float, now: float | None = None) -> str:
    """Group a timestamp into Today / Yesterday / Previous 7 days / Older."""
    now = now if now is not None else time.time()
    today = datetime.fromtimestamp(now).date()
    that = datetime.fromtimestamp(ts).date()
    delta = (today - that).days
    if delta <= 0:
        return "Today"
    if delta == 1:
        return "Yesterday"
    if delta <= 7:
        return "Previous 7 days"
    return "Older"


class ConversationStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        parent = os.path.dirname(os.path.abspath(db_path))
        os.makedirs(parent, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    # -- writes -------------------------------------------------------
    def create(self, title: str = "New incident") -> str:
        cid = uuid.uuid4().hex[:12]
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title, created_ts, updated_ts)"
                " VALUES (?, ?, ?, ?)",
                (cid, title, now, now),
            )
        return cid

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        meta: dict[str, Any] | None = None,
    ) -> None:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, meta_json, ts)"
                " VALUES (?, ?, ?, ?, ?)",
                (conversation_id, role, content,
                 json.dumps(meta) if meta else None, now),
            )
            conn.execute(
                "UPDATE conversations SET updated_ts = ? WHERE id = ?",
                (now, conversation_id),
            )

    def set_title(self, conversation_id: str, title: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE conversations SET title = ? WHERE id = ?",
                (title, conversation_id),
            )

    def rename_if_default(self, conversation_id: str, first_user_msg: str) -> None:
        """Give the thread a real title from its first message, once."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT title FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        if row and row["title"] in ("New incident", "New chat", ""):
            self.set_title(conversation_id, make_title(first_user_msg))

    def delete(self, conversation_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE conversation_id = ?",
                         (conversation_id,))
            conn.execute("DELETE FROM conversations WHERE id = ?",
                         (conversation_id,))

    # -- reads --------------------------------------------------------
    def list_conversations(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, created_ts, updated_ts FROM conversations"
                " ORDER BY updated_ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def grouped(self, limit: int = 50) -> list[tuple[str, list[dict[str, Any]]]]:
        """Conversations bucketed by day, newest bucket first.

        Returns [(bucket_label, [conversations...]), ...] preserving the
        Today > Yesterday > Previous 7 days > Older order.
        """
        order = ["Today", "Yesterday", "Previous 7 days", "Older"]
        buckets: dict[str, list[dict[str, Any]]] = {k: [] for k in order}
        for conv in self.list_conversations(limit):
            buckets[day_bucket(conv["updated_ts"])].append(conv)
        return [(label, buckets[label]) for label in order if buckets[label]]

    def get_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content, meta_json, ts FROM messages"
                " WHERE conversation_id = ? ORDER BY id ASC",
                (conversation_id,),
            ).fetchall()
        out = []
        for r in rows:
            out.append({
                "role": r["role"],
                "content": r["content"],
                "meta": json.loads(r["meta_json"]) if r["meta_json"] else None,
                "ts": r["ts"],
            })
        return out
