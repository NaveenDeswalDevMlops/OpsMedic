# llmops/metrics.py
"""LLMOps metrics collection for OpsMedic.

Every sub-task call is logged as one row in a SQLite table. The module
is standalone (stdlib only) so it is unit-testable without Streamlit,
transformers, or network access.

Metrics captured per request (>=5 required by the assignment):
  1. latency_ms            - wall-clock time of the call
  2. tokens_in / tokens_out- exact (if the API reports usage) or estimated
  3. cost_usd              - estimated from a per-model price table
  4. status / error        - success vs failure => error rate
  5. cache_hit             - whether the response came from cache
Aggregations exposed for the dashboard: p50/p95 latency, throughput
(requests per minute), error rate, cache-hit rate, total cost, token
totals, and per-subtask breakdowns.

Usage:
    logger = MetricsLogger(db_path)
    with logger.track(subtask="summarize", model="distilbart") as rec:
        out = model(...)                     # do the work
        rec.set_tokens(tokens_in=120, tokens_out=45)   # optional, exact
        rec.mark_cache_hit()                 # optional
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

# Default price table (USD per 1M tokens). src/config.py overrides this
# via MetricsLogger(price_table=...). Local HF models cost 0.
_DEFAULT_PRICES: dict[str, dict[str, float]] = {
    "llama-3.3-70b-versatile": {"in": 0.59, "out": 0.79},
    "llama-3.1-8b-instant": {"in": 0.05, "out": 0.08},
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS metrics (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL    NOT NULL,           -- unix epoch seconds
    subtask     TEXT    NOT NULL,           -- e.g. 'retrieve', 'resolve', 'asr'
    model       TEXT    NOT NULL,
    latency_ms  REAL    NOT NULL,
    tokens_in   INTEGER NOT NULL DEFAULT 0,
    tokens_out  INTEGER NOT NULL DEFAULT 0,
    cost_usd    REAL    NOT NULL DEFAULT 0.0,
    status      TEXT    NOT NULL,           -- 'ok' | 'error'
    cache_hit   INTEGER NOT NULL DEFAULT 0, -- 0/1
    error       TEXT,                       -- exception text when status='error'
    extra       TEXT                        -- JSON blob for anything else
);
CREATE INDEX IF NOT EXISTS idx_metrics_ts      ON metrics (ts);
CREATE INDEX IF NOT EXISTS idx_metrics_subtask ON metrics (subtask);
"""


def approx_tokens(text: str) -> int:
    """Rough token estimate when the backend does not report usage.

    English text averages ~4 characters per token for BPE vocabularies;
    this is accurate enough for cost *estimates* and clearly labelled
    as an estimate on the dashboard.
    """
    if not text:
        return 0
    return max(1, round(len(text) / 4))


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (no numpy dependency).

    pct in [0, 100]. Returns 0.0 for an empty list.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if pct <= 0:
        return ordered[0]
    if pct >= 100:
        return ordered[-1]
    rank = math.ceil(pct / 100.0 * len(ordered))
    return ordered[max(0, rank - 1)]


@dataclass
class _Record:
    """Mutable record handed to the caller inside `track()`."""

    subtask: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    cache_hit: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def set_tokens(self, tokens_in: int = 0, tokens_out: int = 0) -> None:
        self.tokens_in = int(tokens_in)
        self.tokens_out = int(tokens_out)

    def estimate_tokens(self, input_text: str = "", output_text: str = "") -> None:
        self.tokens_in = approx_tokens(input_text)
        self.tokens_out = approx_tokens(output_text)
        self.extra["tokens_estimated"] = True

    def mark_cache_hit(self) -> None:
        self.cache_hit = True

    def add_extra(self, **kwargs: Any) -> None:
        self.extra.update(kwargs)


class MetricsLogger:
    """SQLite-backed metrics store. One instance per app process."""

    def __init__(
        self,
        db_path: str | None = None,
        price_table: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.db_path = db_path or os.getenv("METRICS_DB_PATH", "./llmops_metrics.db")
        self.price_table = price_table or _DEFAULT_PRICES
        self._init_db()

    # -- internals ----------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        prices = self.price_table.get(model)
        if not prices:
            return 0.0  # local / unknown models are free
        return (
            tokens_in * prices.get("in", 0.0) + tokens_out * prices.get("out", 0.0)
        ) / 1_000_000.0

    # -- write path ---------------------------------------------------
    @contextmanager
    def track(self, subtask: str, model: str,
              latency_ms: float | None = None) -> Iterator[_Record]:
        """Context manager: times the block and writes one metrics row.

        Exceptions inside the block are logged as status='error' and
        re-raised, so callers keep their normal error handling.

        `latency_ms` overrides the measured duration. Streaming callers
        need this: they can only write their row once the stream is
        exhausted, so the block being timed is just bookkeeping and the
        real generation time has to be passed in. Without it, streamed
        latency was recorded as ~0 ms.
        """
        rec = _Record(subtask=subtask, model=model)
        start = time.perf_counter()
        status, error_text = "ok", None
        try:
            yield rec
        except Exception as exc:  # noqa: BLE001 - we log then re-raise
            status, error_text = "error", f"{type(exc).__name__}: {exc}"
            raise
        finally:
            measured = (time.perf_counter() - start) * 1000.0
            self._write(
                rec,
                measured if latency_ms is None else float(latency_ms),
                status,
                error_text,
            )

    def _write(
        self, rec: _Record, latency_ms: float, status: str, error_text: str | None
    ) -> None:
        cost = self._cost(rec.model, rec.tokens_in, rec.tokens_out)
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO metrics
                   (ts, subtask, model, latency_ms, tokens_in, tokens_out,
                    cost_usd, status, cache_hit, error, extra)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    time.time(),
                    rec.subtask,
                    rec.model,
                    round(latency_ms, 3),
                    rec.tokens_in,
                    rec.tokens_out,
                    cost,
                    status,
                    1 if rec.cache_hit else 0,
                    error_text,
                    json.dumps(rec.extra) if rec.extra else None,
                ),
            )

    # -- read path (dashboard) -----------------------------------------
    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM metrics ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def summary(self, window_seconds: float | None = None) -> dict[str, Any]:
        """Global aggregates, optionally over a trailing time window."""
        where, params = "", []
        if window_seconds:
            where = "WHERE ts >= ?"
            params = [time.time() - window_seconds]
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT latency_ms, tokens_in, tokens_out, cost_usd, status,"
                f" cache_hit, ts FROM metrics {where}",
                params,
            ).fetchall()
        if not rows:
            return {
                "requests": 0, "latency_p50_ms": 0.0, "latency_p95_ms": 0.0,
                "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0,
                "error_rate": 0.0, "cache_hit_rate": 0.0,
                "throughput_rpm": 0.0,
            }
        latencies = [r["latency_ms"] for r in rows]
        errors = sum(1 for r in rows if r["status"] == "error")
        hits = sum(r["cache_hit"] for r in rows)
        span = max(r["ts"] for r in rows) - min(r["ts"] for r in rows)
        rpm = len(rows) / (span / 60.0) if span > 0 else float(len(rows))
        return {
            "requests": len(rows),
            "latency_p50_ms": round(percentile(latencies, 50), 2),
            "latency_p95_ms": round(percentile(latencies, 95), 2),
            "tokens_in": sum(r["tokens_in"] for r in rows),
            "tokens_out": sum(r["tokens_out"] for r in rows),
            "cost_usd": round(sum(r["cost_usd"] for r in rows), 6),
            "error_rate": round(errors / len(rows), 4),
            "cache_hit_rate": round(hits / len(rows), 4),
            "throughput_rpm": round(rpm, 2),
        }

    def summary_by_subtask(self) -> list[dict[str, Any]]:
        """Per-subtask aggregates for the dashboard table."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT subtask, latency_ms, tokens_in, tokens_out, cost_usd,"
                " status, cache_hit FROM metrics"
            ).fetchall()
        by: dict[str, list[sqlite3.Row]] = {}
        for r in rows:
            by.setdefault(r["subtask"], []).append(r)
        out: list[dict[str, Any]] = []
        for subtask, grp in sorted(by.items()):
            lat = [g["latency_ms"] for g in grp]
            errs = sum(1 for g in grp if g["status"] == "error")
            hits = sum(g["cache_hit"] for g in grp)
            out.append({
                "subtask": subtask,
                "requests": len(grp),
                "latency_p50_ms": round(percentile(lat, 50), 2),
                "latency_p95_ms": round(percentile(lat, 95), 2),
                "tokens_in": sum(g["tokens_in"] for g in grp),
                "tokens_out": sum(g["tokens_out"] for g in grp),
                "cost_usd": round(sum(g["cost_usd"] for g in grp), 6),
                "error_rate": round(errs / len(grp), 4),
                "cache_hit_rate": round(hits / len(grp), 4),
            })
        return out
