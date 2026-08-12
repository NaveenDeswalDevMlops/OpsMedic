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
    # Set only by streamed callers via set_latency_ms() / mark_error().
    latency_override: float | None = None
    status_override: str | None = None
    error_override: str | None = None

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

    def set_latency_ms(self, latency_ms: float) -> None:
        """Override the measured wall-clock time for this row.

        Required for streamed calls: the generator is fully drained
        before `track()` is entered, so the context manager would
        otherwise time an empty block and record ~0 ms. The caller
        starts its own timer at first byte and passes real elapsed here.
        """
        self.latency_override = max(0.0, float(latency_ms))

    def mark_error(self, error_text: str) -> None:
        """Record a failure the caller caught instead of raising.

        A streamed call swallows its exception so partial output can
        still be shown. Without this the row is written status='ok' and
        the error-rate metric under-reports every streaming failure.
        """
        self.status_override = "error"
        self.error_override = str(error_text)


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
    def track(self, subtask: str, model: str) -> Iterator[_Record]:
        """Context manager: times the block and writes one metrics row.

        Exceptions inside the block are logged as status='error' and
        re-raised, so callers keep their normal error handling.
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
            latency_ms = (time.perf_counter() - start) * 1000.0
            if rec.latency_override is not None:
                latency_ms = rec.latency_override
            if rec.status_override is not None and status == "ok":
                status = rec.status_override
                error_text = rec.error_override
            self._write(rec, latency_ms, status, error_text)

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
                "latency_p99_ms": 0.0,
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
            "latency_p99_ms": round(percentile(latencies, 99), 2),
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

    # -- read path (Golden Signals tab) ---------------------------------
    #: fields the sub-tasks report via BaseSubTask.report_signals() and
    #: which this module knows how to aggregate. Each entry is
    #: (extra_key, aggregation) where aggregation is "mean" | "rate".
    SIGNAL_FIELDS: tuple[tuple[str, str], ...] = (
        ("retrieval_top_score", "mean"),
        ("retrieval_mean_score", "mean"),
        ("evidence_count", "mean"),
        ("no_evidence", "rate"),
        ("classifier_confidence", "mean"),
        ("low_confidence", "rate"),
        ("asr_audio_s", "mean"),
        ("asr_rtf", "mean"),
        ("tts_fallback", "rate"),
        ("compression_ratio", "mean"),
        ("ttft_ms", "mean"),
        ("tokens_per_sec", "mean"),
    )

    def _rows_in_window(
        self, window_seconds: float | None = None
    ) -> list[dict[str, Any]]:
        where, params = "", []
        if window_seconds:
            where = "WHERE ts >= ?"
            params = [time.time() - window_seconds]
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT ts, subtask, model, latency_ms, tokens_in, tokens_out,"
                f" cost_usd, status, cache_hit, extra FROM metrics {where}"
                f" ORDER BY ts",
                params,
            ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            try:
                d["extra_parsed"] = json.loads(d["extra"]) if d["extra"] else {}
            except (ValueError, TypeError):
                d["extra_parsed"] = {}
            out.append(d)
        return out

    def timeseries(
        self,
        bucket_seconds: float = 60.0,
        window_seconds: float | None = 3600.0,
    ) -> list[dict[str, Any]]:
        """Bucketed traffic + latency, one dict per (bucket, subtask).

        Feeds the stacked-area traffic chart and the sub-task x time
        latency heatmap. Buckets are aligned to the epoch so repeated
        calls return stable x-axis values.
        """
        rows = self._rows_in_window(window_seconds)
        if not rows:
            return []
        buckets: dict[tuple[float, str], list[dict[str, Any]]] = {}
        for r in rows:
            b = math.floor(r["ts"] / bucket_seconds) * bucket_seconds
            buckets.setdefault((b, r["subtask"]), []).append(r)
        out: list[dict[str, Any]] = []
        for (bucket, subtask), grp in sorted(buckets.items()):
            lat = [g["latency_ms"] for g in grp]
            errs = sum(1 for g in grp if g["status"] != "ok")
            out.append({
                "bucket_ts": bucket,
                "subtask": subtask,
                "requests": len(grp),
                "errors": errs,
                "error_rate": round(errs / len(grp), 4),
                "latency_p50_ms": round(percentile(lat, 50), 2),
                "latency_p95_ms": round(percentile(lat, 95), 2),
                "cost_usd": round(sum(g["cost_usd"] for g in grp), 6),
                "cache_hits": sum(g["cache_hit"] for g in grp),
            })
        return out

    def signal_summary(
        self, window_seconds: float | None = None
    ) -> list[dict[str, Any]]:
        """Aggregate the model-quality signals stored in `extra`.

        These are the metrics that say whether the *output* was any
        good, as opposed to how fast it arrived: retrieval confidence,
        classifier confidence, ASR real-time factor, TTS degradation.
        Returns one row per signal that actually has data, so a signal
        nobody reported never shows up as a misleading zero.
        """
        rows = self._rows_in_window(window_seconds)
        out: list[dict[str, Any]] = []
        for key, how in self.SIGNAL_FIELDS:
            vals = [
                r["extra_parsed"][key]
                for r in rows
                if isinstance(r["extra_parsed"], dict)
                and r["extra_parsed"].get(key) is not None
            ]
            if not vals:
                continue
            subtasks = sorted({
                r["subtask"] for r in rows
                if isinstance(r["extra_parsed"], dict)
                and r["extra_parsed"].get(key) is not None
            })
            if how == "rate":
                value = sum(1 for v in vals if v) / len(vals)
            else:
                nums = [float(v) for v in vals if isinstance(v, (int, float))]
                value = sum(nums) / len(nums) if nums else 0.0
            out.append({
                "signal": key,
                "aggregation": how,
                "value": round(value, 4),
                "n": len(vals),
                "subtasks": ", ".join(subtasks),
            })
        return out

    def error_budget(
        self, slo: float = 0.99, window_seconds: float | None = None
    ) -> dict[str, Any]:
        """SLO attainment and error-budget burn for the gauge.

        With a 99% success objective, 1% of requests may fail before the
        budget is exhausted. `burn_pct` is the fraction of that
        allowance already consumed; above 100 the SLO is breached.
        """
        rows = self._rows_in_window(window_seconds)
        total = len(rows)
        if total == 0:
            return {
                "requests": 0, "failures": 0, "success_rate": 1.0,
                "slo": slo, "budget_pct": 0.0, "burn_pct": 0.0,
                "breached": False,
            }
        failures = sum(1 for r in rows if r["status"] != "ok")
        success_rate = (total - failures) / total
        allowed = 1.0 - slo
        burn = (failures / total) / allowed if allowed > 0 else 0.0
        return {
            "requests": total,
            "failures": failures,
            "success_rate": round(success_rate, 4),
            "slo": slo,
            "budget_pct": round(allowed * 100, 2),
            "burn_pct": round(burn * 100, 2),
            "breached": success_rate < slo,
        }
