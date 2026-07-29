# tests/test_llmops.py
"""Tests for the LLMOps core: metrics logger, cache, and BaseSubTask.

Run with pytest on your machine:      pytest tests/test_llmops.py -v
Or with the stdlib runner (no deps):  python tests/test_llmops.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

# Make repo root importable when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmops.cache import ResponseCache
from llmops.metrics import MetricsLogger, approx_tokens, percentile
from models.base import BaseSubTask


# ---------------------------------------------------------------- helpers
def _tmpdb(name: str) -> str:
    return os.path.join(tempfile.mkdtemp(), name)


class EchoTask(BaseSubTask):
    """Minimal sub-task: uppercases text. Used to test the base wrapper."""

    name = "echo"
    category = "NLP"
    model_name = "unit-test-echo"

    def _run(self, payload):
        return str(payload).upper()


class BoomTask(BaseSubTask):
    """Always fails — tests the error path."""

    name = "boom"
    category = "NLP"
    model_name = "unit-test-boom"

    def _run(self, payload):
        raise RuntimeError("intentional failure")


class UsageTask(BaseSubTask):
    """Reports exact usage — tests report_usage()."""

    name = "usage"
    category = "NLP"
    model_name = "unit-test-usage"

    def _run(self, payload):
        self.report_usage(tokens_in=100, tokens_out=25)
        return "ok"


# ---------------------------------------------------------------- metrics
def test_percentile_math():
    assert percentile([], 50) == 0.0
    assert percentile([10.0], 95) == 10.0
    vals = [float(v) for v in range(1, 101)]  # 1..100
    assert percentile(vals, 50) == 50.0
    assert percentile(vals, 95) == 95.0
    assert percentile(vals, 100) == 100.0


def test_approx_tokens():
    assert approx_tokens("") == 0
    assert approx_tokens("abcd") == 1
    assert approx_tokens("a" * 400) == 100


def test_metrics_logger_writes_rows():
    logger = MetricsLogger(_tmpdb("m.db"))
    with logger.track("retrieve", "faiss+minilm") as rec:
        rec.set_tokens(tokens_in=10, tokens_out=0)
        time.sleep(0.01)
    rows = logger.recent()
    assert len(rows) == 1
    row = rows[0]
    assert row["subtask"] == "retrieve"
    assert row["status"] == "ok"
    assert row["tokens_in"] == 10
    assert row["latency_ms"] >= 10.0  # slept 10ms


def test_metrics_logger_records_errors_and_reraises():
    logger = MetricsLogger(_tmpdb("m.db"))
    try:
        with logger.track("resolve", "llama-3.1-8b-instant"):
            raise ValueError("kaboom")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("exception should have propagated")
    row = logger.recent()[0]
    assert row["status"] == "error"
    assert "kaboom" in row["error"]


def test_metrics_cost_estimation_and_summary():
    logger = MetricsLogger(_tmpdb("m.db"))
    with logger.track("resolve", "llama-3.1-8b-instant") as rec:
        rec.set_tokens(tokens_in=1_000_000, tokens_out=1_000_000)
    s = logger.summary()
    assert s["requests"] == 1
    assert abs(s["cost_usd"] - (0.05 + 0.08)) < 1e-9
    assert s["error_rate"] == 0.0
    per = logger.summary_by_subtask()
    assert per[0]["subtask"] == "resolve"
    assert per[0]["requests"] == 1


# ------------------------------------------------------------------ cache
def test_cache_set_get_and_hit_count():
    cache = ResponseCache(_tmpdb("c.db"), ttl_seconds=3600)
    payload = {"q": "vpn down"}
    assert cache.get("resolve", "m", payload) is None
    cache.set("resolve", "m", payload, {"answer": 42})
    assert cache.get("resolve", "m", payload) == {"answer": 42}
    assert cache.get("resolve", "m", payload) == {"answer": 42}
    st = cache.stats()
    assert st["entries"] == 1
    assert st["total_hits"] == 2


def test_cache_ttl_expiry():
    cache = ResponseCache(_tmpdb("c.db"), ttl_seconds=0)  # expire instantly
    cache.set("s", "m", "x", "y")
    time.sleep(0.01)
    assert cache.get("s", "m", "x") is None


# ------------------------------------------------------------- base class
def test_subtask_schema_and_metrics_row():
    db = _tmpdb("m.db")
    task = EchoTask(metrics=MetricsLogger(db))
    result = task.run("hello")
    assert set(result.keys()) == {"output", "metrics"}
    assert result["output"] == "HELLO"
    assert result["metrics"]["status"] == "ok"
    assert task.metrics.recent()[0]["subtask"] == "echo"


def test_subtask_error_does_not_raise_but_is_logged():
    task = BoomTask(metrics=MetricsLogger(_tmpdb("m.db")))
    result = task.run("x")
    assert result["output"] is None
    assert result["metrics"]["status"] == "error"
    assert "intentional failure" in result["metrics"]["error"]
    assert task.metrics.recent()[0]["status"] == "error"


def test_subtask_exact_usage_reported():
    task = UsageTask(metrics=MetricsLogger(_tmpdb("m.db")))
    result = task.run("x")
    assert result["metrics"]["tokens_in"] == 100
    assert result["metrics"]["tokens_out"] == 25
    assert task.metrics.recent()[0]["tokens_in"] == 100


def test_subtask_cache_hit_flow():
    mdb, cdb = _tmpdb("m.db"), _tmpdb("c.db")
    task = EchoTask(metrics=MetricsLogger(mdb), cache=ResponseCache(cdb))
    r1 = task.run("hi")
    r2 = task.run("hi")
    assert r1["metrics"]["cache_hit"] is False
    assert r2["metrics"]["cache_hit"] is True
    assert r2["output"] == "HI"
    rows = task.metrics.recent()
    assert rows[0]["cache_hit"] == 1  # newest row = the hit
    assert rows[1]["cache_hit"] == 0


# ------------------------------------------------------- stdlib test runner
def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
