# tests/test_streaming_offline.py
"""Offline test for ResolutionTask.stream() without hitting Groq.

Monkeypatches requests.post with a fake SSE stream and a temp
metrics+cache, then asserts: chunks are yielded in order, exactly one
metrics row is written with exact token usage, and the full text is
cached. Also covers the no-key fallback path.

Run with pytest:   pytest tests/test_streaming_offline.py -v
Or stdlib runner:  python tests/test_streaming_offline.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmops.cache import ResponseCache
from llmops.metrics import MetricsLogger
from models import resolution as resolution_mod
from models.resolution import ResolutionTask
from src import config


def _tmp(name: str) -> str:
    return os.path.join(tempfile.mkdtemp(), name)


class _FakeSSEResponse:
    """Context-manager mimicking requests' streaming Response."""

    status_code = 200

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_lines(self):
        yield from self._lines


def _fake_post(*args, **kwargs):
    # Two content chunks + a final usage-only chunk + [DONE]
    return _FakeSSEResponse([
        b'data: {"choices":[{"delta":{"content":"Diagnosis: login outage. "}}]}',
        b'data: {"choices":[{"delta":{"content":"Steps: 1) clear SSO cache."}}]}',
        b'data: {"choices":[],"usage":{"prompt_tokens":812,"completion_tokens":40}}',
        b"data: [DONE]",
    ])


def test_stream_yields_chunks_logs_one_row_and_caches(monkeypatch=None):
    # allow both pytest (monkeypatch fixture) and stdlib invocation
    real_key = config.GROQ_API_KEY
    real_post = getattr(resolution_mod, "requests", None)
    config.GROQ_API_KEY = "test-key"  # force the streaming path

    import requests
    orig_post = requests.post
    requests.post = _fake_post
    try:
        metrics = MetricsLogger(_tmp("m.db"))
        cache = ResponseCache(_tmp("c.db"))
        task = ResolutionTask(metrics=metrics, cache=cache)
        payload = {
            "incident": "portal login down",
            "similar": [{"ticket_id": "OPM-00042", "title": "t",
                         "description": "d", "resolution": "r", "score": 0.9}],
            "category": "IT Support",
        }
        chunks = list(task.stream(payload))
        full = "".join(chunks)
        assert "Diagnosis: login outage." in full
        assert "clear SSO cache" in full

        # exactly one metrics row, exact usage recorded
        recent = metrics.recent()
        assert len(recent) == 1
        assert recent[0]["subtask"] == "resolve"
        assert recent[0]["tokens_in"] == 812
        assert recent[0]["tokens_out"] == 40
        assert recent[0]["status"] == "ok"

        # full text cached -> second stream is a cache hit (one more row)
        chunks2 = list(task.stream(payload))
        assert "".join(chunks2) == full
        assert len(metrics.recent()) == 2
        assert metrics.recent()[0]["cache_hit"] == 1
    finally:
        requests.post = orig_post
        config.GROQ_API_KEY = real_key


def test_stream_fallback_without_key():
    real_key = config.GROQ_API_KEY
    config.GROQ_API_KEY = None
    try:
        metrics = MetricsLogger(_tmp("m.db"))
        task = ResolutionTask(metrics=metrics, cache=None)
        payload = {
            "incident": "x",
            "similar": [{"ticket_id": "OPM-1", "title": "t", "description": "d",
                         "resolution": "RESTART THE SERVICE", "score": 0.8}],
            "category": None,
        }
        out = "".join(task.stream(payload))
        assert "RESTART THE SERVICE" in out
        assert len(metrics.recent()) == 1  # fallback still logs a row
    finally:
        config.GROQ_API_KEY = real_key


class _SlowSSEResponse(_FakeSSEResponse):
    """Stream that takes measurable time, to test latency instrumentation."""

    def __init__(self, delay_s: float = 0.12) -> None:
        self.delay_s = delay_s

    def iter_lines(self):
        time.sleep(self.delay_s)            # queue + prompt processing
        yield b'data: {"choices":[{"delta":{"content":"Diagnosis: outage. "}}]}'
        time.sleep(self.delay_s)            # decode time
        yield b'data: {"choices":[{"delta":{"content":"Fix: clear SSO cache."}}]}'
        yield b'data: {"choices":[],"usage":{"prompt_tokens":800,"completion_tokens":40}}'
        yield b"data: [DONE]"


def test_streamed_latency_is_the_real_generation_time():
    """Regression: track() used to open in the finally block, AFTER the
    stream was exhausted, so it timed only the token bookkeeping and wrote
    ~0 ms for a multi-second generation. Every resolve latency on the
    dashboard was wrong."""
    real_key = config.GROQ_API_KEY
    config.GROQ_API_KEY = "test-key"
    import requests
    orig_post = requests.post
    delay = 0.12
    requests.post = lambda *a, **k: _SlowSSEResponse(delay)
    try:
        metrics = MetricsLogger(_tmp("m.db"))
        task = ResolutionTask(metrics=metrics, cache=None)
        wall = time.perf_counter()
        list(task.stream({"incident": "portal down", "similar": [],
                          "category": None}))
        wall_ms = (time.perf_counter() - wall) * 1000.0

        logged = metrics.recent(1)[0]["latency_ms"]
        # must be the real elapsed time, not the bookkeeping block
        assert logged >= 2 * delay * 1000 * 0.8, (
            f"latency {logged:.1f} ms under-reports a "
            f"{2 * delay * 1000:.0f} ms stream")
        assert logged <= wall_ms * 1.05, "latency exceeds wall clock"
    finally:
        requests.post = orig_post
        config.GROQ_API_KEY = real_key


def test_streamed_row_records_ttft_and_throughput():
    real_key = config.GROQ_API_KEY
    config.GROQ_API_KEY = "test-key"
    import requests
    orig_post = requests.post
    delay = 0.12
    requests.post = lambda *a, **k: _SlowSSEResponse(delay)
    try:
        metrics = MetricsLogger(_tmp("m.db"))
        task = ResolutionTask(metrics=metrics, cache=None)
        list(task.stream({"incident": "portal down", "similar": [],
                          "category": None}))
        row = metrics.recent(1)[0]
        extra = json.loads(row["extra"])

        assert "ttft_ms" in extra, "time-to-first-token not recorded"
        # first token arrives after the first delay, before the stream ends
        assert extra["ttft_ms"] >= delay * 1000 * 0.8
        assert extra["ttft_ms"] < row["latency_ms"]
        assert extra["tokens_per_sec"] > 0
        assert extra["prompt_version"] == "resolve-v1-stepwise"
        # exact usage from the stream, so cost is not an estimate
        assert row["tokens_in"] == 800 and row["tokens_out"] == 40
        assert "tokens_estimated" not in extra
        assert row["cost_usd"] > 0
    finally:
        requests.post = orig_post
        config.GROQ_API_KEY = real_key


def test_track_latency_override_is_used_verbatim():
    """The metrics-layer half of the same fix."""
    metrics = MetricsLogger(_tmp("m.db"))
    with metrics.track("resolve", "llama-3.1-8b-instant", latency_ms=4321.5):
        pass                                  # instant block
    assert metrics.recent(1)[0]["latency_ms"] == 4321.5

    with metrics.track("classify", "distilbert"):
        time.sleep(0.05)                      # no override -> self-timed
    measured = metrics.recent(1)[0]["latency_ms"]
    assert 40 <= measured < 500, measured


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
