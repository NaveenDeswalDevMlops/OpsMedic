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

import os
import sys
import tempfile

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
