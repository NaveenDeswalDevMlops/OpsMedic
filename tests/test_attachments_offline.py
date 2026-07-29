# tests/test_attachments_offline.py
"""Offline tests for llmops/attachments.py.

Run with pytest:   pytest tests/test_attachments_offline.py -v
Or stdlib runner:  python tests/test_attachments_offline.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmops.attachments import MAX_CHARS, extract_text, merge_incident


def test_extract_plain_text_and_log():
    r = extract_text("ticket.txt", b"Portal down since 9am. Error 503.")
    assert r["kind"] == "text"
    assert "Portal down" in r["text"]
    assert r["chars"] == len("Portal down since 9am. Error 503.")

    r2 = extract_text("server.log", b"ERROR db timeout\nWARN retry")
    assert r2["kind"] == "text" and "db timeout" in r2["text"]


def test_extract_caps_length():
    big = b"x" * (MAX_CHARS + 500)
    r = extract_text("big.md", big)
    assert len(r["text"]) == MAX_CHARS


def test_image_is_acknowledged_not_extracted():
    r = extract_text("screenshot.png", b"\x89PNG\r\n\x1a\n fake")
    assert r["kind"] == "image"
    assert r["text"] == ""
    assert "screenshot" in r["note"].lower() or "image" in r["note"].lower()


def test_pdf_without_pypdf_degrades_gracefully():
    # If pypdf is installed this returns kind 'pdf' or 'error' on bad bytes;
    # either way it must not raise and must return the schema.
    r = extract_text("sop.pdf", b"%PDF-1.4 broken")
    assert set(r.keys()) == {"kind", "chars", "text", "note"}
    assert r["kind"] in ("pdf", "error")


def test_merge_incident_appends_context():
    inc = "Users cannot log in."
    att = {"kind": "text", "text": "Auth service returned 503 in logs."}
    merged = merge_incident(inc, att)
    assert inc in merged
    assert "Attached context" in merged
    assert "503" in merged
    # empty attachment leaves incident unchanged
    assert merge_incident(inc, {"kind": "image", "text": ""}) == inc


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
