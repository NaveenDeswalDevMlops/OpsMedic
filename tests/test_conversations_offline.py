# tests/test_conversations_offline.py
"""Offline tests for llmops/conversations.py (SQLite chat history).

Run with pytest:   pytest tests/test_conversations_offline.py -v
Or stdlib runner:  python tests/test_conversations_offline.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmops.conversations import ConversationStore, day_bucket, make_title


def _store() -> ConversationStore:
    return ConversationStore(os.path.join(tempfile.mkdtemp(), "conv.db"))


def test_make_title_truncates_on_word_boundary():
    assert make_title("VPN keeps dropping") == "VPN keeps dropping"
    assert make_title("") == "New incident"
    long = "the account management portal is completely unreachable for everyone today"
    t = make_title(long, max_len=40)
    assert t.endswith("…") and len(t) <= 41 and " " in t


def test_day_bucket_boundaries():
    now = time.time()
    assert day_bucket(now, now) == "Today"
    assert day_bucket(now - 86400, now) == "Yesterday"
    assert day_bucket(now - 4 * 86400, now) == "Previous 7 days"
    assert day_bucket(now - 30 * 86400, now) == "Older"


def test_create_add_and_get_messages_roundtrip():
    s = _store()
    cid = s.create()
    s.add_message(cid, "user", "portal down")
    s.add_message(cid, "assistant", "try clearing SSO cache",
                  meta={"grounded_on": ["OPM-00042"], "sop": "it_support"})
    msgs = s.get_messages(cid)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == "portal down"
    assert msgs[1]["meta"]["grounded_on"] == ["OPM-00042"]
    assert msgs[1]["meta"]["sop"] == "it_support"


def test_rename_if_default_sets_title_once():
    s = _store()
    cid = s.create()  # default title "New incident"
    s.rename_if_default(cid, "printer on floor 3 not working at all")
    title = s.list_conversations()[0]["title"]
    assert title.startswith("printer on floor 3")
    # calling again must NOT overwrite a real title
    s.rename_if_default(cid, "totally different text")
    assert s.list_conversations()[0]["title"] == title


def test_list_ordered_by_recency_and_delete():
    s = _store()
    c1 = s.create("first")
    time.sleep(0.01)
    c2 = s.create("second")
    # touching c1 makes it most-recent
    s.add_message(c1, "user", "ping")
    ids = [c["id"] for c in s.list_conversations()]
    assert ids[0] == c1 and c2 in ids
    s.delete(c1)
    remaining = [c["id"] for c in s.list_conversations()]
    assert c1 not in remaining and c2 in remaining
    assert s.get_messages(c1) == []  # messages gone too


def test_grouped_preserves_bucket_order():
    s = _store()
    cid = s.create("today one")
    s.add_message(cid, "user", "hi")
    grouped = s.grouped()
    assert grouped  # non-empty
    labels = [label for label, _ in grouped]
    # order must follow Today > Yesterday > Previous 7 days > Older
    canonical = ["Today", "Yesterday", "Previous 7 days", "Older"]
    assert labels == [x for x in canonical if x in labels]
    assert grouped[0][0] == "Today"


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
