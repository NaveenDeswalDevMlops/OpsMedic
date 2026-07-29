# tests/test_insights_offline.py
"""Offline tests for llmops/insights.py (dashboard data helpers).

Run with pytest:   pytest tests/test_insights_offline.py -v
Or stdlib runner:  python tests/test_insights_offline.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmops.insights import (
    feedback_summary,
    headline_delta,
    per_class_table,
    training_curves,
)


def test_feedback_summary_counts_and_ignores_malformed():
    rows = [
        {"subtask": "feedback", "extra": '{"value": 1}'},
        {"subtask": "feedback", "extra": '{"value": 1}'},
        {"subtask": "feedback", "extra": '{"value": -1}'},
        {"subtask": "feedback", "extra": "not-json"},   # ignored
        {"subtask": "feedback", "extra": None},          # ignored
        {"subtask": "resolve", "extra": '{"value": 1}'},  # wrong subtask
    ]
    assert feedback_summary(rows) == {"up": 2, "down": 1, "total": 3}
    assert feedback_summary([]) == {"up": 0, "down": 0, "total": 0}


def test_training_curves_split_train_and_eval():
    log = [
        {"loss": 2.2, "epoch": 0.1},
        {"loss": 1.8, "epoch": 0.5},
        {"eval_loss": 1.7, "eval_accuracy": 0.34, "eval_f1_macro": 0.25,
         "epoch": 1.0},
        {"loss": 1.6, "epoch": 1.5},
        {"eval_loss": 1.57, "eval_accuracy": 0.39, "eval_f1_macro": 0.35,
         "epoch": 2.0},
        {"train_runtime": 999.0},  # terminal entry: neither series
    ]
    curves = training_curves(log)
    assert curves["train"] == [(0.1, 2.2), (0.5, 1.8), (1.5, 1.6)]
    assert curves["eval"] == [(1.0, 0.34, 0.25, 1.7), (2.0, 0.39, 0.35, 1.57)]
    assert training_curves([]) == {"train": [], "eval": []}


_COMPARE = {
    "base": {
        "accuracy": 0.098, "f1_macro": 0.0222,
        "per_class": {"A": {"precision": 0.1, "recall": 0.1, "f1": 0.1,
                            "support": 50}},
    },
    "finetuned": {
        "accuracy": 0.392, "f1_macro": 0.3516, "eval_rows": 1000,
        "per_class": {
            "A": {"precision": 0.5, "recall": 0.4, "f1": 0.44, "support": 50},
            "B": {"precision": 0.8, "recall": 0.7, "f1": 0.74, "support": 60},
        },
    },
}


def test_headline_delta_matches_users_real_numbers():
    rows = headline_delta(_COMPARE)
    acc = next(r for r in rows if r["metric"] == "accuracy")
    assert acc["base"] == 0.098 and acc["finetuned"] == 0.392
    assert acc["delta"] == 0.294
    f1 = next(r for r in rows if r["metric"] == "f1_macro")
    assert f1["delta"] == 0.3294


def test_per_class_table_sorted_by_f1_desc():
    rows = per_class_table(_COMPARE, "finetuned")
    assert [r["class"] for r in rows] == ["B", "A"]
    assert rows[0]["f1"] == 0.74
    assert per_class_table({}, "finetuned") == []


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
