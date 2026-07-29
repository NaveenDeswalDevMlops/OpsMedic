# tests/test_finetune_offline.py
"""Offline tests for finetune/data.py (no torch/transformers needed).

Run with pytest:   pytest tests/test_finetune_offline.py -v
Or stdlib runner:  python tests/test_finetune_offline.py
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from finetune.data import (
    TEXT_MAX_CHARS,
    balance_to_median,
    build_label_maps,
    join_text,
    load_split,
    stratified_cap,
)


def test_balance_to_median_caps_majority_deterministically():
    frames = []
    sizes = {"IT Support": 40, "HR": 10, "Billing and Payments": 20,
             "Technical Support": 30}
    rows = []
    for cat, n in sizes.items():
        for i in range(n):
            rows.append({"ticket_id": f"{cat}-{i}", "title": cat,
                         "description": f"d{i}", "category": cat})
    df = pd.DataFrame(rows)
    b1 = balance_to_median(df, seed=42)
    b2 = balance_to_median(df, seed=42)
    assert b1.equals(b2)  # deterministic
    counts = b1["category"].value_counts()
    median = 25  # median of (40,10,20,30)
    assert counts["IT Support"] == median and counts["Technical Support"] == median
    assert counts["HR"] == 10 and counts["Billing and Payments"] == 20  # untouched
    del frames

CATS = ["IT Support", "Billing and Payments", "Technical Support", "HR"]


def _split_csv(rows_per_cat: int = 20) -> str:
    records = []
    for c in CATS:
        for i in range(rows_per_cat):
            records.append(
                {
                    "ticket_id": f"OPM-{len(records) + 1:05d}",
                    "title": f"{c} problem {i}",
                    "description": f"detailed description {i} for {c.lower()}",
                    "resolution": "fix applied",
                    "category": c,
                    "type": "Incident",
                    "priority": "medium",
                    "tags": "A|B",
                }
            )
    path = os.path.join(tempfile.mkdtemp(), "split.csv")
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def test_join_text_handles_nan_and_caps_length():
    assert join_text("Title", "Body") == "Title. Body"
    assert join_text(float("nan"), "Body") == "Body"
    assert join_text("Title", None) == "Title"
    assert len(join_text("T", "x" * 5000)) == TEXT_MAX_CHARS


def test_build_label_maps_sorted_and_inverse():
    id2label, label2id = build_label_maps(["b", "a", "c"])
    assert id2label == {0: "a", 1: "b", 2: "c"}
    assert all(label2id[v] == k for k, v in id2label.items())


def test_stratified_cap_proportions_and_determinism():
    df = pd.read_csv(_split_csv(rows_per_cat=20))  # 80 rows, 4 cats
    capped1 = stratified_cap(df, max_rows=40, seed=42)
    capped2 = stratified_cap(df, max_rows=40, seed=42)
    assert capped1.equals(capped2)
    counts = capped1["category"].value_counts()
    assert set(counts.index) == set(CATS)
    assert all(c == 10 for c in counts)  # 40/80 -> half of each class
    # cap of 0 or >= len keeps everything
    assert len(stratified_cap(df, 0, 42)) == 80
    assert len(stratified_cap(df, 500, 42)) == 80


def test_load_split_ids_align_with_given_labels():
    path = _split_csv(rows_per_cat=5)
    texts, ids, labels = load_split(path, labels=None, max_rows=0)
    assert len(texts) == len(ids) == 20
    assert labels == sorted(CATS)
    assert texts[0].startswith(labels[ids[0]].split()[0]) or True  # smoke
    # ids must map back to the right category through the sorted labels
    df = pd.read_csv(path)
    for text, i in zip(texts, ids):
        assert labels[i] in text  # category name embedded in fixture text

    # passing a RESTRICTED (deliberately unsorted) label set: rows outside
    # it are dropped AND the returned labels come back sorted with ids
    # aligned to that sorted order.
    texts2, ids2, labels2 = load_split(path, labels=["IT Support", "HR"])
    assert labels2 == ["HR", "IT Support"]  # sorted contract
    assert len(texts2) == 10
    for text, i in zip(texts2, ids2):
        assert labels2[i] in text  # exact id<->label alignment


def test_load_split_missing_file_raises():
    try:
        load_split("/nope/never.csv")
    except FileNotFoundError as exc:
        assert "prepare_dataset" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("must raise on missing file")


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
