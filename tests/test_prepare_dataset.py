# tests/test_prepare_dataset.py
"""Tests for scripts/prepare_dataset.py using a fixture that mimics the
Tobi-Bueck/customer-support-tickets schema. Runs fully offline.

Run with pytest:   pytest tests/test_prepare_dataset.py -v
Or stdlib runner:  python tests/test_prepare_dataset.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from scripts.prepare_dataset import (
    _read_raw_dir,
    load_and_normalize,
    queue_slug,
    stratified_split,
)
from src import config


def test_read_raw_dir_concatenates_multiple_csvs():
    import tempfile

    d = tempfile.mkdtemp()
    df = _fixture_df(rows_per_queue=4)
    df.iloc[:20].to_csv(os.path.join(d, "part_a.csv"), index=False)
    df.iloc[20:].to_csv(os.path.join(d, "part_b.csv"), index=False)
    combined = _read_raw_dir(d)
    assert combined is not None and len(combined) == len(df)
    assert _read_raw_dir(tempfile.mkdtemp()) is None  # empty dir -> None

TOP_QUEUES = [
    "Technical Support", "IT Support", "Customer Service", "Product Support",
    "Billing and Payments", "Returns and Exchanges",
    "Service Outages and Maintenance", "Sales and Pre-Sales",
    "Human Resources", "General Inquiry",
]


def _fixture_df(rows_per_queue: int = 12) -> pd.DataFrame:
    """Build a raw-schema DataFrame: EN rows per queue + junk to filter."""
    records = []
    n = 0
    for q in TOP_QUEUES:
        for i in range(rows_per_queue):
            n += 1
            records.append(
                {
                    "subject": f"{q} issue {i}",
                    "body": f"Dear team, problem {n} in {q.lower()} area, "
                    f"please assist with detailed case {i}.",
                    "answer": f"Thanks for reporting case {i}; steps: check, "
                    f"fix, confirm for {q.lower()}.",
                    "type": "Incident",
                    "queue": q,
                    "priority": "medium",
                    "language": "en",
                    "tag_1": "Alpha",
                    "tag_2": "Beta" if i % 2 == 0 else None,
                }
            )
    # Rows the pipeline must drop:
    records.append({**records[0]})                                # duplicate body
    records.append({**records[1], "language": "de"})              # wrong language
    records.append({**records[2], "answer": "  ",
                    "body": "unique body no answer"})             # empty answer
    records.append({**records[3], "queue": "Rare Queue X",
                    "body": "unique body rare queue"})            # long-tail queue
    return pd.DataFrame(records)


def test_queue_slug_matches_sop_filenames():
    for q in TOP_QUEUES:
        path = os.path.join(config.SOPS_DIR, queue_slug(q) + ".md")
        assert os.path.isfile(path), f"missing SOP for queue '{q}': {path}"


def test_normalize_filters_schema_and_ids():
    df = load_and_normalize(_fixture_df(), language="en", top_queues=10)
    assert list(df.columns) == [
        "ticket_id", "title", "description", "resolution",
        "category", "type", "priority", "tags",
    ]
    # 10 queues * 12 rows; dup/de/empty-answer/rare-queue rows all dropped
    assert len(df) == 120
    assert set(df["category"]) == set(TOP_QUEUES)
    assert df["ticket_id"].is_unique and df["ticket_id"].iloc[0] == "OPM-00001"
    assert (df["resolution"].str.len() > 0).all()
    # tags joined with '|', null tags omitted
    assert set(df["tags"].unique()) == {"Alpha", "Alpha|Beta"}


def test_normalize_rejects_wrong_schema():
    bad = pd.DataFrame({"foo": ["x"], "bar": ["y"]})
    try:
        load_and_normalize(bad, language="en", top_queues=10)
    except SystemExit as exc:
        assert "missing expected columns" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("should exit on wrong schema")


def test_stratified_split_deterministic_and_leak_free():
    df = load_and_normalize(_fixture_df(), language="en", top_queues=10)
    tr1, te1 = stratified_split(df, 0.15, seed=42)
    tr2, te2 = stratified_split(df, 0.15, seed=42)
    assert tr1.equals(tr2) and te1.equals(te2)
    assert len(tr1) + len(te1) == len(df)
    assert set(tr1["ticket_id"]).isdisjoint(set(te1["ticket_id"]))
    assert set(te1["category"]) == set(TOP_QUEUES)
    # 12 rows/queue -> round(1.8)=2 test rows/queue -> 20 total
    assert len(te1) == 20


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
