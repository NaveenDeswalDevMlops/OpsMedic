# finetune/data.py
"""Data utilities for fine-tuning the ticket-queue classifier.

Pure pandas/stdlib logic (no torch/transformers imports) so it is fully
unit-testable offline. Used by train.py / evaluate.py / compare.py.

Dataset: the stratified splits written by scripts/prepare_dataset.py
(data/finetune_train.csv, data/finetune_test.csv) from
Tobi-Bueck/customer-support-tickets (DOI: 10.57967/hf/6184).
"""
from __future__ import annotations

import os
import random
import sys
from typing import Any

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config  # noqa: E402

TEXT_MAX_CHARS = 2000


def join_text(title: Any, description: Any) -> str:
    """'title. description' with NaN/empty safety, length-capped.

    Mirrors what the app feeds the classifier at inference time (an
    incident narrative that opens with a subject-like line).
    """
    t = "" if title is None or (isinstance(title, float)) else str(title).strip()
    d = (
        ""
        if description is None or (isinstance(description, float))
        else str(description).strip()
    )
    joined = f"{t}. {d}" if t and d else (t or d)
    return joined[:TEXT_MAX_CHARS]


def build_label_maps(labels: list[str]) -> tuple[dict[int, str], dict[str, int]]:
    """Stable id<->label mapping over a SORTED label list."""
    ordered = sorted(labels)
    id2label = {i: lab for i, lab in enumerate(ordered)}
    label2id = {lab: i for i, lab in enumerate(ordered)}
    return id2label, label2id


def stratified_cap(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    """Deterministically cap rows, preserving per-category proportions.

    max_rows <= 0 means 'keep all'. Every category keeps at least 1 row.
    """
    if max_rows <= 0 or max_rows >= len(df):
        return df.reset_index(drop=True)
    rng = random.Random(seed)
    frac = max_rows / len(df)
    keep_idx: list[int] = []
    for _, grp in df.groupby("category"):
        idx = list(grp.index)
        rng.shuffle(idx)
        keep_idx.extend(idx[: max(1, round(len(idx) * frac))])
    return df.loc[sorted(keep_idx)].reset_index(drop=True)


def balance_to_median(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Downsample each category to the median class size.

    Counters majority-class collapse (e.g. everything predicted as the
    biggest queue) without discarding as much data as min-class
    balancing. Deterministic under `seed`.
    """
    counts = df["category"].value_counts()
    cap = int(counts.median())
    rng = random.Random(seed)
    keep_idx: list[int] = []
    for _, grp in df.groupby("category"):
        idx = list(grp.index)
        rng.shuffle(idx)
        keep_idx.extend(idx[:cap])
    return df.loc[sorted(keep_idx)].reset_index(drop=True)


def load_split(
    csv_path: str,
    labels: list[str] | None = None,
    max_rows: int = 0,
    seed: int = 42,
    balanced: bool = False,
) -> tuple[list[str], list[int], list[str]]:
    """Load a split CSV -> (texts, label_ids, labels).

    If `labels` is None it is derived (sorted) from this split; pass the
    TRAIN-derived labels when loading the test split so ids line up.
    Rows whose category is not in `labels` are dropped (with a warning).
    `balanced=True` median-downsamples classes BEFORE the max_rows cap
    (use for training only; keep the eval split at true distribution).
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. Run: python scripts/prepare_dataset.py"
        )
    df = pd.read_csv(csv_path)
    for col in ("title", "description", "category"):
        if col not in df.columns:
            raise ValueError(f"{csv_path} missing required column '{col}'")

    if labels is None:
        labels = sorted(set(df["category"].dropna().astype(str)))
    else:
        labels = sorted(labels)  # ids ALWAYS align with sorted order
    _, label2id = build_label_maps(labels)

    known = df["category"].astype(str).isin(label2id)
    dropped = int((~known).sum())
    if dropped:
        print(f"WARNING: dropping {dropped} rows with labels outside the label set")
    df = df[known]

    if balanced:
        before_bal = len(df)
        df = balance_to_median(df, seed)
        print(f"Balanced classes to median size: {before_bal} -> {len(df)} rows")

    df = stratified_cap(df, max_rows, seed)
    texts = [
        join_text(t, d) for t, d in zip(df["title"], df["description"])
    ]
    label_ids = [label2id[str(c)] for c in df["category"]]
    return texts, label_ids, labels


def default_paths() -> dict[str, str]:
    """Central place for the artifact paths used across scripts."""
    art = config.CLASSIFIER_FINETUNED_DIR
    return {
        "train_csv": config.FINETUNE_TRAIN_CSV,
        "test_csv": config.FINETUNE_TEST_CSV,
        "artifact_dir": art,
        "labels_json": os.path.join(art, "labels.json"),
        "training_log": os.path.join(art, "training_log.json"),
        "compare_json": os.path.join(art, "compare.json"),
    }
