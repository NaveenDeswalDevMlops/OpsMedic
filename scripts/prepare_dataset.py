# scripts/prepare_dataset.py
"""Prepare the Customer IT Support ticket dataset for OpsMedic.

Dataset (cited in the report):
    T. Bueck, "Customer Support Tickets", Hugging Face.
    https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets
    DOI: 10.57967/hf/6184 - License: CC-BY-NC-4.0 (academic use OK).
    61.8k email tickets WITH the agent's resolution reply ("answer"),
    plus type, queue, priority, language, and subcategory tags.

What this script does:
  1. Acquire the raw data:
       a) automatically via `datasets.load_dataset(...)` (Hugging Face), or
       b) manually: download the CSV (HF "Files" tab or the Kaggle mirror
          tobiasbueck/multilingual-customer-support-tickets) into data/raw/
          - the script auto-discovers it.
  2. Filter to English rows with a non-empty body AND answer; keep the
     top-N queues (long tail of rare queues is dropped for stable
     classification); normalize to the OpsMedic KB schema
     (gensense-style):  data/tickets.csv
       ticket_id, title, description, resolution, category(=queue),
       type, priority, tags
  3. Create a stratified, seeded fine-tuning split over `category`:
       data/finetune_train.csv / data/finetune_test.csv (85/15)

Usage:
    python scripts/prepare_dataset.py                 # auto-download (HF)
    python scripts/prepare_dataset.py --raw-csv path/to/file.csv
    python scripts/prepare_dataset.py --kb-rows 5000  # smaller KB index
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config  # noqa: E402

REQUIRED_COLS = {"subject", "body", "answer", "type", "queue", "priority", "language"}
TAG_COLS = [f"tag_{i}" for i in range(1, 9)]
TEST_FRACTION = 0.15


def queue_slug(queue: str) -> str:
    """Map a queue name to its SOP filename stem.

    'Billing and Payments' -> 'billing_and_payments' (data/sops/<slug>.md).
    Used by the resolution sub-task to attach the linked SOP.
    """
    return queue.strip().lower().replace(" ", "_")


def _download_via_hf() -> pd.DataFrame | None:
    """Try the Hugging Face datasets library; return a DataFrame or None."""
    try:
        from datasets import load_dataset  # lazy: heavy import
    except ImportError:
        print("`datasets` not installed - falling back to manual CSV mode.")
        return None
    try:
        ds = load_dataset(config.HF_DATASET, split="train")
        print(f"Downloaded {len(ds)} rows from HF: {config.HF_DATASET}")
        return ds.to_pandas()
    except Exception as exc:  # noqa: BLE001 - network/auth issues
        print(f"HF download failed ({type(exc).__name__}: {exc}).")
        print("Falling back to manual CSV mode.")
        return None


def _read_raw_dir(raw_dir: str) -> pd.DataFrame | None:
    """Read and concatenate ALL CSVs found under raw_dir (or None).

    The upstream dataset ships as multiple CSV versions; concatenating
    them reproduces the full corpus. Differing extra columns across
    versions are tolerated; duplicates are removed later by `body`.
    """
    paths = sorted(
        glob.glob(os.path.join(raw_dir, "**", "*.csv"), recursive=True)
    )
    if not paths:
        return None
    frames = []
    for p in paths:
        frames.append(pd.read_csv(p))
        print(f"Read {len(frames[-1])} rows from {p}")
    return pd.concat(frames, ignore_index=True, sort=False)


def _load_raw(explicit_csv: str | None) -> pd.DataFrame:
    """Load raw data (explicit path > data/raw/*.csv > HF download).

    Local files are checked BEFORE any network call, so org networks
    that block Hugging Face never cause a hang - just drop the CSVs
    into data/raw/ and run.
    """
    if explicit_csv:
        if not os.path.isfile(explicit_csv):
            sys.exit(f"ERROR: --raw-csv path not found: {explicit_csv}")
        print(f"Using raw CSV: {explicit_csv}")
        return pd.read_csv(explicit_csv)

    local = _read_raw_dir(config.RAW_DATA_DIR)
    if local is not None:
        return local

    df = _download_via_hf()
    if df is not None:
        return df

    sys.exit(
        "ERROR: no raw data found.\n"
        f"Download the CSVs from https://huggingface.co/datasets/"
        f"{config.HF_DATASET}/tree/main (or the Kaggle mirror) and place "
        f"them in {config.RAW_DATA_DIR}/, then re-run."
    )


def load_and_normalize(
    df: pd.DataFrame, language: str, top_queues: int
) -> pd.DataFrame:
    """Filter + normalize raw rows to the OpsMedic KB schema."""
    cols_lower = {c.lower().strip(): c for c in df.columns}
    missing = REQUIRED_COLS - set(cols_lower)
    if missing:
        sys.exit(
            f"ERROR: raw data missing expected columns {sorted(missing)}; "
            f"found: {list(df.columns)}"
        )
    d = df.rename(columns={v: k for k, v in cols_lower.items()}).copy()

    before = len(d)
    d = d[d["language"].astype(str).str.lower() == language.lower()]
    for col in ("subject", "body", "answer", "queue"):
        d[col] = d[col].fillna("").astype(str).str.strip()
        d = d[(d[col].str.len() > 0) & (d[col].str.lower() != "nan")]
    d = d.drop_duplicates(subset=["body"]).reset_index(drop=True)
    print(f"Filtered {before} -> {len(d)} rows (lang={language}, non-empty, dedup).")

    keep = d["queue"].value_counts().head(top_queues).index.tolist()
    dropped = len(d) - int(d["queue"].isin(keep).sum())
    d = d[d["queue"].isin(keep)].reset_index(drop=True)
    print(f"Kept top {len(keep)} queues; dropped {dropped} long-tail rows.")

    tags_present = [c for c in TAG_COLS if c in d.columns]
    tags = (
        d[tags_present]
        .fillna("")
        .astype(str)
        .apply(
            lambda row: "|".join(
                t
                for t in (s.strip() for s in row)
                if t and t.lower() not in ("nan", "none", "null")
            ),
            axis=1,
        )
        if tags_present
        else ""
    )

    out = pd.DataFrame(
        {
            "title": d["subject"],
            "description": d["body"],
            "resolution": d["answer"],
            "category": d["queue"],
            "type": d["type"].astype(str).str.strip(),
            "priority": d["priority"].astype(str).str.strip(),
            "tags": tags,
        }
    )
    out.insert(0, "ticket_id", [f"OPM-{i:05d}" for i in range(1, len(out) + 1)])
    return out


def stratified_split(
    df: pd.DataFrame, test_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic stratified split by `category` (stdlib random)."""
    rng = random.Random(seed)
    test_idx: list[int] = []
    for _, grp in df.groupby("category"):
        idx = list(grp.index)
        rng.shuffle(idx)
        n_test = max(1, round(len(idx) * test_fraction))
        test_idx.extend(idx[:n_test])
    mask = df.index.isin(test_idx)
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv", default=None, help="path to a raw dataset CSV")
    parser.add_argument("--language", default=config.DATASET_LANGUAGE)
    parser.add_argument("--top-queues", type=int, default=config.TOP_QUEUES)
    parser.add_argument("--kb-rows", type=int, default=config.KB_ROWS)
    parser.add_argument("--seed", type=int, default=config.FINETUNE_SEED)
    parser.add_argument("--out-tickets", default=config.TICKETS_CSV)
    parser.add_argument("--out-train", default=config.FINETUNE_TRAIN_CSV)
    parser.add_argument("--out-test", default=config.FINETUNE_TEST_CSV)
    args = parser.parse_args()

    raw = _load_raw(args.raw_csv)
    df = load_and_normalize(raw, args.language, args.top_queues)

    print(f"\nUsable tickets (with resolutions): {len(df)}")
    print("Queue distribution:")
    for cat, count in df["category"].value_counts().items():
        print(f"  {cat:<34} {count}")

    # 1) Retrieval knowledge base (optionally capped, stratified sample)
    kb = df
    if args.kb_rows and args.kb_rows < len(df):
        frac = args.kb_rows / len(df)
        kb = (
            df.groupby("category", group_keys=False)
            .sample(frac=frac, random_state=args.seed)
            .sort_index()
            .reset_index(drop=True)
        )
        print(f"KB capped to {len(kb)} rows (stratified sample).")
    kb.to_csv(args.out_tickets, index=False)
    print(f"Wrote KB -> {args.out_tickets} ({len(kb)} rows)")

    # 2) Fine-tuning split over the FULL filtered dataset
    train, test = stratified_split(df, TEST_FRACTION, args.seed)
    train.to_csv(args.out_train, index=False)
    test.to_csv(args.out_test, index=False)
    print(f"Wrote train -> {args.out_train} ({len(train)} rows)")
    print(f"Wrote test  -> {args.out_test} ({len(test)} rows)")
    print("\nDone. Next: python scripts/build_index.py (Phase 2B)")


if __name__ == "__main__":
    main()
