# finetune/evaluate.py
"""Evaluate a classifier (base or fine-tuned) on the held-out test split.

Reports accuracy, macro-F1, and a per-class breakdown. Reused by
compare.py for the base-vs-fine-tuned table.

Usage:
    python finetune/evaluate.py                     # fine-tuned artifact
    python finetune/evaluate.py --variant base      # random-head baseline
    python finetune/evaluate.py --max-rows 500      # quicker pass
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finetune.data import build_label_maps, default_paths, load_split  # noqa: E402
from src import config  # noqa: E402


def predict_batched(
    model_source: str,
    labels: list[str],
    texts: list[str],
    batch_size: int = 32,
) -> list[int]:
    """Batched argmax predictions (lazy torch/transformers imports).

    `model_source` is either a HF model id (base) or a local artifact dir
    (fine-tuned). For the base model a fresh classification head sized to
    `labels` is attached - deliberately untrained, that IS the baseline.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    id2label, label2id = build_label_maps(labels)
    tok_source = (
        model_source
        if os.path.isdir(model_source)
        else config.CLASSIFIER_BASE_MODEL
    )
    tokenizer = AutoTokenizer.from_pretrained(tok_source)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_source,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )
    model.eval()

    preds: list[int] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            enc = tokenizer(
                chunk,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt",
            )
            logits = model(**enc).logits
            preds.extend(logits.argmax(dim=-1).tolist())
    return preds


def evaluate_model(
    model_source: str,
    labels: list[str],
    max_rows: int = 0,          # 0 = full held-out split
    seed: int = 42,
) -> dict[str, Any]:
    """Run held-out evaluation; returns a JSON-serializable report."""
    from sklearn.metrics import (  # lazy
        accuracy_score,
        classification_report,
        f1_score,
    )

    paths = default_paths()
    texts, y_true, _ = load_split(
        paths["test_csv"], labels=labels, max_rows=max_rows, seed=seed
    )
    y_pred = predict_batched(model_source, labels, texts)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "model_source": model_source,
        "eval_rows": len(texts),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(
            f1_score(y_true, y_pred, average="macro", zero_division=0), 4
        ),
        "per_class": {
            lab: {
                "precision": round(report[lab]["precision"], 3),
                "recall": round(report[lab]["recall"], 3),
                "f1": round(report[lab]["f1-score"], 3),
                "support": int(report[lab]["support"]),
            }
            for lab in labels
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["base", "finetuned"],
                        default="finetuned")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=config.FINETUNE_SEED)
    args = parser.parse_args()

    paths = default_paths()
    if args.variant == "finetuned":
        if not os.path.isfile(paths["labels_json"]):
            sys.exit("No fine-tuned artifact. Run: python finetune/train.py")
        with open(paths["labels_json"], "r", encoding="utf-8") as fh:
            labels = sorted(json.load(fh))
        source = paths["artifact_dir"]
    else:
        _, _, labels = load_split(paths["train_csv"], max_rows=1, seed=args.seed)
        source = config.CLASSIFIER_BASE_MODEL

    result = evaluate_model(source, labels, args.max_rows, args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
