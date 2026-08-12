# ui/model_card.py
"""Model-card data preparation for the fine-tuned queue classifier.

This module is the *pure* half of pages/2_Model_Card.py: it reads the
artifacts written by finetune/train.py and finetune/compare.py and
reshapes them for display, with no Streamlit or Plotly imports, so the
arithmetic can be unit-tested offline.

The single most important thing it computes is the **majority-class
baseline**. The `base` variant in compare.json is the pre-trained
encoder with a randomly initialised head; that head is re-randomised on
every construction, so its score moves between runs and the delta
against it is not reproducible. Comparing against "always predict the
most common class" gives a stable floor that does not depend on a seed.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Any


def load_json(path: str) -> dict[str, Any] | None:
    """Read a JSON artifact, returning None if it is absent or corrupt."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (ValueError, OSError):
        return None


def class_distribution(csv_path: str, label_col: str = "category") -> dict[str, int]:
    """Count rows per class in a prepared split CSV."""
    if not os.path.exists(csv_path):
        return {}
    counts: dict[str, int] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            label = (row.get(label_col) or "").strip()
            if label:
                counts[label] = counts.get(label, 0) + 1
    return counts


def majority_baseline(supports: dict[str, int]) -> dict[str, Any]:
    """Metrics for a classifier that always predicts the largest class.

    This is the honest floor. For the majority class C with share p:
    accuracy = p; precision = p, recall = 1, so F1(C) = 2p/(p+1); every
    other class scores 0, so macro F1 = F1(C)/num_classes.
    """
    total = sum(supports.values())
    if total == 0 or not supports:
        return {"label": None, "accuracy": 0.0, "f1_macro": 0.0,
                "share": 0.0, "n_classes": 0}
    label = max(supports, key=lambda k: supports[k])
    p = supports[label] / total
    n_classes = len(supports)
    return {
        "label": label,
        "share": round(p, 4),
        "accuracy": round(p, 4),
        "f1_macro": round((2 * p / (p + 1)) / n_classes, 4),
        "n_classes": n_classes,
    }


def supports_from_compare(cmp_json: dict[str, Any]) -> dict[str, int]:
    """Extract per-class support counts from a compare.json blob."""
    per_class = (cmp_json.get("finetuned") or {}).get("per_class") or {}
    return {k: int(v.get("support", 0)) for k, v in per_class.items()}


def baseline_table(cmp_json: dict[str, Any],
                   supports: dict[str, int]) -> list[dict[str, Any]]:
    """Three-row comparator table: random head, majority class, fine-tuned."""
    base = cmp_json.get("base") or {}
    ft = cmp_json.get("finetuned") or {}
    maj = majority_baseline(supports)
    rows = [
        {"comparator": "Random-head base (seed-dependent)",
         "accuracy": base.get("accuracy", 0.0),
         "f1_macro": base.get("f1_macro", 0.0),
         "stable": "no"},
        {"comparator": f"Majority class - always '{maj['label']}'",
         "accuracy": maj["accuracy"],
         "f1_macro": maj["f1_macro"],
         "stable": "yes"},
        {"comparator": "Fine-tuned DeBERTa-v3-base",
         "accuracy": ft.get("accuracy", 0.0),
         "f1_macro": ft.get("f1_macro", 0.0),
         "stable": "yes"},
    ]
    return rows


def collapse_check(cmp_json: dict[str, Any]) -> dict[str, Any]:
    """Detect whether a variant degenerated to one constant prediction.

    A collapsed model has recall 1.0 on exactly one class and 0.0 on all
    others, which makes macro F1 equal F1(majority)/num_classes. This is
    the failure mode that produced a *positive* delta in an earlier run
    purely because the random baseline landed on a rarer class.
    """
    out: dict[str, Any] = {}
    for side in ("base", "finetuned"):
        per_class = (cmp_json.get(side) or {}).get("per_class") or {}
        if not per_class:
            out[side] = {"collapsed": None, "nonzero_f1": 0, "n_classes": 0}
            continue
        nonzero = [k for k, v in per_class.items() if float(v.get("f1", 0)) > 0]
        full_recall = [k for k, v in per_class.items()
                       if float(v.get("recall", 0)) >= 0.999]
        out[side] = {
            "collapsed": len(nonzero) <= 1,
            "nonzero_f1": len(nonzero),
            "n_classes": len(per_class),
            "always_predicts": full_recall[0] if len(full_recall) == 1
            and len(nonzero) <= 1 else None,
        }
    return out


def per_class_rows(cmp_json: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-class precision/recall/F1/support, fine-tuned vs base."""
    ft = (cmp_json.get("finetuned") or {}).get("per_class") or {}
    base = (cmp_json.get("base") or {}).get("per_class") or {}
    rows: list[dict[str, Any]] = []
    for label in sorted(ft, key=lambda k: -int(ft[k].get("support", 0))):
        f = ft[label]
        b = base.get(label, {})
        rows.append({
            "class": label,
            "support": int(f.get("support", 0)),
            "precision": round(float(f.get("precision", 0)), 4),
            "recall": round(float(f.get("recall", 0)), 4),
            "f1": round(float(f.get("f1", 0)), 4),
            "base_f1": round(float(b.get("f1", 0)), 4),
            "delta_f1": round(float(f.get("f1", 0)) - float(b.get("f1", 0)), 4),
        })
    return rows


def confusion_pairs(cmp_json: dict[str, Any]) -> list[dict[str, Any]]:
    """Misclassifications from the sample rows, for error analysis.

    compare.json stores a handful of sampled predictions rather than the
    full prediction vector, so this is illustrative rather than a
    complete confusion matrix - labelled as such in the UI.
    """
    out: list[dict[str, Any]] = []
    for s in cmp_json.get("samples") or []:
        true, pred = s.get("true"), s.get("finetuned_pred")
        if true and pred and true != pred:
            out.append({"true": true, "predicted": pred,
                        "text": (s.get("text") or "")[:160]})
    return out


def training_curve(log_json: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-epoch eval rows from training_log.json, for the curve chart."""
    history = log_json.get("log_history") or []
    return [
        {
            "epoch": r.get("epoch"),
            "eval_loss": r.get("eval_loss"),
            "accuracy": r.get("eval_accuracy"),
            "f1_macro": r.get("eval_f1_macro"),
        }
        for r in history
        if "eval_loss" in r
    ]


def best_epoch(curve: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The epoch with the highest macro F1.

    train.py does not pass load_best_model_at_end, so the *saved*
    checkpoint is the final epoch. If that is not also the best epoch,
    the shipped artifact is worse than one that was seen during
    training - the UI flags this explicitly.
    """
    scored = [c for c in curve if c.get("f1_macro") is not None]
    if not scored:
        return None
    return max(scored, key=lambda c: c["f1_macro"])


def saved_is_best(curve: list[dict[str, Any]]) -> bool | None:
    """Whether the final (and therefore saved) epoch is also the best."""
    scored = [c for c in curve if c.get("f1_macro") is not None]
    if len(scored) < 2:
        return None
    return scored[-1]["f1_macro"] >= max(c["f1_macro"] for c in scored)
