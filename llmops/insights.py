# llmops/insights.py
"""Pure data-shaping helpers behind the Streamlit dashboard pages.

No streamlit/torch imports - fully unit-testable offline. Pages in
pages/ call these and only handle rendering.
"""
from __future__ import annotations

import json
from typing import Any


def feedback_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Tally 👍/👎 from metrics rows (subtask='feedback').

    Each feedback row stores {"value": +1|-1, ...} in its `extra` JSON.
    Malformed rows are ignored.
    """
    up = down = 0
    for row in rows:
        if row.get("subtask") != "feedback":
            continue
        try:
            extra = json.loads(row.get("extra") or "{}")
            value = int(extra.get("value", 0))
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
        if value > 0:
            up += 1
        elif value < 0:
            down += 1
    return {"up": up, "down": down, "total": up + down}


def training_curves(log_history: list[dict[str, Any]]) -> dict[str, list]:
    """Extract plottable series from HF Trainer's log_history.

    Returns {"train": [(epoch, loss), ...],
             "eval":  [(epoch, accuracy, f1_macro, eval_loss), ...]}
    """
    train: list[tuple[float, float]] = []
    evals: list[tuple[float, float, float, float]] = []
    for entry in log_history or []:
        if "loss" in entry and "eval_loss" not in entry:
            train.append((float(entry.get("epoch", 0)), float(entry["loss"])))
        elif "eval_loss" in entry:
            evals.append(
                (
                    float(entry.get("epoch", 0)),
                    float(entry.get("eval_accuracy", 0.0)),
                    float(entry.get("eval_f1_macro", 0.0)),
                    float(entry["eval_loss"]),
                )
            )
    return {"train": train, "eval": evals}


def per_class_table(compare: dict[str, Any], variant: str) -> list[dict[str, Any]]:
    """Flatten compare.json's per_class dict into sortable rows."""
    out = []
    for label, stats in (compare.get(variant, {}).get("per_class") or {}).items():
        out.append({"class": label, **stats})
    return sorted(out, key=lambda r: r["f1"], reverse=True)


def headline_delta(compare: dict[str, Any]) -> list[dict[str, Any]]:
    """Base-vs-finetuned headline rows for the comparison table."""
    rows = []
    for metric in ("accuracy", "f1_macro"):
        b = float(compare.get("base", {}).get(metric, 0.0))
        t = float(compare.get("finetuned", {}).get(metric, 0.0))
        rows.append(
            {"metric": metric, "base": b, "finetuned": t, "delta": round(t - b, 4)}
        )
    return rows
