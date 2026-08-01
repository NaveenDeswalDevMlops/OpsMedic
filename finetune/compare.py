# finetune/compare.py
"""Base vs fine-tuned comparison on the held-out test split.

Produces the before/after evidence required by the assignment:
  - metrics table (accuracy, macro-F1) for both variants
  - sample predictions side by side on the same tickets
  - saves everything to finetune/artifacts/.../compare.json, which the
    Streamlit app renders in its Fine-tuning tab.

Usage (after finetune/train.py):
    python finetune/compare.py
    python finetune/compare.py --max-rows 500 --samples 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finetune.data import default_paths, load_split  # noqa: E402
from finetune.evaluate import evaluate_model, predict_batched  # noqa: E402
from src import config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # 0 = the FULL held-out split. Capping the eval set was a CPU-era
    # compromise; on a GPU it costs seconds and tighter numbers are more
    # credible in the report.
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=config.FINETUNE_SEED)
    args = parser.parse_args()

    paths = default_paths()
    if not os.path.isfile(paths["labels_json"]):
        sys.exit("No fine-tuned artifact. Run: python finetune/train.py first.")
    with open(paths["labels_json"], "r", encoding="utf-8") as fh:
        labels = sorted(json.load(fh))

    print("Evaluating BASE (untrained head) ...")
    base = evaluate_model(
        config.CLASSIFIER_BASE_MODEL, labels, args.max_rows, args.seed
    )
    print("Evaluating FINE-TUNED ...")
    tuned = evaluate_model(paths["artifact_dir"], labels, args.max_rows, args.seed)

    # ---- headline table -------------------------------------------------
    print("\n================ BASE vs FINE-TUNED ================")
    print(f"{'metric':<12} {'base':>10} {'fine-tuned':>12} {'delta':>10}")
    for metric in ("accuracy", "f1_macro"):
        b, t = base[metric], tuned[metric]
        print(f"{metric:<12} {b:>10.4f} {t:>12.4f} {t - b:>+10.4f}")
    print(f"(held-out rows evaluated: {tuned['eval_rows']})")

    # ---- sample predictions ---------------------------------------------
    texts, y_true, _ = load_split(
        paths["test_csv"], labels=labels, max_rows=args.samples, seed=args.seed + 1
    )
    base_preds = predict_batched(config.CLASSIFIER_BASE_MODEL, labels, texts)
    tuned_preds = predict_batched(paths["artifact_dir"], labels, texts)
    samples = []
    print("\n---------------- sample predictions ----------------")
    for text, yt, pb, pt in zip(texts, y_true, base_preds, tuned_preds):
        row = {
            "text": text[:140],
            "true": labels[yt],
            "base_pred": labels[pb],
            "finetuned_pred": labels[pt],
        }
        samples.append(row)
        mark_b = "OK" if pb == yt else "X "
        mark_t = "OK" if pt == yt else "X "
        print(f"true={row['true']:<32} base[{mark_b}]={row['base_pred']:<32} "
              f"ft[{mark_t}]={row['finetuned_pred']}")

    # ---- persist for the app / report -----------------------------------
    out = {"base": base, "finetuned": tuned, "samples": samples}
    with open(paths["compare_json"], "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved comparison -> {paths['compare_json']}")
    print("Screenshot this output for the report (fine-tuning section).")


if __name__ == "__main__":
    main()
