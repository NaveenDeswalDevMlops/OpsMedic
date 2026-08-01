# finetune/train.py
"""Fine-tune DistilBERT for ticket-queue classification.

Trains distilbert-base-uncased on data/finetune_train.csv (stratified
subset for speed) and evaluates each epoch on data/finetune_test.csv.
Saves to CLASSIFIER_FINETUNED_DIR:
    - the model + tokenizer  (auto-detected by models/classifier.py)
    - labels.json            (stable label order)
    - training_log.json      (loss/metric curve for the report)

Usage (defaults are CPU-friendly: 2000 rows, 2 epochs, ~10-15 min):
    python finetune/train.py
    python finetune/train.py --max-rows 4000 --epochs 3
    python finetune/train.py --max-rows 0        # full train split
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finetune.data import build_label_maps, default_paths, load_split  # noqa: E402
from src import config  # noqa: E402


def compute_metrics_factory():
    """accuracy + macro-F1 for the Trainer (lazy sklearn import)."""
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
        }

    return compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rows", type=int, default=config.FINETUNE_MAX_ROWS)
    parser.add_argument("--eval-rows", type=int, default=0)  # 0 = full test split
    parser.add_argument("--epochs", type=int, default=config.FINETUNE_EPOCHS)
    # batch / lr / grad-accum default to None and are resolved from the base
    # model below, because a -large encoder needs materially gentler settings
    # than a -base one and hard-coded defaults silently ruin one of the two.
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--grad-accum", type=int, default=None,
                        help="gradient accumulation steps (keeps the effective "
                             "batch at 32 when per-device batch is smaller)")
    parser.add_argument("--seed", type=int, default=config.FINETUNE_SEED)
    parser.add_argument("--balanced", action="store_true",
                        help="median-downsample classes to fight majority bias")
    args = parser.parse_args()

    # ---- settings that depend on the base model ----------------------
    big = config.is_large_encoder()
    if args.batch is None:
        args.batch = 16 if big else 32
    if args.lr is None:
        # -large is even more LR-sensitive than -base; 2e-5 overshoots it
        args.lr = 1e-5 if big else 2e-5
    if args.grad_accum is None:
        args.grad_accum = 2 if big else 1

    paths = default_paths()

    # ---- data -------------------------------------------------------
    train_texts, train_ids, labels = load_split(
        paths["train_csv"], labels=None, max_rows=args.max_rows,
        seed=args.seed, balanced=args.balanced,
    )
    test_texts, test_ids, _ = load_split(
        paths["test_csv"], labels=labels, max_rows=args.eval_rows, seed=args.seed
    )
    id2label, label2id = build_label_maps(labels)
    print(
        f"Train rows: {len(train_texts)} | Eval rows: {len(test_texts)} | "
        f"Classes: {len(labels)}"
    )

    # ---- model (lazy heavy imports) -----------------------------------
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.CLASSIFIER_BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.CLASSIFIER_BASE_MODEL,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    class TicketDataset(torch.utils.data.Dataset):
        def __init__(self, texts: list[str], label_ids: list[int]) -> None:
            self.enc = tokenizer(
                texts, truncation=True, padding="max_length", max_length=256
            )
            self.labels = label_ids

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, i: int) -> dict:
            item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[i])
            return item

    train_ds = TicketDataset(train_texts, train_ids)
    eval_ds = TicketDataset(test_texts, test_ids)

    device = config.resolve_device()
    fp16 = config.train_fp16()
    if config.use_fp16() and not fp16:
        print(f"[train] fp16 DISABLED for {config.CLASSIFIER_BASE_MODEL}: this "
              "checkpoint NaNs in half precision (see config.FP16_UNSTABLE). "
              "Force it with TRAIN_FP16=1 if you want to try anyway.")
    print(f"[train] device={device} fp16={fp16} batch={args.batch} "
          f"grad_accum={args.grad_accum} lr={args.lr} "
          f"effective_batch={args.batch * args.grad_accum} "
          f"base={config.CLASSIFIER_BASE_MODEL}")
    training_args = TrainingArguments(
        fp16=fp16,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.06,
        weight_decay=0.01,
        output_dir=os.path.join(paths["artifact_dir"], "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=32,
        learning_rate=args.lr,
        eval_strategy="epoch",
        logging_steps=25,
        save_strategy="no",  # we save the final model ourselves
        seed=args.seed,
        report_to=[],  # no wandb/tensorboard
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics_factory(),
    )

    # ---- train --------------------------------------------------------
    t0 = time.perf_counter()
    trainer.train()
    final_eval = trainer.evaluate()
    train_secs = time.perf_counter() - t0
    print(f"\nTraining finished in {train_secs / 60:.1f} min")
    print(f"Final eval: {json.dumps(final_eval, indent=2)}")

    # ---- save artifact --------------------------------------------------
    os.makedirs(paths["artifact_dir"], exist_ok=True)
    trainer.save_model(paths["artifact_dir"])
    tokenizer.save_pretrained(paths["artifact_dir"])
    with open(paths["labels_json"], "w", encoding="utf-8") as fh:
        json.dump(labels, fh, indent=2)
    with open(paths["training_log"], "w", encoding="utf-8") as fh:
        json.dump(
            {
                "hyperparameters": {
                    "base_model": config.CLASSIFIER_BASE_MODEL,
                    "max_rows": args.max_rows,
                    "balanced": args.balanced,
                    "epochs": args.epochs,
                    "learning_rate": args.lr,
                    "batch_size": args.batch,
                    "grad_accum": args.grad_accum,
                    "effective_batch": args.batch * args.grad_accum,
                    "fp16": fp16,
                    "device": device,
                    "max_length": 256,
                    "seed": args.seed,
                },
                "train_seconds": round(train_secs, 1),
                "final_eval": final_eval,
                "log_history": trainer.state.log_history,
            },
            fh,
            indent=2,
        )
    print(f"\nSaved fine-tuned model -> {paths['artifact_dir']}")
    print("Next: python finetune/compare.py   (base vs fine-tuned)")


if __name__ == "__main__":
    main()
