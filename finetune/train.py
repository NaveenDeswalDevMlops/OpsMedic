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
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finetune.data import build_label_maps, default_paths, load_split  # noqa: E402
from finetune.device import (
    NonFiniteLoss,
    check_log_row,
    resolve_precision,
    resolve_training_device,
)  # noqa: E402
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
    parser.add_argument("--eval-rows", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=config.FINETUNE_EPOCHS)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1,
                        help="fraction of total training steps used for warmup")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="gradient accumulation steps for an effective larger batch")
    parser.add_argument("--seed", type=int, default=config.FINETUNE_SEED)
    parser.add_argument("--balanced", action="store_true",
                        help="median-downsample classes to fight majority bias")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="select the training device explicitly",
    )
    parser.add_argument(
        "--force-mps",
        action="store_true",
        help="override the MPS safety policy for known-unsafe architectures",
    )
    args = parser.parse_args()

    device, device_notes = resolve_training_device(
        config.CLASSIFIER_BASE_MODEL,
        requested=args.device,
        force_mps=args.force_mps,
    )
    for note in device_notes:
        print(f"[policy] {note}")

    fp16, bf16, precision_notes = resolve_precision(config.CLASSIFIER_BASE_MODEL, device)
    for note in precision_notes:
        print(f"[policy] {note}")

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
        TrainerCallback,
        set_seed,
    )

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.CLASSIFIER_BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.CLASSIFIER_BASE_MODEL,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    ).to(device)
    print(f"[policy] selected device={device}")

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

    num_update_steps_per_epoch = math.ceil(
        len(train_ds) / (args.batch * args.grad_accum)
    )
    warmup_steps = max(1, int(args.warmup_ratio * num_update_steps_per_epoch * args.epochs))

    training_args = TrainingArguments(
        output_dir=os.path.join(paths["artifact_dir"], "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=32,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        max_grad_norm=args.max_grad_norm,
        fp16=fp16,
        bf16=bf16,
        dataloader_pin_memory=(device == "cuda"),
        eval_strategy="epoch",
        logging_steps=25,
        save_strategy="no",  # we save the final model ourselves
        seed=args.seed,
        report_to=[],  # no wandb/tensorboard
    )

    class _AbortOnNaN(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            check_log_row(logs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics_factory(),
        callbacks=[_AbortOnNaN()],
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
                    "batch_size": args.batch,
                    "learning_rate": args.lr,
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
