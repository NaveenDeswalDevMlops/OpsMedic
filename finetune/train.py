# finetune/train.py
"""Fine-tune a transformer encoder for ticket-queue classification.

Trains on the FULL data/finetune_train.csv split by default and evaluates
on the FULL data/finetune_test.csv each epoch. Saves to
CLASSIFIER_FINETUNED_DIR:
    - the best model + tokenizer (auto-detected by models/classifier.py)
    - labels.json         (stable, sorted label order)
    - training_log.json   (loss/metric curve + resolved run config)

Safety rails added after a 105-minute run silently produced NaN weights:
    1. device policy   - MPS is refused for DeBERTa (broken kernels)
    2. precision policy - fp16 only where it does not overflow
    3. preflight        - ~1 min overfit check on 64 rows before the real run
    4. tripwire         - aborts the moment loss or grad-norm goes non-finite

Presets
-------
    large     microsoft/deberta-v3-large  (434M) - best accuracy, needs CUDA
    base      microsoft/deberta-v3-base   (184M) - strong, needs CUDA
    mps-safe  roberta-large               (355M) - trains on Apple Silicon

Usage
-----
    python finetune/train.py                       # full data, auto preset
    python finetune/train.py --preset large        # Kaggle / CUDA
    python finetune/train.py --preset mps-safe     # M-series Mac
    python finetune/train.py --preflight-only      # 1-minute numerics check
    python finetune/train.py --max-rows 2000       # quick subset
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finetune.data import (  # noqa: E402
    build_label_maps,
    class_weights,
    default_paths,
    load_split,
    split_stats,
)
from finetune.device import (  # noqa: E402
    NonFiniteLoss,
    check_log_row,
    resolve_precision,
    resolve_training_device,
)
from src import config  # noqa: E402

MAX_LENGTH = 256

# preset -> (model id, per-device batch, grad accum, learning rate)
PRESETS: dict[str, tuple[str, int, int, float]] = {
    "large": ("microsoft/deberta-v3-large", 8, 4, 8e-6),
    "base": ("microsoft/deberta-v3-base", 16, 2, 2e-5),
    "mps-safe": ("roberta-large", 8, 4, 1e-5),
}


def choose_preset(explicit: str | None) -> str:
    """Pick a preset: explicit wins, else CUDA -> large, Apple -> mps-safe."""
    if explicit:
        return explicit
    try:
        import torch
    except ImportError:
        return "base"
    if torch.cuda.is_available():
        return "large"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps-safe"
    return "base"


def compute_metrics_factory():
    """accuracy + macro-F1 + weighted-F1 for the Trainer (lazy sklearn)."""
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "f1_weighted": f1_score(
                labels, preds, average="weighted", zero_division=0
            ),
        }

    return compute_metrics


def build_dataset(tokenizer, texts: list[str], label_ids: list[int]):
    """Tokenise WITHOUT padding; the collator pads per batch (much faster)."""
    import torch

    class TicketDataset(torch.utils.data.Dataset):
        def __init__(self) -> None:
            self.enc = tokenizer(texts, truncation=True, max_length=MAX_LENGTH)
            self.labels = label_ids

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, i: int) -> dict:
            item = {k: v[i] for k, v in self.enc.items()}
            item["labels"] = self.labels[i]
            return item

    return TicketDataset()


def make_trainer_class(weights: list[float] | None):
    """Trainer subclass applying class weights in the loss (or plain CE)."""
    import torch
    from transformers import Trainer

    class WeightedTrainer(Trainer):
        def compute_loss(
            self, model, inputs, return_outputs=False, **kwargs
        ):  # noqa: ANN001 - signature dictated by transformers
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            if weights is None:
                loss_fn = torch.nn.CrossEntropyLoss()
            else:
                loss_fn = torch.nn.CrossEntropyLoss(
                    weight=torch.tensor(
                        weights, dtype=logits.dtype, device=logits.device
                    )
                )
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            inputs["labels"] = labels
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


def make_tripwire():
    """TrainerCallback that aborts the run on non-finite numerics."""
    from transformers import TrainerCallback

    class TripwireCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
            if logs:
                check_log_row({**logs, "step": state.global_step})

    return TripwireCallback()


def preflight(model_name: str, device: str, num_labels: int, lr: float) -> None:
    """Overfit 64 synthetic-free real rows for 30 steps; assert loss falls.

    Cheap insurance: if the forward/backward pass is numerically broken on
    this device (the MPS/DeBERTa failure mode), this raises in about a
    minute instead of after hours of training.
    """
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        set_seed,
    )

    print(f"\n[preflight] 30 steps on 64 rows, device={device} ...")
    set_seed(0)
    paths = default_paths()
    texts, ids, labels = load_split(
        paths["train_csv"], labels=None, max_rows=64, seed=0
    )
    id2label, label2id = build_label_maps(labels)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, id2label=id2label, label2id=label2id
    ).to(device)
    model.train()
    enc = tok(
        texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt"
    ).to(device)
    y = torch.tensor(ids, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr * 5)
    first = last = None
    for step in range(30):
        opt.zero_grad()
        out = model(**enc, labels=y)
        loss = out.loss
        value = float(loss.detach().cpu())
        if step == 0:
            first = value
        last = value
        if not (value == value) or value in (float("inf"), float("-inf")):
            raise NonFiniteLoss(
                f"[preflight] loss became {value} at step {step} on device="
                f"{device} with model={model_name}.\n"
                "The forward/backward pass is numerically broken on this "
                "device. Do NOT start the real run. Use --device cpu, or "
                "--preset mps-safe, or train on a CUDA GPU."
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    print(f"[preflight] loss {first:.4f} -> {last:.4f}")
    if last >= first:
        raise NonFiniteLoss(
            f"[preflight] loss did not decrease ({first:.4f} -> {last:.4f}). "
            "Optimisation is not working on this device/model combination."
        )
    print("[preflight] OK - numerics are healthy, starting the real run\n")
    del model, opt, enc, y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None)
    parser.add_argument("--model", default=None, help="override the preset model id")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--force-mps", action="store_true",
                        help="train on MPS even for architectures known to NaN")
    parser.add_argument("--max-rows", type=int, default=0,
                        help="0 = FULL train split (default)")
    parser.add_argument("--eval-rows", type=int, default=0,
                        help="0 = FULL test split (default)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=config.FINETUNE_SEED)
    parser.add_argument("--balanced", action="store_true",
                        help="median-downsample instead of class weighting")
    parser.add_argument("--no-class-weight", action="store_true",
                        help="plain cross-entropy (not recommended on this split)")
    parser.add_argument("--no-preflight", action="store_true")
    parser.add_argument("--preflight-only", action="store_true",
                        help="run the 1-minute numerics check and exit")
    args = parser.parse_args()

    preset = choose_preset(args.preset)
    p_model, p_batch, p_accum, p_lr = PRESETS[preset]
    model_name = args.model or os.getenv("CLASSIFIER_BASE_MODEL") or p_model
    batch = args.batch or p_batch
    accum = args.grad_accum or p_accum
    lr = args.lr or p_lr

    device, dev_notes = resolve_training_device(
        model_name, args.device, force_mps=args.force_mps
    )
    fp16, bf16, prec_notes = resolve_precision(model_name, device)
    for note in dev_notes + prec_notes:
        print(f"[policy] {note}")

    paths = default_paths()

    # ---- data ---------------------------------------------------------
    train_texts, train_ids, labels = load_split(
        paths["train_csv"], labels=None, max_rows=args.max_rows,
        seed=args.seed, balanced=args.balanced,
    )
    test_texts, test_ids, _ = load_split(
        paths["test_csv"], labels=labels, max_rows=args.eval_rows, seed=args.seed
    )
    id2label, label2id = build_label_maps(labels)
    weights = None if args.no_class_weight else class_weights(train_ids, len(labels))

    print(
        f"\n[data] train rows: {len(train_texts)} | eval rows: {len(test_texts)} | "
        f"classes: {len(labels)}"
    )
    print(f"{'class':<34}{'rows':>7}{'share':>9}{'weight':>9}")
    for lab, n, share in split_stats(train_ids, labels):
        w = weights[label2id[lab]] if weights else 1.0
        print(f"{lab:<34}{n:>7}{share:>8.1%}{w:>9.3f}")

    if args.preflight_only:
        preflight(model_name, device, len(labels), lr)
        print("preflight-only: done, nothing trained.")
        return

    # ---- model (lazy heavy imports) -----------------------------------
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        TrainingArguments,
        set_seed,
    )

    if not args.no_preflight:
        preflight(model_name, device, len(labels), lr)

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    train_ds = build_dataset(tokenizer, train_texts, train_ids)
    eval_ds = build_dataset(tokenizer, test_texts, test_ids)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    effective = batch * accum
    print(
        f"[train] model={model_name} preset={preset} device={device} "
        f"fp16={fp16} bf16={bf16} batch={batch} grad_accum={accum} "
        f"effective_batch={effective} lr={lr} epochs={args.epochs}"
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(paths["artifact_dir"], "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=max(batch, 16),
        gradient_accumulation_steps=accum,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.06,
        max_grad_norm=1.0,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=10,
        group_by_length=True,
        dataloader_pin_memory=(device == "cuda"),
        fp16=fp16,
        bf16=bf16,
        use_cpu=(device == "cpu"),
        seed=args.seed,
        report_to=[],
    )
    trainer_cls = make_trainer_class(weights)
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics_factory(),
        callbacks=[make_tripwire()],
    )

    # ---- train --------------------------------------------------------
    t0 = time.perf_counter()
    try:
        trainer.train()
    except NonFiniteLoss as exc:
        print(f"\nABORTED: {exc}")
        raise SystemExit(2) from exc
    final_eval = trainer.evaluate()
    train_secs = time.perf_counter() - t0
    print(f"\nTraining finished in {train_secs / 60:.1f} min")
    print(f"Final eval: {json.dumps(final_eval, indent=2)}")

    if not all(
        final_eval.get(k, 0) == final_eval.get(k, 0)
        for k in ("eval_loss", "eval_accuracy", "eval_f1_macro")
    ):
        raise SystemExit("Final eval contains NaN - refusing to save this artifact.")

    # ---- save artifact ------------------------------------------------
    os.makedirs(paths["artifact_dir"], exist_ok=True)
    trainer.save_model(paths["artifact_dir"])
    tokenizer.save_pretrained(paths["artifact_dir"])
    with open(paths["labels_json"], "w", encoding="utf-8") as fh:
        json.dump(labels, fh, indent=2)
    with open(paths["training_log"], "w", encoding="utf-8") as fh:
        json.dump(
            {
                "hyperparameters": {
                    "base_model": model_name,
                    "preset": preset,
                    "device": device,
                    "fp16": fp16,
                    "bf16": bf16,
                    "train_rows": len(train_texts),
                    "eval_rows": len(test_texts),
                    "max_rows": args.max_rows,
                    "balanced_downsample": args.balanced,
                    "class_weighted_loss": weights is not None,
                    "class_weights": weights,
                    "epochs": args.epochs,
                    "batch_size": batch,
                    "grad_accum": accum,
                    "effective_batch": effective,
                    "learning_rate": lr,
                    "warmup_ratio": 0.06,
                    "max_grad_norm": 1.0,
                    "max_length": MAX_LENGTH,
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
