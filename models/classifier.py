# models/classifier.py
"""Sub-task 2 (NLP): ticket queue classification - the fine-tuned sub-task.

Predicts which queue/category a new incident belongs to. Two variants,
switchable at construction (and via a UI toggle in the app):

  variant="base"       CLASSIFIER_BASE_MODEL (microsoft/deberta-v3-base)
                       with a randomly initialized 10-class head. Note
                       this is a chance-level floor, not a tuned
                       competitor: the head is re-randomised on every
                       construction, so its score moves between runs.
                       Compare against the majority-class baseline
                       (0.289 on our test split) for a stable reference.
  variant="finetuned"  the same architecture after finetune/train.py has
                       trained it on the dataset's train split -> the
                       "after". Loaded from CLASSIFIER_FINETUNED_DIR.
  variant="auto"       finetuned if the artifact exists, else base.

Model: DeBERTa-v3-base (184M params), read from config so the choice is
never hard-coded here. Chosen for: disentangled attention and
ELECTRA-style pre-training give it a clear edge over BERT/DistilBERT at
the same depth on short, jargon-dense ticket text, while still being
small enough to fine-tune in a free GPU session.
"""
from __future__ import annotations

import json
import os
from typing import Any

from models.base import BaseSubTask, resolve_device
from src import config

# Fallback label set (the dataset's top-10 queues) used only when neither
# a fine-tuned artifact nor the train CSV is available to derive labels.
DEFAULT_LABELS = [
    "Billing and Payments", "Customer Service", "General Inquiry",
    "Human Resources", "IT Support", "Product Support",
    "Returns and Exchanges", "Sales and Pre-Sales",
    "Service Outages and Maintenance", "Technical Support",
]
MAX_INPUT_CHARS = 2000


def resolve_labels() -> list[str]:
    """Determine the label list (sorted for a stable id<->label mapping).

    Priority: labels.json saved by finetune/train.py > unique categories
    in the train split CSV > DEFAULT_LABELS. Pure logic, testable offline.
    """
    labels_json = os.path.join(config.CLASSIFIER_FINETUNED_DIR, "labels.json")
    if os.path.isfile(labels_json):
        with open(labels_json, "r", encoding="utf-8") as fh:
            return sorted(json.load(fh))
    if os.path.isfile(config.FINETUNE_TRAIN_CSV):
        import pandas as pd  # lazy

        cats = pd.read_csv(config.FINETUNE_TRAIN_CSV)["category"].dropna()
        labels = sorted(set(cats.astype(str)))
        if labels:
            return labels
    return list(DEFAULT_LABELS)


def pick_variant(requested: str) -> str:
    """'auto' -> 'finetuned' if the artifact exists, else 'base'."""
    if requested not in ("auto", "base", "finetuned"):
        raise ValueError(f"unknown variant: {requested}")
    if requested != "auto":
        return requested
    has_artifact = os.path.isfile(
        os.path.join(config.CLASSIFIER_FINETUNED_DIR, "config.json")
    )
    return "finetuned" if has_artifact else "base"


def artifact_version_stamp(artifact_dir: str) -> str:
    """Version stamp derived from labels.json mtime ('' if absent).

    Appended to the classifier's model identity so (a) the response
    cache auto-invalidates after every retrain and (b) the metrics
    dashboard records WHICH model version served each request.
    """
    import time as _time

    labels_json = os.path.join(artifact_dir, "labels.json")
    if not os.path.isfile(labels_json):
        return ""
    mtime = os.path.getmtime(labels_json)
    return "@" + _time.strftime("%Y%m%d-%H%M%S", _time.localtime(mtime))


class ClassifierTask(BaseSubTask):
    name = "classify"
    category = "NLP"
    cacheable = True

    def __init__(self, variant: str = "auto", *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.variant = pick_variant(variant)
        self.labels = resolve_labels()
        if self.variant == "finetuned":
            self._model_path = config.CLASSIFIER_FINETUNED_DIR
            self.model_name = (
                config.CLASSIFIER_FINETUNED_DIR
                + artifact_version_stamp(config.CLASSIFIER_FINETUNED_DIR)
            )
        else:
            self._model_path = config.CLASSIFIER_BASE_MODEL
            self.model_name = config.CLASSIFIER_BASE_MODEL
        self._pipe = None

    def _ensure_loaded(self) -> None:
        if self._pipe is not None:
            return
        from transformers import (  # lazy heavy imports
            AutoModelForSequenceClassification,
            AutoTokenizer,
            pipeline,
        )

        device = resolve_device()
        id2label = {i: lab for i, lab in enumerate(self.labels)}
        label2id = {lab: i for i, lab in enumerate(self.labels)}
        model = AutoModelForSequenceClassification.from_pretrained(
            self._model_path,
            num_labels=len(self.labels),
            id2label=id2label,
            label2id=label2id,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            config.CLASSIFIER_BASE_MODEL
            if self.variant == "base"
            else self._model_path
        )
        pipeline_device: int | str
        if device == "cuda":
            pipeline_device = 0
        elif device == "mps":
            pipeline_device = "mps"
        else:
            pipeline_device = -1
        self._pipe = pipeline(
            "text-classification", model=model, tokenizer=tokenizer, device=pipeline_device
        )

    def _run(self, payload: Any) -> dict[str, Any]:
        text = str(payload).strip()
        if not text:
            raise ValueError("empty text")
        self._ensure_loaded()
        pred = self._pipe(text[:MAX_INPUT_CHARS], truncation=True)[0]
        return {
            "label": pred["label"],
            "confidence": round(float(pred["score"]), 4),
            "variant": self.variant,
        }
