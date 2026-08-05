# models/summarizer.py
"""Sub-task 4 (NLP): ticket / resolution summarization.

Condenses a long ticket thread or generated resolution into a 2-3
sentence handover summary (for shift notes and the TTS read-out).

Model: sshleifer/distilbart-cnn-12-6 (distilled BART, ~300 MB).
Chosen for: near-BART ROUGE quality at ~2x speed on CPU, free/local.
"""
from __future__ import annotations

from typing import Any

from models.base import BaseSubTask, resolve_device
from src import config

MAX_INPUT_CHARS = 3500  # ~ model's 1024-token limit with margin


class SummarizerTask(BaseSubTask):
    name = "summarize"
    category = "NLP"
    cacheable = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model_name = config.SUMMARIZER_MODEL
        self._pipe = None

    def _ensure_loaded(self) -> None:
        if self._pipe is None:
            from transformers import pipeline  # lazy heavy import

            device = resolve_device()
            pipeline_device: int | str
            if device == "cuda":
                pipeline_device = 0
            elif device == "mps":
                pipeline_device = "mps"
            else:
                pipeline_device = -1
            self._pipe = pipeline("summarization", model=self.model_name, device=pipeline_device)

    def _run(self, payload: Any) -> str:
        text = str(payload).strip()
        if len(text) < 40:
            raise ValueError("text too short to summarize (min 40 chars)")
        self._ensure_loaded()
        result = self._pipe(
            text[:MAX_INPUT_CHARS],
            max_length=120,
            min_length=25,
            do_sample=False,
            truncation=True,
        )
        return result[0]["summary_text"].strip()
