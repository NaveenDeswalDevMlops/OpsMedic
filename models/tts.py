# models/tts.py
"""Sub-task 6 (Speech): read the resolution / summary aloud.

Converts the generated resolution (or its summary) into a WAV file so a
field engineer can listen hands-free.

Model: facebook/mms-tts-eng (VITS, ~140 MB). Chosen for: fully local,
open-licensed English TTS with simple one-pass inference and no speaker
embeddings or vocoder setup required.
"""
from __future__ import annotations

import os
from typing import Any

from models.base import BaseSubTask
from src import config

MAX_TTS_CHARS = 600  # keep clips short and generation fast
DEFAULT_OUT = "./data/tts_out.wav"


class TTSTask(BaseSubTask):
    name = "tts"
    category = "Speech"
    cacheable = False  # output is a binary file, not JSON-cacheable

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model_name = config.TTS_MODEL
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from transformers import AutoTokenizer, VitsModel  # lazy

            self._model = VitsModel.from_pretrained(self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def _run(self, payload: Any) -> dict[str, Any]:
        """payload: str, or {"text": str, "out_path": str}.

        Returns {"audio_path", "sampling_rate", "duration_s"}."""
        import soundfile as sf  # lazy
        import torch  # lazy

        if isinstance(payload, dict):
            text = str(payload.get("text", "")).strip()
            out_path = payload.get("out_path") or DEFAULT_OUT
        else:
            text, out_path = str(payload).strip(), DEFAULT_OUT
        if not text:
            raise ValueError("empty text")

        self._ensure_loaded()
        inputs = self._tokenizer(text[:MAX_TTS_CHARS], return_tensors="pt")
        with torch.no_grad():
            waveform = self._model(**inputs).waveform

        audio = waveform.squeeze().cpu().numpy()
        sr = int(self._model.config.sampling_rate)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        sf.write(out_path, audio, sr)
        return {
            "audio_path": out_path,
            "sampling_rate": sr,
            "duration_s": round(len(audio) / sr, 2),
        }
