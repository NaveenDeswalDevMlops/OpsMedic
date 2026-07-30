# models/tts.py
"""Sub-task 6 (Speech): read the resolution / summary aloud.

Converts the generated resolution (or its summary) into a WAV file so a
field engineer can listen hands-free.

Model: facebook/mms-tts-eng (VITS, ~140 MB). Chosen for: fully local,
open-licensed English TTS with simple one-pass inference and no speaker
embeddings or vocoder setup required.

Delivery is shaped by models/tts_prosody.py rather than by the model,
because a small VITS has no prosody or style conditioning:

  * text is split into sentences and numbered steps, so each is a
    separate utterance the model can phrase ("1. Clear the SSO cache"
    is spoken as "First, Clear the SSO cache"), and markdown is stripped
    instead of read out;
  * per-chunk silence padding is trimmed and the chunks rejoined with
    deliberate pauses, which removes the dead air that made a single
    600-char pass sound halting (measured at 34% of the clip);
  * the assembled audio gets a rumble cut and a 1.5-4 kHz presence lift.
    Measured on real output: 83% of energy sat below 500 Hz with only
    2.7% in the consonant band, so words arrived mushy. That is a
    clarity problem, not harshness — hence no high-frequency softening
    by default (config.TTS_SOFTEN = 0.0).

Every tunable lives in src/config.py so it is overridable from .env, and
per-call overrides are accepted in the payload for a UI tone control.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from models.base import BaseSubTask
from models.tts_prosody import (
    finalize,
    flatten_for_speech,
    presence_ratio,
    render_chunks,
    split_for_speech,
)
from src import config

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
        self._kokoro = None
        self._knobs: dict[str, float] = {}
        self._sr: int | None = None

    # -- backend -------------------------------------------------------
    @property
    def backend(self) -> str:
        """'kokoro' or 'vits', chosen from the model name.

        Kokoro-82M is a different architecture (not a VitsModel) with its
        own pipeline object and voice presets, so it needs a separate
        loader rather than a config tweak.
        """
        return "kokoro" if "kokoro" in self.model_name.lower() else "vits"

    # -- model ---------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None or self._kokoro is not None:
            return
        if self.backend == "kokoro":
            self._load_kokoro()
        else:
            self._load_vits()

    def _load_vits(self) -> None:
        from transformers import AutoTokenizer, VitsModel  # lazy

        self._model = VitsModel.from_pretrained(self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        device = config.resolve_device()
        try:
            self._model = self._model.to(device)
        except Exception:  # noqa: BLE001 - fall back rather than fail the demo
            device = "cpu"
        self._device = device
        self._sr = int(self._model.config.sampling_rate)
        self._apply_vits_knobs()

    def _load_kokoro(self) -> None:
        """Load Kokoro-82M. Requires `pip install kokoro` and espeak-ng."""
        try:
            from kokoro import KPipeline  # lazy, optional dependency
        except ImportError as exc:  # noqa: BLE001
            raise RuntimeError(
                "TTS_MODEL is set to Kokoro but the 'kokoro' package is not "
                "installed. Run: pip install kokoro   (and install espeak-ng: "
                "macOS `brew install espeak-ng`, Debian/Ubuntu "
                "`apt-get install espeak-ng`). Set "
                "TTS_MODEL=facebook/mms-tts-eng to use the built-in voice."
            ) from exc
        self._kokoro = KPipeline(lang_code=config.KOKORO_LANG)
        self._device = config.resolve_device()
        self._sr = 24000          # Kokoro outputs 24 kHz

    def _apply_vits_knobs(self) -> None:
        """Set VITS sampling knobs, tolerating attribute renames.

        transformers exposes speaking_rate / noise_scale /
        noise_scale_duration on VitsModel, but the names have moved
        between versions, so each is set only if present and what
        actually applied is reported in the output for the record.
        """
        if self.backend != "vits" or self._model is None:
            self._knobs = {}
            return
        wanted = {
            "speaking_rate": config.TTS_SPEAKING_RATE,
            "noise_scale": config.TTS_NOISE_SCALE,
            "noise_scale_duration": config.TTS_NOISE_SCALE_DURATION,
        }
        applied: dict[str, float] = {}
        for attr, value in wanted.items():
            for target in (self._model, getattr(self._model, "config", None)):
                if target is not None and hasattr(target, attr):
                    try:
                        setattr(target, attr, float(value))
                        applied[attr] = float(value)
                    except Exception:  # noqa: BLE001 - never break synthesis
                        pass
                    break
        self._knobs = applied

    def _synth(self, text: str) -> np.ndarray:
        """Synthesize one chunk to a float32 mono waveform."""
        if self.backend == "kokoro":
            return self._synth_kokoro(text)
        return self._synth_vits(text)

    def _synth_vits(self, text: str) -> np.ndarray:
        import torch  # lazy

        inputs = self._tokenizer(text, return_tensors="pt")
        device = getattr(self, "_device", "cpu")
        if device != "cpu":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            waveform = self._model(**inputs).waveform
        return waveform.squeeze().detach().cpu().numpy().astype(np.float32)

    def _synth_kokoro(self, text: str) -> np.ndarray:
        """Kokoro yields (graphemes, phonemes, audio) per segment.

        Tolerant of the exact yield shape: anything array-like in the
        tuple is treated as the waveform, so a minor API change in the
        kokoro package degrades to a clear error rather than silence.
        """
        pieces: list[np.ndarray] = []
        for item in self._kokoro(text, voice=config.KOKORO_VOICE,
                                speed=config.KOKORO_SPEED):
            audio = item[-1] if isinstance(item, (tuple, list)) else item
            if hasattr(audio, "detach"):          # torch tensor
                audio = audio.detach().cpu().numpy()
            arr = np.asarray(audio, dtype=np.float32).squeeze()
            if arr.ndim > 1:
                arr = arr.mean(axis=tuple(range(1, arr.ndim)))
            if arr.size:
                pieces.append(arr)
        if not pieces:
            raise RuntimeError(
                f"Kokoro returned no audio for voice {config.KOKORO_VOICE!r}. "
                "Check the voice id is valid for KOKORO_LANG="
                f"{config.KOKORO_LANG!r}."
            )
        return np.concatenate(pieces).astype(np.float32)

    # -- sub-task ------------------------------------------------------
    def _run(self, payload: Any) -> dict[str, Any]:
        """payload: str, or {"text", "out_path", and optional tone keys
        "presence", "rumble", "soften", "peak", "max_chars"}.

        Returns {"audio_path", "sampling_rate", "duration_s", "chunks",
        "truncated", "tone", "vits_knobs", "presence_ratio"}.
        """
        import soundfile as sf  # lazy

        if isinstance(payload, dict):
            text = str(payload.get("text", "")).strip()
            out_path = payload.get("out_path") or DEFAULT_OUT
        else:
            text, out_path = str(payload).strip(), DEFAULT_OUT
        if not text:
            raise ValueError("empty text")

        opts = payload if isinstance(payload, dict) else {}
        presence = float(opts.get("presence", config.TTS_PRESENCE))
        rumble = float(opts.get("rumble", config.TTS_RUMBLE_CUT))
        soften_amount = float(opts.get("soften", config.TTS_SOFTEN))
        peak = float(opts.get("peak", config.TTS_PEAK))
        budget = int(opts.get("max_chars", config.TTS_MAX_CHARS))

        chunked = bool(opts.get("chunked", config.TTS_CHUNKED))

        if not chunked:
            spoken = flatten_for_speech(text, budget)
            if not spoken:
                raise ValueError("text contained no speakable content")
        else:
            chunks = split_for_speech(text)
            if not chunks:
                raise ValueError("text contained no speakable content")

        self._ensure_loaded()
        sr = int(self._sr or 16000)

        if chunked:
            audio, rendered, truncated = render_chunks(
                chunks, self._synth, sr, char_budget=budget)
        else:
            # single continuous pass: the model phrases the whole passage,
            # which is the pipeline the approved tone sample used
            audio = np.asarray(self._synth(spoken), dtype=np.float32)
            rendered, truncated = 1, len(spoken) < len(text)
        if audio.size == 0:
            raise RuntimeError("synthesis produced no audio")

        audio = finalize(audio, sr, presence=presence, rumble=rumble,
                         soften_amount=soften_amount, peak=peak)

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        sf.write(out_path, audio, sr)
        return {
            "audio_path": out_path,
            "sampling_rate": sr,
            "duration_s": round(len(audio) / sr, 2),
            "chunks": rendered,
            "chunked": chunked,
            "truncated": truncated,
            "tone": {"presence": presence, "rumble": rumble,
                     "soften": soften_amount, "peak": peak},
            "backend": self.backend,
            "device": getattr(self, "_device", "cpu"),
            "voice": config.KOKORO_VOICE if self.backend == "kokoro" else "default",
            "vits_knobs": dict(self._knobs),
            "presence_ratio": round(presence_ratio(audio, sr), 4),
        }