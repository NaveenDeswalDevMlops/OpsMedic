# models/tts.py
"""Sub-task 6 (Speech): read the resolution / summary aloud.

Converts the generated resolution (or its summary) into a WAV file so a
field engineer can listen hands-free.

Model: espnet/kan-bayashi_ljspeech_vits. Chosen for higher-quality English VITS synthesis with local inference and straightforward tokenizer support.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Any

import numpy as np
import soundfile as sf

from models.base import BaseSubTask, resolve_device
from src import config
from src import config

MAX_TTS_CHARS = 600  # keep clips short and generation fast
DEFAULT_OUT = "./data/tts_out.wav"


def _is_kokoro_available() -> bool:
    try:
        import kokoro  # noqa: F401
    except Exception:
        return False
    return True


class TTSTask(BaseSubTask):
    name = "tts"
    category = "Speech"
    cacheable = False  # output is a binary file, not JSON-cacheable

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model_name = config.TTS_MODEL
        self._model = None
        self._tokenizer = None
        self._device = "cpu"

    def _ensure_loaded(self) -> None:
        if self._model is None:
            import torch
            from transformers import AutoTokenizer, VitsModel  # lazy

            device = resolve_device()
            if self.model_name == "hexgrad/Kokoro-82M" and _is_kokoro_available():
                from kokoro import KPipeline

                self._model = KPipeline(lang_code=config.KOKORO_LANG)
                self._tokenizer = None
                self._device = device
                return

            self._model = VitsModel.from_pretrained(
                self.model_name,
                local_files_only=True,
            ).to(device)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=True,
            )
            self._device = device

    def _write_fallback_audio(self, text: str, out_path: str) -> dict[str, Any]:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

        # Prefer a local macOS voice engine if it is available.
        if os.name == "posix":
            for voice in ("Samantha", "Alex", "Daniel"):
                try:
                    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
                        fh.write(text)
                        tmp_txt = fh.name
                    with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as fh:
                        tmp_audio = fh.name
                    try:
                        subprocess.run(
                            ["say", "-v", voice, "-r", "190", "-o", tmp_audio, "-f", tmp_txt],
                            check=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        data, sr = sf.read(tmp_audio)
                        sf.write(out_path, data, sr)
                        return {
                            "audio_path": out_path,
                            "sampling_rate": sr,
                            "duration_s": round(len(data) / sr, 2),
                            "engine": f"macos-say:{voice}",
                            "fallback": True,
                        }
                    finally:
                        for path in (tmp_txt, tmp_audio):
                            try:
                                os.unlink(path)
                            except FileNotFoundError:
                                pass
                except (FileNotFoundError, subprocess.CalledProcessError, RuntimeError):
                    continue

        # Deterministic offline fallback: synthesize a more speech-like tone stream.
        sr = 22050
        duration_s = max(0.9, min(3.5, 0.06 * max(1, len(text))))
        t = np.linspace(0.0, duration_s, int(sr * duration_s), endpoint=False)
        envelope = np.sin(np.pi * t / max(duration_s, 1e-6))
        carrier = 220.0
        wave = np.zeros_like(t)
        for idx, ch in enumerate(text):
            if not ch.isalnum():
                continue
            freq = carrier * (1.0 + 0.02 * idx)
            wave += 0.05 * np.sin(2 * np.pi * freq * t + 0.08 * idx)
        wave = np.clip(wave * envelope, -0.8, 0.8).astype(np.float32)
        sf.write(out_path, wave, sr)
        return {
            "audio_path": out_path,
            "sampling_rate": sr,
            "duration_s": round(len(wave) / sr, 2),
            "engine": "tone-stream",
            "fallback": True,
        }

    def _run(self, payload: Any) -> dict[str, Any]:
        """payload: str, or {"text": str, "out_path": str}.

        Returns {"audio_path", "sampling_rate", "duration_s"}."""
        import torch  # lazy

        if isinstance(payload, dict):
            text = str(payload.get("text", "")).strip()
            out_path = payload.get("out_path") or DEFAULT_OUT
        else:
            text, out_path = str(payload).strip(), DEFAULT_OUT
        if not text:
            raise ValueError("empty text")

        try:
            self._ensure_loaded()
            if self.model_name == "hexgrad/Kokoro-82M" and _is_kokoro_available() and self._tokenizer is None:
                import soundfile as sf_local
                import torch
                import numpy as np

                generator = self._model(text[:MAX_TTS_CHARS], voice=config.KOKORO_VOICE, speed=config.KOKORO_SPEED)
                audio = np.concatenate([np.array(chunk) for chunk in generator if chunk is not None], axis=0)
                if audio.size == 0:
                    raise RuntimeError("kokoro produced no audio")
                audio = audio.astype(np.float32)
                sr = 24000
                os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
                sf_local.write(out_path, audio, sr)
                return {
                    "audio_path": out_path,
                    "sampling_rate": sr,
                    "duration_s": round(len(audio) / sr, 2),
                }

            inputs = self._tokenizer(text[:MAX_TTS_CHARS], return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
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
                "engine": self.model_name,
                "fallback": False,
            }
        except Exception as exc:  # noqa: BLE001
            # Never fail the journey, but never hide the degradation either:
            # the fallback is macOS `say` or a tone stream, NOT real synthesis.
            import warnings

            warnings.warn(
                f"TTS model {self.model_name!r} failed ({type(exc).__name__}: "
                f"{exc}); falling back to non-VITS audio. The report must not "
                f"claim VITS synthesis for this output.",
                RuntimeWarning,
                stacklevel=2,
            )
            out = self._write_fallback_audio(text, out_path)
            out["fallback_reason"] = f"{type(exc).__name__}: {exc}"
            return out
