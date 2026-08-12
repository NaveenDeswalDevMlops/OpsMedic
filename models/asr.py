# models/asr.py
"""Sub-task 5 (Speech Recognition): voice-reported incident -> text.

An engineer/user records the incident by voice (Streamlit mic input or
an uploaded audio file); Whisper transcribes it into the incident text
that feeds the rest of the pipeline.

Model: openai/whisper-large. Chosen for stronger transcription quality on GPU machines; whisper-tiny is still available for low-resource local CPU use.
"""
from __future__ import annotations

import time as _time

from typing import Any

import numpy as np

from models.base import BaseSubTask, resolve_device
from src import config

TARGET_SR = 16_000  # Whisper's expected sampling rate


def to_mono_16k(audio: "np.ndarray", sr: int) -> "np.ndarray":
    """Downmix to mono float32 and resample to 16 kHz (pure, testable)."""
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 2:  # (frames, channels) -> mono
        arr = arr.mean(axis=1)
    if sr != TARGET_SR:
        import librosa  # lazy

        arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
    return arr


class ASRTask(BaseSubTask):
    name = "asr"
    category = "Speech"
    cacheable = False  # raw audio payloads are not cache-key friendly

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model_name = config.ASR_MODEL
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

            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                chunk_length_s=30,
                device=pipeline_device,
            )

    def _run(self, payload: Any) -> str:
        """payload: path to an audio file (wav/flac/ogg) OR a dict
        {"array": np.ndarray, "sampling_rate": int}. Returns transcript."""
        self._ensure_loaded()
        if isinstance(payload, dict) and "array" in payload:
            audio = to_mono_16k(payload["array"], int(payload["sampling_rate"]))
        else:
            import soundfile as sf  # lazy

            data, sr = sf.read(str(payload))
            audio = to_mono_16k(data, sr)
        if audio.size < TARGET_SR // 4:  # < 0.25 s
            raise ValueError("audio too short to transcribe")
        audio_s = audio.size / TARGET_SR
        t0 = _time.perf_counter()
        result = self._pipe(
            {"array": audio, "sampling_rate": TARGET_SR},
            generate_kwargs={"language": "english", "task": "transcribe"},
        )
        decode_s = _time.perf_counter() - t0
        transcript = str(result["text"]).strip()
        # Real-time factor is the standard ASR efficiency metric:
        # decode time divided by audio duration. RTF < 1 means faster
        # than real time. whisper-large on CPU sits far above 1, which
        # is a deliberate accuracy-for-speed trade we report rather
        # than hide.
        self.report_signals(
            asr_audio_s=round(audio_s, 3),
            asr_decode_s=round(decode_s, 3),
            asr_rtf=round(decode_s / audio_s, 3) if audio_s > 0 else None,
            asr_chars=len(transcript),
        )
        return transcript
