from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tts import TTSTask


def test_tts_falls_back_to_local_wav_when_model_load_fails(tmp_path):
    task = TTSTask()

    def _explode():
        raise RuntimeError("model download blocked")

    task._ensure_loaded = _explode  # type: ignore[assignment]

    out_path = tmp_path / "summary.wav"
    result = task.run({"text": "handover summary", "out_path": str(out_path)})

    assert result["metrics"]["status"] == "ok"
    assert os.path.exists(out_path)
    assert result["output"]["audio_path"] == str(out_path)
    assert result["output"]["duration_s"] >= 0
