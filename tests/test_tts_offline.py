# tests/test_tts_offline.py
"""Offline tests for the wired TTSTask.

transformers/torch/soundfile are never installed in CI here, and both
are lazy-imported inside TTSTask, so this suite injects a fake
`soundfile` module and substitutes the model + `_synth` on the instance.
That exercises the real orchestration — payload parsing, tone overrides,
chunking, the render loop, finalize(), the file write and the returned
metadata — without downloading a 140 MB checkpoint.

Run:  python tests/test_tts_offline.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---- fake soundfile, captured so tests can inspect what was written ----
WRITES: list[dict] = []


def _install_fake_soundfile() -> None:
    mod = types.ModuleType("soundfile")

    def write(path, data, samplerate, *a, **k):
        arr = np.asarray(data)
        WRITES.append({"path": path, "samples": arr.size,
                       "sr": samplerate,
                       "peak": float(np.max(np.abs(arr))) if arr.size else 0.0})
        with open(path, "wb") as fh:      # touch a real file on disk
            fh.write(b"RIFF")

    mod.write = write
    sys.modules["soundfile"] = mod


_install_fake_soundfile()

from llmops.metrics import MetricsLogger  # noqa: E402
from models.tts import TTSTask  # noqa: E402
from models.tts_prosody import Chunk, render_chunks  # noqa: E402
from src import config  # noqa: E402

SR = 16000


class _FakeVitsConfig:
    sampling_rate = SR
    speaking_rate = 1.0
    noise_scale = 0.667
    noise_scale_duration = 0.8


class _FakeModel:
    def __init__(self) -> None:
        self.config = _FakeVitsConfig()
        self.speaking_rate = 1.0
        self.noise_scale = 0.667
        self.noise_scale_duration = 0.8


def _tone(ms: int, freq: float = 180.0) -> np.ndarray:
    t = np.arange(int(SR * ms / 1000.0)) / SR
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _task(tmpdir: str | None = None) -> TTSTask:
    """A TTSTask with the model faked out and _synth stubbed."""
    db = os.path.join(tmpdir or tempfile.mkdtemp(), "m.db")
    task = TTSTask(metrics=MetricsLogger(db), cache=None)
    task._model = _FakeModel()
    task._tokenizer = object()
    task._ensure_loaded = lambda: None            # type: ignore[method-assign]
    # 60 ms of tone per chunk, padded with silence like a real VITS
    task._synth = lambda text: np.concatenate([      # type: ignore[method-assign]
        np.zeros(int(SR * 0.15), np.float32),
        _tone(60),
        np.zeros(int(SR * 0.15), np.float32),
    ])
    task.calls = []                               # type: ignore[attr-defined]
    real_synth = task._synth

    def counting(text):
        task.calls.append(text)                   # type: ignore[attr-defined]
        return real_synth(text)

    task._synth = counting                        # type: ignore[method-assign]
    return task


def _reset() -> None:
    WRITES.clear()


# --------------------------------------------------------- happy path
def test_string_payload_writes_a_wav_and_reports_metadata():
    _reset()
    task = _task()
    out = task.run("Portal login is failing. Clear the SSO cache.")
    assert out["metrics"]["status"] == "ok", out["metrics"]
    o = out["output"]
    assert o["audio_path"].endswith(".wav")
    assert o["sampling_rate"] == SR
    assert o["duration_s"] > 0
    assert o["chunks"] == 2, o["chunks"]
    assert o["truncated"] is False
    assert len(WRITES) == 1 and WRITES[0]["samples"] > 0


def test_each_sentence_is_synthesized_separately():
    """The whole point of chunking: phrasing per sentence, not one pass."""
    _reset()
    task = _task()
    task.run("First line here. Second line here. Third line here.")
    assert len(task.calls) == 3, task.calls


def test_numbered_steps_are_spoken_as_ordinals():
    _reset()
    task = _task()
    task.run("1. Clear the cache.\n2. Restart the agent.")
    assert task.calls[0].startswith("First,"), task.calls
    assert task.calls[1].startswith("Second,"), task.calls


def test_markdown_is_not_sent_to_the_model():
    _reset()
    task = _task()
    task.run("**Diagnosis:** the `SSO` cache is _stale_.")
    spoken = " ".join(task.calls)
    for ch in "*_`#":
        assert ch not in spoken, f"{ch!r} reached the model"


def test_dead_air_is_trimmed_so_output_is_shorter_than_raw_synthesis():
    """_synth pads 300 ms of silence per chunk; trimming must reclaim it."""
    _reset()
    task = _task()
    out = task.run("One. Two. Three.")
    raw_samples = 3 * (int(SR * 0.15) * 2 + _tone(60).size)
    assert WRITES[0]["samples"] < raw_samples, (
        f"{WRITES[0]['samples']} >= raw {raw_samples}: padding not trimmed")
    assert out["output"]["duration_s"] > 0.1, "over-trimmed to nothing"


def test_peak_is_normalised_to_the_configured_target():
    _reset()
    task = _task()
    task.run("Check the VPN tunnel.")
    assert abs(WRITES[0]["peak"] - config.TTS_PEAK) < 1e-3, WRITES[0]["peak"]


# ------------------------------------------------------ tone overrides
def test_per_call_tone_overrides_are_honoured_and_reported():
    _reset()
    task = _task()
    out = task.run({"text": "Reset the account.", "peak": 0.5,
                    "presence": 0.45, "rumble": 0.2, "soften": 0.1})
    tone = out["output"]["tone"]
    assert tone == {"presence": 0.45, "rumble": 0.2,
                    "soften": 0.1, "peak": 0.5}, tone
    assert abs(WRITES[0]["peak"] - 0.5) < 1e-3


def test_defaults_come_from_config():
    _reset()
    task = _task()
    out = task.run("Restart the print spooler.")
    tone = out["output"]["tone"]
    assert tone["presence"] == config.TTS_PRESENCE
    assert tone["rumble"] == config.TTS_RUMBLE_CUT
    assert tone["soften"] == config.TTS_SOFTEN == 0.0, \
        "softening must stay off by default for this engine"


def test_custom_out_path_is_used():
    _reset()
    tmp = tempfile.mkdtemp()
    target = os.path.join(tmp, "nested", "clip.wav")
    task = _task(tmp)
    out = task.run({"text": "Escalate to network team.", "out_path": target})
    assert out["output"]["audio_path"] == target
    assert os.path.isfile(target), "directory not created / file not written"


# --------------------------------------------------------- budget + errors
def test_char_budget_truncates_on_a_chunk_boundary():
    _reset()
    task = _task()
    text = " ".join(f"Sentence number {i} here." for i in range(1, 21))
    out = task.run({"text": text, "max_chars": 60})
    o = out["output"]
    assert o["truncated"] is True
    assert o["chunks"] < 20
    # the cut is on a boundary: every spoken chunk is a whole sentence
    assert all(c.strip().endswith(".") for c in task.calls), task.calls


def test_empty_text_is_an_error_not_a_crash():
    _reset()
    task = _task()
    out = task.run("")
    assert out["output"] is None
    assert out["metrics"]["status"] == "error"
    assert "empty text" in out["metrics"]["error"]


def test_markdown_only_text_is_reported_as_unspeakable():
    _reset()
    task = _task()
    out = task.run("***")
    assert out["metrics"]["status"] == "error"
    assert "no speakable content" in out["metrics"]["error"]
    assert WRITES == [], "nothing should have been written"


def test_synthesis_returning_nothing_is_an_error():
    _reset()
    task = _task()
    task._synth = lambda text: np.zeros(0, np.float32)  # type: ignore
    out = task.run("Something went wrong.")
    assert out["metrics"]["status"] == "error"
    assert "no audio" in out["metrics"]["error"]


def test_failure_still_logs_a_metrics_row():
    _reset()
    task = _task()
    task.run("")
    rows = task.metrics.recent(1)
    assert rows and rows[0]["subtask"] == "tts"
    assert rows[0]["status"] == "error"


# --------------------------------------------------------- VITS knobs
def test_vits_knobs_are_applied_and_reported():
    task = _task()
    task._model = _FakeModel()
    task._apply_vits_knobs()
    assert task._knobs == {
        "speaking_rate": config.TTS_SPEAKING_RATE,
        "noise_scale": config.TTS_NOISE_SCALE,
        "noise_scale_duration": config.TTS_NOISE_SCALE_DURATION,
    }, task._knobs
    assert task._model.speaking_rate == config.TTS_SPEAKING_RATE


def test_missing_knob_attributes_are_skipped_not_fatal():
    """Guards the attribute-rename risk across transformers versions."""
    class _Bare:
        def __init__(self):
            self.config = types.SimpleNamespace(sampling_rate=SR)

    task = _task()
    task._model = _Bare()
    task._apply_vits_knobs()          # must not raise
    assert task._knobs == {}, task._knobs


# ------------------------------------------------ single-pass (sample) path
def test_single_pass_makes_exactly_one_synth_call():
    """TTS_CHUNKED=0 must reproduce the approved sample's pipeline."""
    _reset()
    task = _task()
    out = task.run({"text": "One line. Two lines. Three lines.",
                    "chunked": False})
    assert out["metrics"]["status"] == "ok", out["metrics"]
    assert len(task.calls) == 1, task.calls
    assert out["output"]["chunks"] == 1
    assert out["output"]["chunked"] is False


def test_single_pass_still_strips_markdown_and_speaks_steps():
    _reset()
    task = _task()
    task.run({"text": "**Do this:**\n1. Clear the `SSO` cache.",
              "chunked": False})
    spoken = task.calls[0]
    assert "*" not in spoken and "`" not in spoken
    assert "First," in spoken, spoken


def test_single_pass_has_no_inserted_pauses():
    """One utterance: output length must match raw synthesis, not exceed it."""
    _reset()
    task = _task()
    task.run({"text": "Alpha. Beta. Gamma.", "chunked": False})
    one_chunk = int(SR * 0.15) * 2 + _tone(60).size
    # trimming may shorten it; nothing may lengthen it
    assert WRITES[0]["samples"] <= one_chunk, WRITES[0]["samples"]


def test_chunked_and_single_pass_differ_in_call_count():
    _reset()
    a = _task(); a.run({"text": "One. Two. Three.", "chunked": True})
    b = _task(); b.run({"text": "One. Two. Three.", "chunked": False})
    assert len(a.calls) == 3 and len(b.calls) == 1


def test_single_pass_truncates_on_a_word_boundary():
    _reset()
    task = _task()
    task.run({"text": "alpha bravo charlie delta echo foxtrot golf hotel",
              "chunked": False, "max_chars": 20})
    spoken = task.calls[0]
    assert len(spoken) <= 20
    assert not spoken.endswith(" ")
    # no partial word survived
    assert all(w in "alpha bravo charlie delta echo foxtrot golf hotel".split()
               for w in spoken.split()), spoken


def test_default_knobs_do_not_alter_synthesis():
    """The approved sample was EQ over DEFAULT-synthesised audio."""
    from src import config as cfg
    assert cfg.TTS_SPEAKING_RATE == 1.0
    assert cfg.TTS_NOISE_SCALE == 0.667
    assert cfg.TTS_NOISE_SCALE_DURATION == 0.8
    fresh = _FakeModel()
    task = _task(); task._model = fresh; task._apply_vits_knobs()
    assert fresh.speaking_rate == 1.0
    assert fresh.noise_scale == 0.667
    assert fresh.noise_scale_duration == 0.8


# ------------------------------------------------- render loop directly
def test_render_chunks_skips_silent_synthesis_and_drops_trailing_gap():
    chunks = [Chunk("a", 200), Chunk("b", 200), Chunk("c", 200)]

    def synth(text):
        return np.zeros(0, np.float32) if text == "b" else _tone(50)

    audio, rendered, truncated = render_chunks(chunks, synth, SR)
    assert rendered == 2, rendered
    assert truncated is False
    # two 50 ms tones + exactly one 200 ms gap between them
    assert audio.size < 2 * _tone(50).size + int(SR * 0.25)


def test_render_chunks_respects_budget_but_always_renders_one():
    chunks = [Chunk("a very long sentence indeed", 100), Chunk("b", 0)]
    audio, rendered, truncated = render_chunks(
        chunks, lambda t: _tone(30), SR, char_budget=5)
    assert rendered == 1 and truncated is True
    assert audio.size > 0


# ------------------------------------------------------- stdlib test runner
def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())