# tests/test_tts_prosody_offline.py
"""Offline tests for the TTS prosody/tone layer.

Runs on synthetic waveforms with numpy only — no torch, no soundfile, no
audio device — so the claims about softening and phrasing are measured
rather than asserted. The softening test checks real spectral energy
above 4 kHz, not just that numbers changed.

Run:  python tests/test_tts_prosody_offline.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.tts_prosody import (  # noqa: E402
    CLAUSE_GAP_MS,
    DEFAULT_PEAK,
    SENTENCE_GAP_MS,
    STEP_GAP_MS,
    Chunk,
    apply_fades,
    cut_rumble,
    emphasize_presence,
    finalize,
    high_band_ratio,
    join_with_pauses,
    normalize_peak,
    presence_ratio,
    silence,
    soften,
    split_for_speech,
    trim_silence,
)

SR = 16000


def _tone(freq: float, ms: int, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(SR * ms / 1000.0)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _speechlike(ms: int = 500) -> np.ndarray:
    """Low fundamental + a hissy top end, like sibilant TTS output."""
    return (_tone(180, ms, 0.45) + _tone(6500, ms, 0.22)).astype(np.float32)


# ------------------------------------------------------------- text side
def test_numbered_steps_become_spoken_ordinals():
    chunks = split_for_speech("1. Clear the SSO cache.\n2. Restart the agent.")
    texts = [c.text for c in chunks]
    assert texts == ["First, Clear the SSO cache.",
                     "Second, Restart the agent."], texts


def test_step_prefix_variants_all_parse():
    for raw in ["3) Reset it.", "(3) Reset it.", "Step 3: Reset it.",
                "step 3 - Reset it."]:
        chunks = split_for_speech(raw)
        assert chunks[0].text.startswith("Third,"), (raw, chunks[0].text)


def test_markdown_is_stripped_not_spoken():
    chunks = split_for_speech("**Diagnosis:** `SSO` cache is _stale_.")
    joined = " ".join(c.text for c in chunks)
    for ch in "*_`#>|":
        assert ch not in joined, f"{ch!r} survived into speech"
    assert "Diagnosis:" in joined and "SSO" in joined


def test_bullets_get_a_longer_pause_before_them():
    chunks = split_for_speech("Summary line.\n- First action\n- Second action")
    # the chunk before a bullet must carry the longer step gap
    assert chunks[0].gap_ms == STEP_GAP_MS, chunks[0].gap_ms
    assert "First action" in chunks[1].text


def test_sentences_split_but_abbreviations_survive():
    chunks = split_for_speech("Check the VPN. Then e.g. restart it. Done.")
    texts = [c.text for c in chunks]
    assert "Check the VPN." in texts
    assert any("e.g. restart it." in t for t in texts), texts
    assert len(texts) == 3, texts


def test_clause_gaps_are_shorter_than_sentence_gaps():
    chunks = split_for_speech("One thing. Another thing.\nNew line here.")
    gaps = [c.gap_ms for c in chunks]
    assert CLAUSE_GAP_MS < SENTENCE_GAP_MS < STEP_GAP_MS
    assert gaps[-1] == 0, "last chunk must not trail a pause"
    assert all(g >= 0 for g in gaps)


def test_empty_and_whitespace_text_is_safe():
    assert split_for_speech("") == []
    assert split_for_speech("   \n\n  ") == []
    assert split_for_speech("***") == []


def test_chunk_equality_helper():
    assert Chunk("a", 10) == Chunk("a", 10)
    assert Chunk("a", 10) != Chunk("a", 20)


# ------------------------------------------------------------ audio side
def _mumbly(ms: int = 500) -> np.ndarray:
    """Boomy + consonant-poor, like the measured mms-tts-eng output."""
    return (_tone(110, ms, 0.60)      # dominant low fundamental
            + _tone(300, ms, 0.30)
            + _tone(2500, ms, 0.05)   # barely-there consonant band
            ).astype(np.float32)


def test_presence_lift_raises_the_intelligibility_band():
    raw = _mumbly()
    before = presence_ratio(raw, SR)
    after = presence_ratio(emphasize_presence(raw, SR, amount=0.35), SR)
    assert after > before * 1.15, (
        f"1.5-4kHz share {before * 100:.2f}% -> {after * 100:.2f}%: no lift")
    assert emphasize_presence(raw, SR, 0.35).size == raw.size


def test_presence_amount_is_monotonic_and_zero_is_a_noop():
    raw = _mumbly()
    ratios = [presence_ratio(emphasize_presence(raw, SR, amount=a), SR)
              for a in (0.0, 0.2, 0.4, 0.8)]
    assert ratios == sorted(ratios), ratios
    assert np.allclose(emphasize_presence(raw, SR, amount=0.0), raw)


def test_cut_rumble_attenuates_sub_speech_boom():
    raw = _mumbly()
    from models.tts_prosody import band_ratio
    before = band_ratio(raw, SR, 0, 150)
    after = band_ratio(cut_rumble(raw, SR, amount=0.8), SR, 0, 150)
    assert after < before * 0.8, (before, after)
    assert cut_rumble(raw, SR).size == raw.size


def test_soften_defaults_to_a_noop_because_the_output_is_already_dull():
    """Guard the correction: measured output had 0.13% above 4 kHz, so
    softening by default would only muddy it further."""
    assert np.allclose(soften(_mumbly()), _mumbly()), "soften must default to no-op"
    # still available, and still effective, when explicitly asked for —
    # measured on a signal that actually HAS high-frequency content
    hissy = _speechlike()          # includes a 6.5 kHz component
    assert high_band_ratio(soften(hissy, 0.6, SR), SR) < \
        high_band_ratio(hissy, SR) * 0.9


def test_clarity_chain_does_not_gut_the_fundamental():
    raw = _mumbly()
    out = emphasize_presence(cut_rumble(raw, SR, amount=0.6), SR, 0.3)
    spec_in = np.abs(np.fft.rfft(raw))
    spec_out = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(raw.size, d=1.0 / SR)
    band = (freqs > 250) & (freqs < 360)      # the 300 Hz component
    kept = spec_out[band].sum() / spec_in[band].sum()
    assert kept > 0.75, f"voice body lost {(1 - kept) * 100:.1f}%"


def test_trim_silence_strips_padding_but_keeps_a_margin():
    body = _tone(200, 200)
    padded = np.concatenate([silence(SR, 400), body, silence(SR, 400)])
    out = trim_silence(padded, SR, keep_ms=40)
    expected = body.size + 2 * int(SR * 40 / 1000.0)
    assert abs(out.size - expected) < SR * 0.02, (out.size, expected)
    assert out.size < padded.size


def test_trim_silence_on_all_silence_returns_empty():
    assert trim_silence(silence(SR, 500), SR).size == 0


def test_normalize_peak_hits_the_target_and_never_boosts_silence():
    hot = _tone(200, 100, amp=0.98)
    out = normalize_peak(hot, DEFAULT_PEAK)
    assert abs(float(np.max(np.abs(out))) - DEFAULT_PEAK) < 1e-3
    quiet = silence(SR, 50)
    assert float(np.max(np.abs(normalize_peak(quiet)))) == 0.0


def test_fades_prevent_clicks_at_the_joins():
    dc = np.full(int(SR * 0.1), 0.8, dtype=np.float32)   # worst case
    faded = apply_fades(dc, SR, ms=8)
    assert abs(float(faded[0])) < 0.05, faded[0]
    assert abs(float(faded[-1])) < 0.05, faded[-1]
    joined = join_with_pauses([(dc, 100), (dc, 0)], SR)
    assert float(np.max(np.abs(np.diff(joined)))) < 0.2, "click at the join"


def test_join_inserts_the_requested_pauses():
    a, b = _tone(200, 100), _tone(300, 100)
    out = join_with_pauses([(a, 260), (b, 0)], SR)
    gap = int(SR * 260 / 1000.0)
    # trimmed segments + one gap; allow slack for edge margins
    assert out.size > a.size + b.size, "pause not inserted"
    assert out.size < a.size + b.size + gap + int(SR * 0.2)


def test_join_of_nothing_is_empty_not_a_crash():
    assert join_with_pauses([], SR).size == 0
    assert join_with_pauses([(silence(SR, 50), 0)], SR).size == 0


def test_finalize_clarifies_and_tames_the_level_together():
    raw = np.clip(_mumbly(300) * 1.9, -1.0, 1.0).astype(np.float32)
    out = finalize(raw, SR)
    assert abs(float(np.max(np.abs(out))) - DEFAULT_PEAK) < 1e-3
    assert presence_ratio(out, SR) > presence_ratio(raw, SR), \
        "finalize must improve intelligibility, not reduce it"


def test_default_peak_is_below_the_raw_mms_level():
    """The uploaded sample peaked at 0.86 full scale, which reads as harsh."""
    assert DEFAULT_PEAK < 0.86


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