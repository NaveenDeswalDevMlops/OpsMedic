# models/tts_prosody.py
"""Prosody + tone shaping for TTS output — engine-agnostic.

`facebook/mms-tts-eng` (and every other small VITS) has no prosody or
style conditioning: its learned prior is flat scripted narration, so no
parameter makes it "expressive". Natural-sounding delivery therefore has
to be built around the model:

  text side   split into speakable chunks so the engine phrases each
              sentence, instead of one long breathless pass whose only
              rhythm comes from whatever punctuation survived markdown
  audio side  trim the dead air the model pads each chunk with, rejoin
              with *deliberate* pauses (short within a step, longer
              between steps), then soften: gentle high-frequency
              roll-off and a lower peak target, because small VITS
              models are brittle and sibilant near full scale

Everything here is pure numpy on float32 mono in [-1, 1], so it works
with Piper, edge-tts, SpeechT5 or MMS alike, and is unit-testable with
no torch, no soundfile and no audio device.
"""
from __future__ import annotations

import re

import numpy as np

# ---------------------------------------------------------------- defaults
DEFAULT_PEAK = 0.72          # 0.86 (raw MMS) sounds hot and harsh
SENTENCE_GAP_MS = 260        # between sentences
STEP_GAP_MS = 420            # before a new numbered step
CLAUSE_GAP_MS = 140          # after a comma-ish break
EDGE_SILENCE_MS = 40         # dead air to leave at a chunk edge
FADE_MS = 8                  # anti-click fade at every join
SILENCE_FLOOR = 0.012        # |amplitude| below this counts as silence

_ORDINALS = {
    1: "First", 2: "Second", 3: "Third", 4: "Fourth", 5: "Fifth",
    6: "Sixth", 7: "Seventh", 8: "Eighth", 9: "Ninth", 10: "Tenth",
}

# "1. ", "1) ", "(1) ", "Step 1:" at the start of a line
_NUMBERED = re.compile(r"^\s*\(?(\d{1,2})[.):]\s+|^\s*step\s+(\d{1,2})\s*[:.\-]?\s+",
                       re.IGNORECASE)
_BULLET = re.compile(r"^\s*[-*•·]\s+")
_MD_NOISE = re.compile(r"[*_`#>|]+")
_MULTISPACE = re.compile(r"[ \t]+")
# split after . ! ? or ; when followed by whitespace, but not inside a
# common abbreviation or a decimal number
_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z(\"'\d])")
_ABBREV = re.compile(r"\b(?:e\.g|i\.e|etc|vs|no|fig|approx|mr|mrs|dr)\.$",
                     re.IGNORECASE)


class Chunk:
    """One speakable unit plus the pause that should follow it."""

    __slots__ = ("text", "gap_ms")

    def __init__(self, text: str, gap_ms: int) -> None:
        self.text = text
        self.gap_ms = gap_ms

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Chunk({self.text!r}, gap_ms={self.gap_ms})"

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, Chunk) and other.text == self.text
                and other.gap_ms == self.gap_ms)


# ------------------------------------------------------------- text side
def _clean_line(line: str) -> str:
    line = _MD_NOISE.sub(" ", line)
    return _MULTISPACE.sub(" ", line).strip()


def split_for_speech(text: str) -> list[Chunk]:
    """Turn resolution/summary text into speakable chunks with pauses.

    A numbered step becomes "First, <text>" and earns a longer pause
    before it, which is what makes a list *sound* like a list instead of
    a run-on sentence. Markdown is stripped rather than spoken.
    """
    if not text or not text.strip():
        return []

    chunks: list[Chunk] = []
    for raw_line in text.splitlines():
        line = _clean_line(raw_line)
        if not line:
            continue

        gap_before_this_line = SENTENCE_GAP_MS
        m = _NUMBERED.match(line)
        if m:
            n = int(m.group(1) or m.group(2))
            line = _NUMBERED.sub("", line, count=1).strip()
            word = _ORDINALS.get(n, f"Step {n}")
            line = f"{word}, {line}" if line else word
            gap_before_this_line = STEP_GAP_MS
        elif _BULLET.match(line):
            line = _BULLET.sub("", line).strip()
            gap_before_this_line = STEP_GAP_MS

        if not line:
            continue
        # a longer pause belongs BEFORE a step: lengthen the previous gap
        if chunks and gap_before_this_line == STEP_GAP_MS:
            chunks[-1].gap_ms = STEP_GAP_MS

        parts = [p.strip() for p in _SENT_SPLIT.split(line) if p.strip()]
        merged: list[str] = []
        for part in parts:
            # don't split on "e.g." and friends
            if merged and _ABBREV.search(merged[-1]):
                merged[-1] = f"{merged[-1]} {part}"
            else:
                merged.append(part)
        for i, sentence in enumerate(merged):
            last = i == len(merged) - 1
            chunks.append(Chunk(
                sentence,
                SENTENCE_GAP_MS if last else CLAUSE_GAP_MS,
            ))

    if chunks:
        chunks[-1].gap_ms = 0        # no trailing pause on the last chunk
    return chunks


def flatten_for_speech(text: str, char_budget: int = 900) -> str:
    """One continuous utterance: markdown stripped, steps spoken, no splits.

    Used when TTS_CHUNKED is off. Prosody then comes entirely from the
    model's own contour over the whole passage, which is what the approved
    tone sample sounded like — chunking resets that contour per sentence.
    Markdown is still removed, because reading asterisks aloud is a text
    defect, not a tone choice.
    """
    parts = [c.text for c in split_for_speech(text)]
    if not parts:
        return ""
    joined = " ".join(parts)
    if len(joined) <= char_budget:
        return joined
    # cut on a word boundary, never mid-word
    cut = joined[:char_budget].rsplit(" ", 1)[0]
    return cut or joined[:char_budget]


# ------------------------------------------------------------ audio side
def _as_float_mono(audio: np.ndarray) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim > 1:
        a = a.mean(axis=tuple(range(1, a.ndim)))
    return a


def silence(sr: int, ms: int) -> np.ndarray:
    """`ms` of digital silence at sample rate `sr`."""
    return np.zeros(max(0, int(sr * ms / 1000.0)), dtype=np.float32)


def trim_silence(audio: np.ndarray, sr: int,
                 floor: float = SILENCE_FLOOR,
                 keep_ms: int = EDGE_SILENCE_MS) -> np.ndarray:
    """Strip dead air from both edges, keeping a short natural margin.

    Small VITS models pad generous silence onto every utterance; left in
    place while chunking, those pads stack into the long dead gaps that
    make the delivery sound halting.
    """
    a = _as_float_mono(audio)
    if a.size == 0:
        return a
    loud = np.flatnonzero(np.abs(a) >= floor)
    if loud.size == 0:
        return np.zeros(0, dtype=np.float32)
    keep = int(sr * keep_ms / 1000.0)
    start = max(0, int(loud[0]) - keep)
    end = min(a.size, int(loud[-1]) + 1 + keep)
    return a[start:end]


def apply_fades(audio: np.ndarray, sr: int, ms: int = FADE_MS) -> np.ndarray:
    """Linear fade in/out so concatenation cannot click."""
    a = _as_float_mono(audio).copy()
    n = int(sr * ms / 1000.0)
    if n <= 0 or a.size == 0:
        return a
    n = min(n, a.size // 2) or 1
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    a[:n] *= ramp
    a[-n:] *= ramp[::-1]
    return a


def _sinc_lowpass(cutoff_hz: float, sr: int, taps: int = 257) -> np.ndarray:
    """Hann-windowed sinc low-pass kernel (pure numpy, no scipy)."""
    if taps % 2 == 0:
        taps += 1
    fc = float(np.clip(cutoff_hz / sr, 1e-4, 0.499))
    n = np.arange(taps) - (taps - 1) / 2.0
    h = np.sinc(2 * fc * n) * np.hanning(taps)
    return (h / h.sum()).astype(np.float32)


def _lowpass(audio: np.ndarray, cutoff_hz: float, sr: int) -> np.ndarray:
    a = _as_float_mono(audio)
    h = _sinc_lowpass(cutoff_hz, sr)
    if a.size < h.size:
        return a
    pad = h.size // 2
    padded = np.concatenate([np.full(pad, a[0], np.float32), a,
                             np.full(pad, a[-1], np.float32)])
    return np.convolve(padded, h, mode="valid")[:a.size].astype(np.float32)


def soften(audio: np.ndarray, amount: float = 0.0, sr: int = 16000,
           cutoff_hz: float = 4500.0) -> np.ndarray:
    """High-frequency roll-off, for engines whose output is harsh.

    NOTE the default is 0.0 (no-op). Measured on real mms-tts-eng output,
    only 0.13% of energy sits above 4 kHz — that output is already dull,
    and rolling it off further only muddies it. Reach for this ONLY if a
    replacement engine (22-24 kHz Piper / edge-tts voices) turns out
    sibilant. Length is preserved exactly.
    """
    a = _as_float_mono(audio)
    amount = float(np.clip(amount, 0.0, 1.0))
    if a.size < 8 or amount == 0.0:
        return a
    return ((1.0 - amount) * a + amount * _lowpass(a, cutoff_hz, sr)
            ).astype(np.float32)


def emphasize_presence(audio: np.ndarray, sr: int, amount: float = 0.30,
                       lo_hz: float = 1500.0,
                       hi_hz: float = 4000.0) -> np.ndarray:
    """Lift the consonant band so words articulate instead of mumbling.

    Intelligibility lives in roughly 1.5-4 kHz (fricatives, plosives,
    formant transitions). The uploaded mms-tts-eng sample carried only
    2.69% of its energy there against 83% below 500 Hz, which is what
    makes it sound like it is mispronouncing words. Adds a band-passed
    copy back at `amount`; 0.25-0.40 is the useful range.
    """
    a = _as_float_mono(audio)
    amount = float(max(0.0, amount))
    if a.size < 8 or amount == 0.0:
        return a
    band = _lowpass(a, hi_hz, sr) - _lowpass(a, lo_hz, sr)
    return (a + amount * band).astype(np.float32)


def cut_rumble(audio: np.ndarray, sr: int, below_hz: float = 110.0,
               amount: float = 0.6) -> np.ndarray:
    """Attenuate sub-speech boom that masks everything above it."""
    a = _as_float_mono(audio)
    amount = float(np.clip(amount, 0.0, 1.0))
    if a.size < 8 or amount == 0.0:
        return a
    return (a - amount * _lowpass(a, below_hz, sr)).astype(np.float32)


def normalize_peak(audio: np.ndarray, target: float = DEFAULT_PEAK) -> np.ndarray:
    """Scale so the loudest sample sits at `target` (never amplifies noise)."""
    a = _as_float_mono(audio)
    peak = float(np.max(np.abs(a))) if a.size else 0.0
    if peak <= 1e-6:
        return a
    return (a * (float(target) / peak)).astype(np.float32)


def join_with_pauses(pieces: list[tuple[np.ndarray, int]], sr: int,
                     fade_ms: int = FADE_MS) -> np.ndarray:
    """Concatenate (audio, gap_ms_after) pairs, trimming and fading each."""
    out: list[np.ndarray] = []
    for audio, gap_ms in pieces:
        seg = apply_fades(trim_silence(audio, sr), sr, fade_ms)
        if seg.size:
            out.append(seg)
        if gap_ms > 0:
            out.append(silence(sr, gap_ms))
    if not out:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(out).astype(np.float32)


def finalize(audio: np.ndarray, sr: int, presence: float = 0.30,
             rumble: float = 0.6, soften_amount: float = 0.0,
             peak: float = DEFAULT_PEAK) -> np.ndarray:
    """The last step before writing the WAV.

    Order matters: cut the boom first so the presence lift is not just
    amplifying mud, then normalise last so the level is predictable
    whatever the EQ did. `soften_amount` defaults to 0.0 — see soften().
    """
    a = cut_rumble(audio, sr, amount=rumble)
    a = emphasize_presence(a, sr, amount=presence)
    if soften_amount > 0:
        a = soften(a, soften_amount, sr)
    return normalize_peak(a, peak)


def render_chunks(chunks: list[Chunk], synth, sr: int,
                  char_budget: int = 900,
                  fade_ms: int = FADE_MS) -> tuple[np.ndarray, int, bool]:
    """Synthesize each chunk with `synth`, then join with its pauses.

    `synth(text) -> np.ndarray` is injected rather than imported, which
    keeps this loop engine-agnostic (MMS / Piper / edge-tts / SpeechT5)
    and lets the tests exercise the whole assembly path with a fake
    synth — no torch, no model download.

    `char_budget` caps total spoken characters so a long resolution can't
    turn into a two-minute clip mid-demo; the cut lands on a chunk
    boundary rather than mid-word as a raw slice would.

    Returns (audio, chunks_rendered, truncated).
    """
    pieces: list[tuple[np.ndarray, int]] = []
    used = 0
    truncated = False
    for chunk in chunks:
        if pieces and used + len(chunk.text) > char_budget:
            truncated = True
            break
        audio = np.asarray(synth(chunk.text), dtype=np.float32)
        used += len(chunk.text)
        if audio.size:
            pieces.append((audio, chunk.gap_ms))
    if pieces:
        # never trail a pause on the final piece
        pieces[-1] = (pieces[-1][0], 0)
    return join_with_pauses(pieces, sr, fade_ms), len(pieces), truncated


# ------------------------------------------------------ measurement aids
def band_ratio(audio: np.ndarray, sr: int, lo_hz: float,
               hi_hz: float | None = None) -> float:
    """Fraction of spectral energy in [lo_hz, hi_hz).

    Lets the tests assert on real spectral change rather than trusting
    that a filter did what its name claims.
    """
    a = _as_float_mono(audio)
    if a.size < 8:
        return 0.0
    spectrum = np.abs(np.fft.rfft(a)) ** 2
    freqs = np.fft.rfftfreq(a.size, d=1.0 / sr)
    total = float(spectrum.sum())
    if total <= 0:
        return 0.0
    mask = freqs >= lo_hz if hi_hz is None else (freqs >= lo_hz) & (freqs < hi_hz)
    return float(spectrum[mask].sum() / total)


def high_band_ratio(audio: np.ndarray, sr: int,
                    split_hz: float = 4000.0) -> float:
    """Fraction of spectral energy above `split_hz`."""
    return band_ratio(audio, sr, split_hz, None)


def presence_ratio(audio: np.ndarray, sr: int) -> float:
    """Fraction of energy in the 1.5-4 kHz intelligibility band."""
    return band_ratio(audio, sr, 1500.0, 4000.0)