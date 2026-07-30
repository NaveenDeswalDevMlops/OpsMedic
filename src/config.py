# src/config.py
"""Central configuration for OpsMedic — GenAI Incident Copilot.

Standalone project for BITS Pilani CCZG506 Assignment II.
All settings are env-driven with sensible free/local defaults.
Rename the product by changing PRODUCT_NAME in .env — nothing else.

python-dotenv is optional: if installed (it is in requirements.txt),
a local .env file is loaded; otherwise plain environment variables
are used, so this module imports cleanly in any environment (CI,
tests, containers) with zero side effects.
"""
from __future__ import annotations

import os

try:  # optional .env support
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv simply not installed
    pass


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


# --- Branding -------------------------------------------------------
PRODUCT_NAME: str = os.getenv("PRODUCT_NAME", "OpsMedic")
PRODUCT_TAGLINE: str = os.getenv(
    "PRODUCT_TAGLINE", "GenAI Incident Copilot — diagnose, resolve, learn"
)
ORG_NAME: str = os.getenv("ORG_NAME", "Genpact")

# --- LLM (Groq free tier; OpenAI-compatible endpoint) ---------------
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
RESOLUTION_MODEL: str = os.getenv("RESOLUTION_MODEL", "llama-3.1-8b-instant")

# --- Sub-task models (all free; local Hugging Face by default) ------
# NLP category
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
SUMMARIZER_MODEL: str = os.getenv("SUMMARIZER_MODEL", "facebook/bart-large-cnn")
# DeBERTa-v3-base (184M, MIT) beats DistilBERT on short jargon-dense text and
# is the one model we fine-tune, so it drives the before/after numbers.
# NOTE: needs `sentencepiece` installed, and is learning-rate sensitive —
# train at ~2e-5, not the 5e-5 that suited DistilBERT.
CLASSIFIER_BASE_MODEL: str = os.getenv(
    "CLASSIFIER_BASE_MODEL", "microsoft/deberta-v3-base"
)
CLASSIFIER_FINETUNED_DIR: str = os.getenv(
    "CLASSIFIER_FINETUNED_DIR", "./finetune/artifacts/deberta-tickets"
)
# Speech Recognition category
# whisper-small (244M) is a large WER win over tiny and needs no transformers
# bump. whisper-large-v3-turbo is better still but wants transformers >= 4.45
# and will thermal-throttle a fanless MacBook Air on a long demo.
ASR_MODEL: str = os.getenv("ASR_MODEL", "openai/whisper-small")
# Default stays mms-tts-eng so a fresh clone runs with no extra system
# packages. Kokoro-82M sounds markedly better (24 kHz, real voice presets)
# but needs `pip install kokoro` AND the espeak-ng binary, so it is opt-in:
#     TTS_MODEL=hexgrad/Kokoro-82M
# models/tts.py picks the backend from this string.
TTS_MODEL: str = os.getenv("TTS_MODEL", "facebook/mms-tts-eng")
# Kokoro voice presets: af_*/am_* American female/male, bf_*/bm_* British.
KOKORO_VOICE: str = os.getenv("KOKORO_VOICE", "bf_emma")
KOKORO_LANG: str = os.getenv("KOKORO_LANG", "b")   # 'a' American, 'b' British
KOKORO_SPEED: float = float(os.getenv("KOKORO_SPEED", "1.0"))

# --- TTS delivery / tone (see models/tts_prosody.py) ---
# Measured on real mms-tts-eng output: 83% of energy below 500 Hz and only
# 2.7% in the 1.5-4 kHz consonant band, i.e. boomy and mushy rather than
# harsh. So the chain LIFTS presence and cuts rumble; softening defaults
# off and is only for a future brighter engine.
TTS_PRESENCE: float = float(os.getenv("TTS_PRESENCE", "0.30"))
TTS_RUMBLE_CUT: float = float(os.getenv("TTS_RUMBLE_CUT", "0.6"))
TTS_SOFTEN: float = float(os.getenv("TTS_SOFTEN", "0.0"))
TTS_PEAK: float = float(os.getenv("TTS_PEAK", "0.72"))
TTS_MAX_CHARS: int = int(os.getenv("TTS_MAX_CHARS", "900"))
# Chunked delivery: synthesize sentence-by-sentence and rejoin with
# deliberate pauses. Fixes the 34% dead air, but prosody resets per chunk,
# so the result differs audibly from a single continuous pass. Set 0 to get
# the single-pass path (raw text -> one synth call -> EQ), which is exactly
# what the approved tone sample used.
TTS_CHUNKED: bool = _get_bool("TTS_CHUNKED", True)

# VITS sampling knobs. Defaults are the model's OWN defaults, i.e. no change
# to synthesis: the approved tone sample was EQ applied on top of
# default-synthesised audio, so deviating here changes the voice itself.
# speaking_rate <1 slows delivery; noise_scale_duration is the stochastic
# duration temperature (rhythm variability). Tune only after A/B-ing.
TTS_SPEAKING_RATE: float = float(os.getenv("TTS_SPEAKING_RATE", "1.0"))
TTS_NOISE_SCALE: float = float(os.getenv("TTS_NOISE_SCALE", "0.667"))
TTS_NOISE_SCALE_DURATION: float = float(
    os.getenv("TTS_NOISE_SCALE_DURATION", "0.8"))

# --- Data: Customer IT Support ticket dataset (with resolutions) ----
# Citation: T. Bueck, "Customer Support Tickets", Hugging Face.
#   https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets
#   DOI: 10.57967/hf/6184 · License: CC-BY-NC-4.0 (academic use OK)
#   Also on Kaggle as "Customer IT Support - Ticket Dataset"
#   (tobiasbueck/multilingual-customer-support-tickets).
# 61.8k rows; columns: subject, body, answer (agent resolution reply),
# type, queue, priority, language (en/de), tag_1..tag_8.
HF_DATASET: str = os.getenv("HF_DATASET", "Tobi-Bueck/customer-support-tickets")
DATASET_LANGUAGE: str = os.getenv("DATASET_LANGUAGE", "en")
TOP_QUEUES: int = int(os.getenv("TOP_QUEUES", "10"))
RAW_DATA_DIR: str = os.getenv("RAW_DATA_DIR", "./data/raw")
TICKETS_CSV: str = os.getenv("TICKETS_CSV", "./data/tickets.csv")
SOPS_DIR: str = os.getenv("SOPS_DIR", "./data/sops")
FINETUNE_TRAIN_CSV: str = os.getenv("FINETUNE_TRAIN_CSV", "./data/finetune_train.csv")
FINETUNE_TEST_CSV: str = os.getenv("FINETUNE_TEST_CSV", "./data/finetune_test.csv")
# Cap on rows embedded into the retrieval index (0 = all English rows).
KB_ROWS: int = int(os.getenv("KB_ROWS", "0"))

INDEX_DIR: str = os.getenv("INDEX_DIR", "./data/index")
TOP_K: int = int(os.getenv("TOP_K", "3"))
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))

# --- LLMOps: metrics + cache -----------------------------------------
METRICS_DB_PATH: str = os.getenv("METRICS_DB_PATH", "./data/llmops_metrics.db")
CACHE_DB_PATH: str = os.getenv("CACHE_DB_PATH", "./data/llmops_cache.db")
CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
CACHE_ENABLED: bool = _get_bool("CACHE_ENABLED", True)

# Approximate Groq pricing, USD per 1M tokens (editable estimates;
# local Hugging Face models cost $0). Feeds the cost-per-request metric.
PRICE_PER_MTOK: dict[str, dict[str, float]] = {
    "llama-3.3-70b-versatile": {"in": 0.59, "out": 0.79},
    "llama-3.1-8b-instant": {"in": 0.05, "out": 0.08},
    "gemma2-9b-it": {"in": 0.20, "out": 0.20},
}

# --- Fine-tuning -----------------------------------------------------
# Public dataset id is pinned inside finetune/data.py with its citation.
FINETUNE_MAX_ROWS: int = int(os.getenv("FINETUNE_MAX_ROWS", "2000"))
FINETUNE_EPOCHS: int = int(os.getenv("FINETUNE_EPOCHS", "2"))
FINETUNE_SEED: int = int(os.getenv("FINETUNE_SEED", "42"))


# --- Compute device -------------------------------------------------
def resolve_device() -> str:
    """Return 'cuda', 'mps' or 'cpu' for model placement.

    A function rather than a module constant so importing config stays
    cheap — torch is only imported when a model is actually loaded.

    Set DEVICE=cpu in .env to force a fallback; on Apple Silicon also set
    PYTORCH_ENABLE_MPS_FALLBACK=1, because a few ops are still
    unimplemented on the MPS backend and hard-error without it.
    """
    want = os.getenv("DEVICE", "auto").strip().lower()
    if want and want != "auto":
        return want
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001 - never let device probing break startup
        pass
    return "cpu"


def use_fp16() -> bool:
    """fp16 is safe on CUDA; on MPS it still produces NaNs in some models."""
    return resolve_device() == "cuda"