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
SUMMARIZER_MODEL: str = os.getenv("SUMMARIZER_MODEL", "sshleifer/distilbart-cnn-12-6")
CLASSIFIER_BASE_MODEL: str = os.getenv(
    "CLASSIFIER_BASE_MODEL", "distilbert-base-uncased"
)
CLASSIFIER_FINETUNED_DIR: str = os.getenv(
    "CLASSIFIER_FINETUNED_DIR", "./finetune/artifacts/distilbert-tickets"
)
# Speech Recognition category
ASR_MODEL: str = os.getenv("ASR_MODEL", "openai/whisper-tiny")
TTS_MODEL: str = os.getenv("TTS_MODEL", "facebook/mms-tts-eng")

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
