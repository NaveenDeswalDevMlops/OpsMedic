# scripts/smoke_test.py
"""OpsMedic Phase-2B smoke test: run all six sub-tasks end to end.

Journey (mirrors the app): incident text -> [classify] -> [retrieve
similar tickets] -> [resolve via Groq RAG] -> [summarize] -> [tts of the
summary] -> [asr transcribes the generated wav back] (a full speech
round-trip that needs no microphone). Prints each sub-task's
{output, metrics} and the aggregated LLMOps metrics at the end.

Prerequisites (on your machine):
    pip install -r requirements.txt
    python scripts/prepare_dataset.py
    python scripts/build_index.py
    GROQ_API_KEY in .env (resolve step degrades gracefully without it)

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --incident "custom incident text"
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmops.cache import ResponseCache  # noqa: E402
from llmops.metrics import MetricsLogger  # noqa: E402
from models.asr import ASRTask  # noqa: E402
from models.classifier import ClassifierTask  # noqa: E402
from models.resolution import ResolutionTask  # noqa: E402
from models.retrieval import RetrievalTask  # noqa: E402
from models.summarizer import SummarizerTask  # noqa: E402
from models.tts import TTSTask  # noqa: E402
from src import config  # noqa: E402

DEFAULT_INCIDENT = (
    "Dear support team, since this morning several colleagues cannot sign in "
    "to the account management portal. Login attempts time out on multiple "
    "browsers and the mobile app, and password reset emails are not arriving. "
    "This is blocking access to customer accounts across the sales team."
)


def show(step: str, result: dict) -> None:
    print(f"\n=== {step} ===")
    print(json.dumps(result, indent=2, default=str)[:1500])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incident", default=DEFAULT_INCIDENT)
    args = parser.parse_args()

    metrics = MetricsLogger(config.METRICS_DB_PATH)
    cache = ResponseCache(config.CACHE_DB_PATH, config.CACHE_TTL_SECONDS)
    kw = {"metrics": metrics, "cache": cache if config.CACHE_ENABLED else None}

    print(f"{config.PRODUCT_NAME} smoke test — incident:\n{args.incident[:200]}")

    # 1) classify (fine-tuned if artifact exists, else honest base baseline)
    clf = ClassifierTask(variant="auto", **kw)
    r_cls = clf.run(args.incident)
    show(f"1. classify [{clf.variant}]", r_cls)
    category = (r_cls["output"] or {}).get("label")

    # 2) retrieve similar resolved tickets
    r_ret = RetrievalTask(**kw).run(args.incident)
    show("2. retrieve", r_ret)
    similar = r_ret["output"] or []

    # 3) grounded resolution (needs GROQ_API_KEY; logs error row otherwise)
    r_res = ResolutionTask(**kw).run(
        {"incident": args.incident, "similar": similar, "category": category}
    )
    show("3. resolve", r_res)
    resolution_text = (r_res["output"] or {}).get(
        "resolution",
        "Resolution unavailable (no GROQ_API_KEY). Using top similar "
        "ticket's resolution instead: "
        + (similar[0]["resolution"] if similar else "n/a"),
    )

    # 4) summarize the resolution for handover / audio
    r_sum = SummarizerTask(**kw).run(resolution_text)
    show("4. summarize", r_sum)
    summary_text = r_sum["output"] or resolution_text[:300]

    # 5) tts: speak the summary
    r_tts = TTSTask(**kw).run({"text": summary_text, "out_path": "./data/tts_out.wav"})
    show("5. tts", r_tts)

    # 6) asr: transcribe the wav we just generated (speech round-trip)
    if r_tts["output"]:
        r_asr = ASRTask(**kw).run(r_tts["output"]["audio_path"])
        show("6. asr (round-trip of the TTS wav)", r_asr)

    print("\n=== LLMOps metrics: per-subtask summary ===")
    for row in metrics.summary_by_subtask():
        print(json.dumps(row))
    print("\n=== LLMOps metrics: global ===")
    print(json.dumps(metrics.summary(), indent=2))
    print(f"\nCache stats: {cache.stats()}")
    print("\nSmoke test complete. Screenshot this output for the report.")


if __name__ == "__main__":
    main()
