# models/base.py
"""Uniform sub-task interface for OpsMedic.

Every sub-task (retrieval, resolution, summarizer, classifier, ASR, TTS)
subclasses BaseSubTask and implements only `_run(payload)`. The base
class provides, for free:
  - metrics logging (latency, tokens, cost, errors) via llmops.metrics
  - optional response caching via llmops.cache
  - the assignment-mandated return schema: {"output": ..., "metrics": ...}

Concrete wrappers report exact token usage (when the backend provides
it, e.g. Groq) by calling `self.report_usage(...)` inside `_run`;
otherwise tokens are estimated from the input/output text lengths.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from llmops.cache import ResponseCache
from llmops.metrics import MetricsLogger, approx_tokens


class BaseSubTask(ABC):
    #: unique short id, e.g. "retrieve", "resolve", "summarize",
    #: "classify", "asr", "tts"  (set by subclasses)
    name: str = "base"
    #: assignment category: "NLP" or "Speech"
    category: str = "NLP"
    #: model identifier shown on the dashboard, e.g. "openai/whisper-tiny"
    model_name: str = "unknown"
    #: whether identical inputs may be served from cache (ASR/TTS on raw
    #: audio bytes default to False; text tasks default to True)
    cacheable: bool = True

    def __init__(
        self,
        metrics: MetricsLogger | None = None,
        cache: ResponseCache | None = None,
    ) -> None:
        self.metrics = metrics or MetricsLogger()
        self.cache = cache
        self._usage: dict[str, int] | None = None

    # -- for subclasses -------------------------------------------------
    @abstractmethod
    def _run(self, payload: Any) -> Any:
        """Do the actual work and return the raw output."""

    def report_usage(self, tokens_in: int, tokens_out: int) -> None:
        """Call inside _run when the backend reports exact token usage."""
        self._usage = {"tokens_in": int(tokens_in), "tokens_out": int(tokens_out)}

    # -- public API (assignment schema) ----------------------------------
    def run(self, payload: Any) -> dict[str, Any]:
        """Execute the sub-task. Returns {"output": ..., "metrics": {...}}.

        Never raises for model errors: failures are logged to the metrics
        store and surfaced as {"output": None, "metrics": {..., "status":
        "error", "error": "..."}} so the Streamlit UI can render a banner
        instead of crashing mid-demo.
        """
        self._usage = None

        # 1) cache lookup (text-payload tasks only)
        if self.cache is not None and self.cacheable:
            hit = self.cache.get(self.name, self.model_name, payload)
            if hit is not None:
                with self.metrics.track(self.name, self.model_name) as rec:
                    rec.mark_cache_hit()
                    rec.estimate_tokens(str(payload), str(hit))
                return {
                    "output": hit,
                    "metrics": {
                        "subtask": self.name,
                        "model": self.model_name,
                        "cache_hit": True,
                        "status": "ok",
                    },
                }

        # 2) real execution, metered
        error_text: str | None = None
        output: Any = None
        try:
            with self.metrics.track(self.name, self.model_name) as rec:
                output = self._run(payload)
                if self._usage is not None:
                    rec.set_tokens(**self._usage)
                else:
                    rec.estimate_tokens(str(payload), str(output))
        except Exception as exc:  # noqa: BLE001 - demo must not crash
            error_text = f"{type(exc).__name__}: {exc}"

        # 3) cache fill on success
        if (
            error_text is None
            and self.cache is not None
            and self.cacheable
            and output is not None
        ):
            self.cache.set(self.name, self.model_name, payload, output)

        return {
            "output": output,
            "metrics": {
                "subtask": self.name,
                "model": self.model_name,
                "cache_hit": False,
                "status": "ok" if error_text is None else "error",
                **({"error": error_text} if error_text else {}),
                **(
                    self._usage
                    or {
                        "tokens_in": approx_tokens(str(payload)),
                        "tokens_out": approx_tokens(str(output)) if output else 0,
                        "tokens_estimated": True,
                    }
                ),
            },
        }
