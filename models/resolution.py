# models/resolution.py
"""Sub-task 3 (NLP / GenAI): grounded resolution generation.

Takes the new incident plus the retrieved similar tickets (with their
real resolutions) and the linked category SOP, and asks a Groq-hosted
Llama model for numbered resolution steps grounded ONLY on that
evidence - classic RAG synthesis.

Model: llama-3.1-8b-instant on Groq free tier. Chosen for: fluent
instruction-following at near-zero cost and very low latency; the paid
upgrade path (larger Llama / GPT-4o) is a one-line config change.
"""
from __future__ import annotations

import os
from typing import Any

from models.base import BaseSubTask
from scripts.prepare_dataset import queue_slug
from src import config

PROMPT_VERSION = "resolve-v1"  # LLMOps: prompt versioning (base family)
MAX_SOP_CHARS = 1600
MAX_SIMILAR = 3

# Every style shares this grounding contract; only the output FORMAT
# changes. Keeping the anti-hallucination clause style-independent means
# the "Writing Styles" control can never loosen the guardrail.
GROUNDING_RULE = (
    "You are OpsMedic, an L1 incident-resolution copilot. "
    "Ground every recommendation ONLY on the SIMILAR RESOLVED TICKETS "
    "and the STANDARD OPERATING PROCEDURE provided. "
)
_INSUFFICIENT = (
    "If the evidence is insufficient, say so explicitly and recommend "
    "escalation instead of inventing steps."
)

SYSTEM_PROMPT = GROUNDING_RULE + (
    "Reply with: (1) a one-line diagnosis; (2) numbered resolution steps; "
    "(3) an escalation condition. Cite ticket ids like [OPM-00042] next to "
    "steps taken from them. " + _INSUFFICIENT
)

DEFAULT_STYLE = "stepwise"

#: Writing Styles offered by the composer. Each entry is a distinct,
#: versioned prompt — the dashboard groups metrics by `prompt_version`,
#: so switching style is a first-class LLMOps experiment, not cosmetics.
STYLE_PROMPTS: dict[str, dict[str, str]] = {
    "stepwise": {
        "version": "resolve-v1-stepwise",
        "system": SYSTEM_PROMPT,
    },
    "concise": {
        "version": "resolve-v1-concise",
        "system": GROUNDING_RULE + (
            "Answer in at most four short bullets and no preamble: "
            "diagnosis, the two highest-value fix steps, escalation "
            "trigger. Cite ticket ids like [OPM-00042]. " + _INSUFFICIENT
        ),
    },
    "customer": {
        "version": "resolve-v1-customer",
        "system": GROUNDING_RULE + (
            "Write a short, polite reply addressed to the person who "
            "raised the ticket. Plain language, no internal jargon, no "
            "server or tool names. State what you will do and what you "
            "need from them. End with an internal 'Evidence:' line "
            "listing the ticket ids used. " + _INSUFFICIENT
        ),
    },
    "handover": {
        "version": "resolve-v1-handover",
        "system": GROUNDING_RULE + (
            "Write a shift-handover note under exactly these headings: "
            "IMPACT, DIAGNOSIS, ACTIONS TAKEN, NEXT STEPS FOR ON-CALL, "
            "ESCALATE IF. One or two lines per heading, telegraphic "
            "style. Cite ticket ids like [OPM-00042]. " + _INSUFFICIENT
        ),
    },
}


def resolve_style(style: str | None) -> str:
    """Return a known style key, falling back to the default.

    Pure and total: an unknown or empty style never raises, so a stale
    session value or an old cached payload cannot break a live demo.
    """
    key = (style or "").strip().lower()
    return key if key in STYLE_PROMPTS else DEFAULT_STYLE


def system_prompt_for(style: str | None) -> str:
    """System prompt text for a writing style."""
    return STYLE_PROMPTS[resolve_style(style)]["system"]


def prompt_version_for(style: str | None) -> str:
    """LLMOps prompt version string for a writing style."""
    return STYLE_PROMPTS[resolve_style(style)]["version"]


def load_sop(category: str | None) -> str:
    """Return the linked SOP text for a queue/category ('' if none)."""
    if not category:
        return ""
    path = os.path.join(config.SOPS_DIR, queue_slug(category) + ".md")
    if not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()[:MAX_SOP_CHARS]


def build_resolution_prompt(
    incident: str,
    similar: list[dict[str, Any]],
    category: str | None,
    style: str | None = DEFAULT_STYLE,
) -> list[dict[str, str]]:
    """Pure prompt construction (unit-testable offline).

    Returns OpenAI-style chat messages for the Groq API. `style` selects
    one of STYLE_PROMPTS (the composer's "Writing Styles" control); the
    evidence block is identical across styles so only the output format
    varies.
    """
    blocks: list[str] = [f"NEW INCIDENT:\n{incident.strip()}"]
    if category:
        blocks.append(f"PREDICTED CATEGORY: {category}")

    if similar:
        lines = []
        for t in similar[:MAX_SIMILAR]:
            lines.append(
                f"[{t['ticket_id']}] (similarity {t.get('score', '?')}) "
                f"{t['title']}\n"
                f"  Problem: {t['description']}\n"
                f"  Resolution applied: {t['resolution']}"
            )
        blocks.append("SIMILAR RESOLVED TICKETS:\n" + "\n\n".join(lines))
    else:
        blocks.append("SIMILAR RESOLVED TICKETS: none above threshold.")

    sop = load_sop(category)
    if sop:
        blocks.append(f"STANDARD OPERATING PROCEDURE ({category}):\n{sop}")

    return [
        {"role": "system", "content": system_prompt_for(style)},
        {"role": "user", "content": "\n\n".join(blocks)},
    ]


class ResolutionTask(BaseSubTask):
    name = "resolve"
    category = "NLP"
    cacheable = True  # identical incident+evidence -> reuse, saves quota

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model_name = config.RESOLUTION_MODEL

    def _run(self, payload: Any) -> dict[str, Any]:
        """payload: {"incident": str, "similar": [...], "category": str|None}

        Returns {"resolution": str, "prompt_version": str, "grounded_on":
        [ticket_ids], "sop": category-slug or None}.
        """
        import requests  # lazy (light, but keeps pattern uniform)

        if not isinstance(payload, dict):
            raise ValueError('payload must be {"incident": ..., "similar": ...}')
        incident = str(payload.get("incident", "")).strip()
        if not incident:
            raise ValueError("payload.incident is required")
        similar = payload.get("similar") or []
        category = payload.get("category")
        style = resolve_style(payload.get("style"))

        if not config.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at "
                "https://console.groq.com and put it in .env"
            )

        messages = build_resolution_prompt(incident, similar, category, style)
        resp = requests.post(
            f"{config.GROQ_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {config.GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 700,
            },
            timeout=60,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Groq API {resp.status_code}: {resp.text[:300]}")
        data = resp.json()

        usage = data.get("usage", {})
        self.report_usage(
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
        )
        return {
            "resolution": data["choices"][0]["message"]["content"],
            "prompt_version": prompt_version_for(style),
            "style": style,
            "grounded_on": [t["ticket_id"] for t in similar[:MAX_SIMILAR]],
            "sop": queue_slug(category) if category else None,
        }

    def stream(self, payload: dict[str, Any]):
        """Yield resolution text token-by-token for the chat UI.

        Generator: yields incremental text chunks, then finalizes by
        writing ONE metrics row (with exact token usage from the stream's
        final chunk) and caching the full text. Falls back to the most
        similar ticket's real resolution if no Groq key is set (yields
        that text and logs an error row), so the demo never dead-ends.
        """
        import json as _json
        import time as _time

        import requests

        if not isinstance(payload, dict):
            raise ValueError('payload must be {"incident": ..., "similar": ...}')
        incident = str(payload.get("incident", "")).strip()
        similar = payload.get("similar") or []
        category = payload.get("category")
        style = resolve_style(payload.get("style"))

        # cache hit -> yield whole thing, log a cache-hit row
        if self.cache is not None and self.cacheable:
            hit = self.cache.get(self.name, self.model_name, payload)
            if hit is not None:
                with self.metrics.track(self.name, self.model_name) as rec:
                    rec.mark_cache_hit()
                    rec.estimate_tokens(str(payload), str(hit))
                    rec.add_extra(prompt_version=prompt_version_for(style),
                                  style=style)
                yield hit.get("resolution", "")
                return

        if not config.GROQ_API_KEY:
            fallback = (
                "**Groq key missing — showing the most similar past "
                "ticket's real resolution:**\n\n"
                + (similar[0]["resolution"] if similar else "No resolution available.")
            )
            with self.metrics.track(self.name, self.model_name) as rec:
                rec.estimate_tokens(str(payload), fallback)
                rec.add_extra(error="GROQ_API_KEY not set (streamed fallback)",
                              prompt_version=prompt_version_for(style),
                              style=style)
            yield fallback
            return

        messages = build_resolution_prompt(incident, similar, category, style)
        start = _time.perf_counter()
        collected: list[str] = []
        tokens_in = tokens_out = 0
        ttft_ms: float | None = None  # time to first content token
        status, error_text = "ok", None
        try:
            with requests.post(
                f"{config.GROQ_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {config.GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": 0.2,
                    "max_tokens": 700,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
                timeout=60,
                stream=True,
            ) as resp:
                if resp.status_code != 200:
                    raise RuntimeError(f"Groq API {resp.status_code}: {resp.text[:300]}")
                for line in resp.iter_lines():
                    if not line:
                        continue
                    raw = line.decode("utf-8").removeprefix("data: ").strip()
                    if raw == "[DONE]":
                        break
                    try:
                        chunk = _json.loads(raw)
                    except _json.JSONDecodeError:
                        continue
                    if chunk.get("usage"):
                        tokens_in = chunk["usage"].get("prompt_tokens", 0)
                        tokens_out = chunk["usage"].get("completion_tokens", 0)
                    for choice in chunk.get("choices", []):
                        piece = (choice.get("delta") or {}).get("content")
                        if piece:
                            if ttft_ms is None:
                                ttft_ms = (_time.perf_counter() - start) * 1000.0
                            collected.append(piece)
                            yield piece
        except Exception as exc:  # noqa: BLE001
            status, error_text = "error", f"{type(exc).__name__}: {exc}"
            yield f"\n\n_(streaming error: {error_text})_"
        finally:
            full = "".join(collected)
            latency_ms = (_time.perf_counter() - start) * 1000.0
            # write exactly one metrics row for the whole streamed call
            with self.metrics.track(self.name, self.model_name) as trk:
                # The generator is already drained, so track() would time
                # an empty block and record ~0 ms. Inject the real
                # elapsed time measured across the whole stream.
                trk.set_latency_ms(latency_ms)
                if tokens_in or tokens_out:
                    trk.set_tokens(tokens_in=tokens_in, tokens_out=tokens_out)
                else:
                    trk.estimate_tokens(str(payload), full)
                tokens_total = tokens_out or trk.tokens_out
                grounded_ids = [t["ticket_id"] for t in similar[:MAX_SIMILAR]]
                cited = [tid for tid in grounded_ids if str(tid) in full]
                trk.add_extra(
                    streamed=True,
                    prompt_version=prompt_version_for(style),
                    style=style,
                    ttft_ms=round(ttft_ms, 1) if ttft_ms else None,
                    tokens_per_sec=(
                        round(tokens_total / (latency_ms / 1000.0), 2)
                        if latency_ms > 0 and tokens_total else None
                    ),
                    evidence_count=len(grounded_ids),
                    citations_found=len(cited),
                    grounded=bool(cited),
                )
                if error_text:
                    # status must flip too, or the error-rate metric lies.
                    trk.mark_error(error_text)
            if status == "ok" and full and self.cache is not None and self.cacheable:
                self.cache.set(
                    self.name,
                    self.model_name,
                    payload,
                    {
                        "resolution": full,
                        "prompt_version": prompt_version_for(style),
                        "style": style,
                        "grounded_on": [t["ticket_id"] for t in similar[:MAX_SIMILAR]],
                        "sop": queue_slug(category) if category else None,
                    },
                )
