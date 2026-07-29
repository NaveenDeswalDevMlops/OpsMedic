# tests/test_composer_offline.py
"""Offline tests for the composer and the Writing Styles prompt registry.

ui/composer.py touches Streamlit only inside render(); every decision
helper is pure, so this suite installs a bare `streamlit` stub when the
real package is absent (CI, container) and then exercises the helpers
with plain dicts. No model downloads, no network, no browser.

Run:  python tests/test_composer_offline.py     (or: pytest tests -v)
"""
from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# --- stub Streamlit if it isn't installed (import-time only need) ------
if "streamlit" not in sys.modules:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        sys.modules["streamlit"] = types.ModuleType("streamlit")

from models.resolution import (  # noqa: E402
    DEFAULT_STYLE,
    GROUNDING_RULE,
    STYLE_PROMPTS,
    SYSTEM_PROMPT,
    build_resolution_prompt,
    prompt_version_for,
    resolve_style,
    system_prompt_for,
)
from ui import composer  # noqa: E402

SIMILAR = [
    {"ticket_id": "OPM-00042", "title": "Portal login fails",
     "description": "users cannot login", "resolution": "reset SSO cache",
     "score": 0.88},
]


# ------------------------------------------------- style registry wiring
def test_labels_and_prompts_cover_the_same_styles():
    """A label without a prompt (or vice versa) would break the popover."""
    assert set(composer.STYLE_LABELS) == set(STYLE_PROMPTS)
    assert set(composer.STYLE_ORDER) == set(STYLE_PROMPTS)
    assert set(composer.STYLE_HELP) == set(STYLE_PROMPTS)
    assert len(composer.STYLE_ORDER) == len(set(composer.STYLE_ORDER))


def test_default_style_is_first_in_the_popover():
    assert composer.STYLE_ORDER[0] == DEFAULT_STYLE
    assert DEFAULT_STYLE in STYLE_PROMPTS


def test_every_style_keeps_the_grounding_rule():
    """The style control must never be able to relax the guardrail."""
    for key, spec in STYLE_PROMPTS.items():
        assert spec["system"].startswith(GROUNDING_RULE), key
        assert "escalation" in spec["system"].lower(), key


def test_prompt_versions_are_unique_per_style():
    versions = [prompt_version_for(k) for k in STYLE_PROMPTS]
    assert len(versions) == len(set(versions))
    assert all(v.startswith("resolve-v1") for v in versions)


def test_resolve_style_is_total():
    assert resolve_style("concise") == "concise"
    assert resolve_style("CONCISE") == "concise"
    assert resolve_style("  handover ") == "handover"
    assert resolve_style(None) == DEFAULT_STYLE
    assert resolve_style("") == DEFAULT_STYLE
    assert resolve_style("no-such-style") == DEFAULT_STYLE


def test_style_label_falls_back_safely():
    assert composer.style_label("customer") == "Customer-facing"
    assert composer.style_label("bogus") == composer.STYLE_LABELS[DEFAULT_STYLE]


# ------------------------------------------------- prompt construction
def test_default_style_prompt_is_unchanged():
    """Backward compatibility: the 3-arg call still yields SYSTEM_PROMPT."""
    msgs = build_resolution_prompt("login portal down", SIMILAR, "IT Support")
    assert msgs[0]["content"] == SYSTEM_PROMPT
    assert system_prompt_for(DEFAULT_STYLE) == SYSTEM_PROMPT


def test_style_changes_system_prompt_but_not_evidence():
    a = build_resolution_prompt("login down", SIMILAR, "IT Support", "stepwise")
    b = build_resolution_prompt("login down", SIMILAR, "IT Support", "handover")
    assert a[0]["content"] != b[0]["content"]
    assert a[1]["content"] == b[1]["content"]      # identical evidence block
    assert "IMPACT" in b[0]["content"]
    assert "[OPM-00042]" in b[1]["content"]


def test_unknown_style_does_not_raise_in_prompt_build():
    msgs = build_resolution_prompt("x", [], None, "nonsense")
    assert msgs[0]["content"] == SYSTEM_PROMPT


# ------------------------------------------------- search / retrieval mode
def test_search_modes_map_to_retrieval_flag():
    assert set(composer.SEARCH_ORDER) == set(composer.SEARCH_MODES)
    assert composer.use_retrieval("kb") is True
    assert composer.use_retrieval(None) is True     # default is grounded
    assert composer.use_retrieval("off") is False


# ------------------------------------------------- session-state helpers
def test_init_state_seeds_once_and_does_not_clobber():
    state: dict[str, object] = {"composer_text": "already typed"}
    composer.init_state(state, top_k=5, threshold=0.4)
    assert state["composer_text"] == "already typed"
    assert state["writing_style"] == DEFAULT_STYLE
    assert state["search_mode"] == "kb"
    assert state["variant"] == "auto"
    assert state["pending_attachment"] is None
    assert state["composer_top_k"] == 5
    assert state["composer_threshold"] == 0.4

    state["composer_top_k"] = 9
    composer.init_state(state, top_k=5, threshold=0.4)   # idempotent
    assert state["composer_top_k"] == 9


def test_submit_queues_text_and_clears_the_field():
    state: dict[str, object] = {"composer_text": "  printer jammed  ",
                                "_asr_note": "stale note"}
    composer.submit(state)
    assert state["_pending"] == "printer jammed"
    assert state["composer_text"] == ""
    assert "_asr_note" not in state


def test_submit_ignores_blank_input():
    state: dict[str, object] = {"composer_text": "   "}
    composer.submit(state)
    assert "_pending" not in state
    assert state["composer_text"] == ""


def test_drain_voice_draft_moves_transcript_into_the_text_field():
    state: dict[str, object] = {"composer_text": "",
                                "_voice_draft": "the vpn keeps dropping"}
    assert composer.drain_voice_draft(state) is True
    assert state["composer_text"] == "the vpn keeps dropping"
    assert "_voice_draft" not in state
    # second call is a no-op, so a rerun cannot re-inject the transcript
    assert composer.drain_voice_draft(state) is False
    assert state["composer_text"] == "the vpn keeps dropping"


def test_audio_signature_is_stable_and_content_addressed():
    a = composer.audio_signature(b"RIFF....fake wav bytes")
    b = composer.audio_signature(b"RIFF....fake wav bytes")
    c = composer.audio_signature(b"RIFF....different bytes")
    assert a == b and a != c
    assert len(a) == 16 and a.isalnum()


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
