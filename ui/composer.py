# ui/composer.py
"""The OpsMedic chat composer — one bordered box, reference layout.

Reproduces the reference React composer exactly: a hint line and the
text field on top, then a single control row with pill buttons on the
left (Attach / Search / Writing Styles / Voice) and a dark circular
send button on the right.

Everything in the row is a REAL Streamlit widget wired to a real code
path — no decorative HTML:

  Attach          -> llmops.attachments.extract_text, folded into the
                     incident text before retrieval
  Search          -> retrieval scope (RAG on/off) + Top-K + similarity
                     threshold; these used to clutter the sidebar
  Writing Styles  -> models.resolution.STYLE_PROMPTS, i.e. a versioned
                     prompt swap that the metrics store records
  Voice (mic)     -> st.audio_input -> Whisper (models.asr.ASRTask);
                     the transcript is dropped into the text field so it
                     can be edited before sending

The module owns no models and no metrics logic: the caller injects an
`asr_runner` callable, so every pure helper below is unit-testable with
plain dicts and no Streamlit runtime.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from typing import Any, Callable, MutableMapping

import streamlit as st

from llmops.attachments import extract_text
from models.resolution import DEFAULT_STYLE, STYLE_PROMPTS, resolve_style

# ---------------------------------------------------------------- registries

#: Display metadata for the Writing Styles popover. Keys MUST match
#: models.resolution.STYLE_PROMPTS so the label and the prompt cannot
#: drift apart (asserted in tests/test_composer_offline.py).
STYLE_LABELS: dict[str, str] = {
    "stepwise": "Step-by-step",
    "concise": "Concise",
    "customer": "Customer-facing",
    "handover": "Shift handover",
}
STYLE_ORDER: list[str] = ["stepwise", "concise", "customer", "handover"]
STYLE_HELP: dict[str, str] = {
    "stepwise": "Diagnosis, numbered fix steps, escalation condition.",
    "concise": "Four bullets, no preamble — fastest to read on a call.",
    "customer": "Polite reply to the requester, no internal jargon.",
    "handover": "IMPACT / DIAGNOSIS / ACTIONS / NEXT / ESCALATE IF.",
}

#: Retrieval scope offered by the Search popover.
SEARCH_MODES: dict[str, str] = {
    "kb": "Ticket KB (RAG)",
    "off": "No retrieval",
}
SEARCH_ORDER: list[str] = ["kb", "off"]

AUDIO_TYPES = ["wav", "flac", "ogg", "mp3", "m4a"]
ATTACH_TYPES = ["txt", "log", "md", "csv", "json", "pdf", "png", "jpg", "jpeg"]

# session-state keys owned by the composer
K_TEXT = "composer_text"
K_STYLE = "writing_style"
K_SEARCH = "search_mode"
K_VARIANT = "variant"
K_ATTACH = "pending_attachment"
K_DRAFT = "_voice_draft"        # staged transcript, moved into K_TEXT
K_ASR_SIG = "_asr_signature"    # de-dupes repeat transcription on rerun
K_ASR_NOTE = "_asr_note"        # "transcribed in 812 ms" caption
K_PENDING = "_pending"          # outgoing message picked up by app.py


@dataclass
class ComposerSettings:
    """What the composer decided this run; applied by the caller."""

    style: str = DEFAULT_STYLE
    use_retrieval: bool = True
    top_k: int = 3
    similarity_threshold: float = 0.35
    variant: str = "auto"


# ------------------------------------------------------------ pure helpers
def audio_signature(data: bytes) -> str:
    """Short stable fingerprint of audio bytes.

    Used to transcribe a recording exactly once: Streamlit re-runs the
    script on every interaction and `st.audio_input` keeps returning the
    same buffer, so without this the mic would re-run Whisper (and log a
    junk metrics row) on every click.
    """
    return hashlib.sha1(data).hexdigest()[:16]


def drain_voice_draft(state: MutableMapping[str, Any]) -> bool:
    """Move a staged transcript into the text field. Returns True if moved.

    Must be called BEFORE the text widget is instantiated: Streamlit
    forbids writing a widget's session-state key after that widget has
    been created in the same script run.
    """
    draft = state.pop(K_DRAFT, None)
    if draft is None:
        return False
    state[K_TEXT] = str(draft)
    return True


def style_label(key: str) -> str:
    """Human label for a style key ('' or unknown -> the default's label)."""
    return STYLE_LABELS[resolve_style(key)]


def use_retrieval(mode: str | None) -> bool:
    """Whether the Search setting means 'retrieve evidence'."""
    return (mode or "kb") != "off"


def init_state(state: MutableMapping[str, Any], *, top_k: int,
               threshold: float) -> None:
    """Seed every composer key exactly once (idempotent)."""
    state.setdefault(K_TEXT, "")
    state.setdefault(K_STYLE, DEFAULT_STYLE)
    state.setdefault(K_SEARCH, "kb")
    state.setdefault(K_VARIANT, "auto")
    state.setdefault(K_ATTACH, None)
    state.setdefault("composer_top_k", int(top_k))
    state.setdefault("composer_threshold", float(threshold))


def submit(state: MutableMapping[str, Any]) -> None:
    """Queue the current text as the outgoing message and clear the field."""
    value = str(state.get(K_TEXT, "")).strip()
    if value:
        state[K_PENDING] = value
    state[K_TEXT] = ""
    state.pop(K_ASR_NOTE, None)


# --------------------------------------------------------------- rendering
def _on_submit() -> None:
    submit(st.session_state)


def render(
    *,
    asr_runner: Callable[[dict[str, Any]], dict[str, Any]],
    asr_latency_ms: Callable[[], float | None],
    model_label: str,
    groq_connected: bool,
    default_top_k: int,
    default_threshold: float,
) -> ComposerSettings:
    """Draw the composer and return the settings the caller should apply.

    `asr_runner` receives {"array", "sampling_rate"} and returns the
    standard {"output", "metrics"} envelope (i.e. ASRTask.run).
    `asr_latency_ms` reports the latency of the last logged call so the
    UI can show real measured transcription time rather than a guess.
    """
    init_state(st.session_state, top_k=default_top_k,
               threshold=default_threshold)
    # must happen before the text widget below is created
    drain_voice_draft(st.session_state)

    box = st.container()
    with box:
        # CSS anchor, emitted FIRST: theme.py scopes every composer rule to
        # the block whose first child holds this span. A bare :has(#anchor)
        # would also match all ancestor blocks (Streamlit wraps every
        # block), which is what leaked the send-button styling onto the
        # suggestion chips. The border/radius/shadow are applied to this
        # block by CSS, so no st.container(border=True) is needed.
        st.markdown('<span id="opm-composer-anchor"></span>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="opm-composer-hint">Ask me anything — I\'m your '
            'incident copilot <span>with retrieval, fine-tuned triage and '
            'voice.</span></div>',
            unsafe_allow_html=True,
        )
        st.text_input(
            "incident",
            key=K_TEXT,
            label_visibility="collapsed",
            placeholder="Describe the incident, or record it with the mic…",
            on_change=_on_submit,
        )

        # staged context lines (plain text: no extra st.button in this box,
        # which is what lets theme.py style the send button by structure)
        attach = st.session_state.get(K_ATTACH)
        if attach:
            st.markdown(
                f'<div class="opm-composer-chip">📎 {attach.get("kind")} '
                f'attached · {attach.get("chars", 0)} chars of context — '
                'remove it from the Attach menu.</div>',
                unsafe_allow_html=True,
            )
        note = st.session_state.get(K_ASR_NOTE)
        if note:
            st.markdown(f'<div class="opm-composer-chip">🎙 {note}</div>',
                        unsafe_allow_html=True)

        cols = st.columns([1.0, 1.0, 1.65, 0.95, 3.6, 0.62], gap="small")

        # ---- Attach -------------------------------------------------
        with cols[0]:
            with st.popover("📎 Attach", use_container_width=True):
                st.caption("Log file, exported ticket, SOP PDF or screenshot.")
                up = st.file_uploader(
                    "Attach a file", type=ATTACH_TYPES,
                    key="attach_uploader", label_visibility="collapsed",
                )
                if up is not None:
                    info = extract_text(up.name, up.getvalue())
                    st.session_state[K_ATTACH] = info
                    if info["kind"] == "error":
                        st.error(info["note"])
                    else:
                        st.success(f"Attached: {up.name}")
                        if info["note"]:
                            st.caption(info["note"])
                        if info["chars"]:
                            st.caption(f"{info['chars']} chars will be added "
                                       "to the incident context.")
                if st.session_state.get(K_ATTACH) and st.button(
                        "Remove attachment", key="rm_attach",
                        use_container_width=True):
                    st.session_state[K_ATTACH] = None
                    st.rerun()

        # ---- Search (retrieval scope + knobs) -----------------------
        with cols[1]:
            with st.popover("🔍 Search", use_container_width=True):
                st.markdown("**Evidence retrieval**")
                current = st.session_state[K_SEARCH]
                st.session_state[K_SEARCH] = st.radio(
                    "Scope", SEARCH_ORDER,
                    index=SEARCH_ORDER.index(current)
                    if current in SEARCH_ORDER else 0,
                    format_func=lambda k: SEARCH_MODES[k],
                    key="search_pick", label_visibility="collapsed",
                    help="'No retrieval' answers from the SOP and the model "
                         "alone — useful for showing what grounding adds.",
                )
                st.markdown("**Retrieval depth**")
                st.slider("Top-K similar tickets", 1, 10,
                          key="composer_top_k")
                st.slider("Similarity threshold", 0.0, 1.0, step=0.05,
                          key="composer_threshold")

        # ---- Writing Styles -----------------------------------------
        with cols[2]:
            # Fixed label (as in the reference) rather than the active
            # style name: a variable-width label was overflowing the
            # column and colliding with the Voice pill. The active style
            # is reported in the settings caption under the box.
            with st.popover("✍ Writing Styles", use_container_width=True):
                st.markdown("**Writing style**")
                st.session_state[K_STYLE] = st.radio(
                    "Style", STYLE_ORDER,
                    index=STYLE_ORDER.index(resolve_style(
                        st.session_state[K_STYLE])),
                    format_func=lambda k: STYLE_LABELS[k],
                    key="style_pick", label_visibility="collapsed",
                )
                st.caption(STYLE_HELP[resolve_style(st.session_state[K_STYLE])])
                st.divider()
                st.markdown("**Triage model**")
                st.session_state[K_VARIANT] = st.radio(
                    "Variant", ["auto", "finetuned", "base"],
                    index=["auto", "finetuned", "base"].index(
                        st.session_state[K_VARIANT]),
                    key="variant_pick", label_visibility="collapsed",
                    help="base = untrained head (before) · finetuned = "
                         "trained (after) · auto = finetuned if present.",
                )
                st.caption(f"Resolver: {model_label} · Groq "
                           f"{'🟢 connected' if groq_connected else '🔴 no key'}")

        # ---- Voice (mic -> Whisper) ---------------------------------
        with cols[3]:
            with st.popover("🎙 Voice", use_container_width=True):
                st.caption("Record or upload; Whisper transcribes into the "
                           "message box so you can edit before sending.")
                mic = st.audio_input("Record the incident", key="mic_input")
                aud = st.file_uploader("…or upload audio", type=AUDIO_TYPES,
                                       key="audio_uploader")
                src = mic or aud
                if src is not None:
                    raw = src.getvalue()
                    sig = audio_signature(raw)
                    if sig != st.session_state.get(K_ASR_SIG):
                        st.session_state[K_ASR_SIG] = sig
                        _transcribe(raw, asr_runner, asr_latency_ms)

        # cols[4] is the flexible gap that pushes send to the right edge

        # ---- Send (the only st.button inside this box) --------------
        with cols[5]:
            st.button("⬆", key="send_btn", help="Send",
                      on_click=_on_submit, use_container_width=True)

    return ComposerSettings(
        style=resolve_style(st.session_state[K_STYLE]),
        use_retrieval=use_retrieval(st.session_state[K_SEARCH]),
        top_k=int(st.session_state["composer_top_k"]),
        similarity_threshold=float(st.session_state["composer_threshold"]),
        variant=st.session_state[K_VARIANT],
    )


def _transcribe(
    raw: bytes,
    asr_runner: Callable[[dict[str, Any]], dict[str, Any]],
    asr_latency_ms: Callable[[], float | None],
) -> None:
    """Decode audio bytes, run Whisper, stage the transcript, rerun."""
    try:
        import soundfile as sf  # lazy: only needed when the mic is used

        data, sr = sf.read(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001 - bad audio must not crash the demo
        st.error(f"Could not read that audio ({type(exc).__name__}). "
                 "Try a WAV recording.")
        return

    with st.spinner("Transcribing with Whisper…"):
        result = asr_runner({"array": data, "sampling_rate": sr})

    if result["metrics"]["status"] != "ok" or not result["output"]:
        st.error(result["metrics"].get("error", "Transcription produced no text."))
        return

    st.session_state[K_DRAFT] = result["output"]
    latency = asr_latency_ms()
    st.session_state[K_ASR_NOTE] = (
        f"Transcribed by Whisper Large"
        + (f" in {latency:.0f} ms" if latency else "")
        + " — edit the text and press send."
    )
    st.rerun()
