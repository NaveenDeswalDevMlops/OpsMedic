# app.py
"""OpsMedic — GenAI Incident Copilot (Claude-style chat UI).

Dark-light sidebar with workspace nav buttons + conversation history,
a floating speaker hero with a LIVE clock, streaming responses, a
"Sources" card strip, and a Claude-style composer with working popover
widgets (model→variant selector, Attach for incident files/screenshots/
SOPs, and a mic). The LLM-Ops console is the Monitor page.

Journey per message: [attach/voice -> incident] -> [classify: fine-tuned
DistilBERT] -> [retrieve similar resolved tickets] -> [resolve: Groq RAG,
streamed] -> [summarize] -> optional [TTS]. Every call is metered; each
turn is persisted to the conversation store.

Run:  streamlit run app.py
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from llmops.attachments import merge_incident
from llmops.cache import ResponseCache
from llmops.conversations import ConversationStore, make_title
from llmops.metrics import MetricsLogger
from models.asr import ASRTask
from models.classifier import ClassifierTask
from models.resolution import ResolutionTask
from models.retrieval import RetrievalTask
from models.summarizer import SummarizerTask
from models.tts import TTSTask
from src import config
from ui import composer as composer_ui
from ui import nav, theme

st.set_page_config(
    page_title=f"{config.PRODUCT_NAME} — GenAI Incident Copilot",
    page_icon="🩺",
    layout="wide",
)
theme.inject_css()

CONV_DB = config.METRICS_DB_PATH.replace("llmops_metrics.db", "conversations.db")
SUGGESTIONS = [
    "Several users can't sign in to the account portal; resets aren't arriving.",
    "Shared printer on floor 3 shows jobs queued but nothing prints.",
    "Customer disputes a duplicate charge on last month's invoice.",
    "VPN keeps disconnecting every few minutes on Windows 11.",
]


# ---------------------------------------------------------- singletons
@st.cache_resource
def get_metrics() -> MetricsLogger:
    return MetricsLogger(config.METRICS_DB_PATH, price_table=config.PRICE_PER_MTOK)


@st.cache_resource
def get_cache() -> ResponseCache | None:
    return ResponseCache(config.CACHE_DB_PATH, config.CACHE_TTL_SECONDS) \
        if config.CACHE_ENABLED else None


@st.cache_resource
def get_store() -> ConversationStore:
    return ConversationStore(CONV_DB)


@st.cache_resource
def get_task(kind: str, variant: str = "auto") -> Any:
    kw = {"metrics": get_metrics(), "cache": get_cache()}
    if kind == "classify":
        return ClassifierTask(variant=variant, **kw)
    return {"asr": ASRTask, "retrieve": RetrievalTask, "resolve": ResolutionTask,
            "summarize": SummarizerTask, "tts": TTSTask}[kind](**kw)


def log_feedback(value: int, ctx: str) -> None:
    with get_metrics().track("feedback", model="user") as rec:
        rec.add_extra(value=value, context=ctx)


def last_latency_ms() -> float | None:
    """Latency of the most recent metered call, read back from SQLite.

    Lets the composer show a real measured Whisper time instead of an
    estimate (BaseSubTask.run's envelope omits latency by design).
    """
    rows = get_metrics().recent(1)
    return float(rows[0]["latency_ms"]) if rows else None


# ------------------------------------------------------ session bootstrap
store = get_store()
# conversation_id stays None until the first message -> no empty threads
if "conversation_id" not in st.session_state:
    convs = store.list_conversations(1)
    st.session_state.conversation_id = convs[0]["id"] if convs else None
if "messages" not in st.session_state:
    cid = st.session_state.conversation_id
    st.session_state.messages = store.get_messages(cid) if cid else []
if "variant" not in st.session_state:
    st.session_state.variant = "auto"
if "pending_attachment" not in st.session_state:
    st.session_state.pending_attachment = None


def start_new_chat() -> None:
    st.session_state.conversation_id = None   # created lazily on first send
    st.session_state.messages = []
    st.session_state.pending_attachment = None


def load_chat(cid: str) -> None:
    st.session_state.conversation_id = cid
    st.session_state.messages = store.get_messages(cid)


# ------------------------------------------------------------- sidebar
with st.sidebar:
    nav.render_brand()
    nav.render_workspace_nav(active_label="💬  Chat")

    if st.button("＋  New chat", use_container_width=True, key="newchat"):
        start_new_chat()
        st.rerun()

    # ---- conversation history (collapsible; day-grouped recents) ----
    with nav.section_header("Recents", expanded=True):
        grouped = store.grouped(limit=40)
        if not grouped:
            st.caption("No conversations yet — start chatting below.")
        for label, convs in grouped:
            st.caption(label.upper())
            for conv in convs:
                active = conv["id"] == st.session_state.conversation_id
                row = st.columns([6, 1], gap="small")
                title = ("● " if active else "") + conv["title"]
                if row[0].button(title, key=f"conv_{conv['id']}",
                                 use_container_width=True,
                                 help=conv["title"]):
                    load_chat(conv["id"])
                    st.rerun()
                if row[1].button("🗑", key=f"del_{conv['id']}",
                                 help="Delete this conversation"):
                    store.delete(conv["id"])
                    if active:                      # deleted the open one
                        start_new_chat()
                    st.rerun()

    # ---- Knowledge Base panel (collapsible; counts + Rebuild) ----
    with nav.section_header("Knowledge Base", expanded=False):
        from llmops.system_stats import kb_stats
        kb = kb_stats(config.TICKETS_CSV, config.INDEX_DIR, config.SOPS_DIR)
        kb_cols = st.columns(2)
        kb_cols[0].metric("Tickets", kb["tickets_indexed"])
        kb_cols[1].metric("SOP pages", kb["sop_pages"])
        st.caption(f"Index built: {kb['index_built'] or '—'}")
        if st.button("🔄 Rebuild Index", use_container_width=True,
                     key="rebuild_idx"):
            with st.spinner("Rebuilding FAISS index over the ticket KB…"):
                import subprocess
                import sys
                proc = subprocess.run(
                    [sys.executable, "scripts/build_index.py"],
                    capture_output=True, text=True,
                )
            if proc.returncode == 0:
                st.success("Index rebuilt. Reloading…")
                get_task.clear()      # drop cached retrieval task (stale index)
                st.rerun()
            else:
                st.error("Rebuild failed — see console.")
                st.caption((proc.stderr or proc.stdout)[-300:])

    # Retrieval depth (Top-K / threshold) now lives in the composer's
    # "Search" popover, next to the retrieval on/off switch it belongs
    # with — the sidebar was carrying unrelated controls.
    theme.profile_card()


# -------------------------------------------------- assistant meta render
def render_assistant_meta(meta: dict[str, Any], key: str) -> None:
    cls = meta.get("classify") or {}
    if cls.get("output"):
        o = cls["output"]
        st.markdown(
            theme.status_badge(
                f"{o['label']} · {o['confidence']:.0%} · {o['variant']} model",
                "info"),
            unsafe_allow_html=True,
        )
    theme.source_cards(meta.get("similar") or [])
    if meta.get("summary"):
        st.info(f"**Handover summary:** {meta['summary']}")
    if meta.get("grounded_on") or meta.get("style"):
        st.caption(f"Grounded on: {', '.join(meta.get('grounded_on') or []) or '—'} · "
                   f"SOP: {meta.get('sop') or '—'} · "
                   f"Style: {composer_ui.style_label(meta.get('style'))}")
    mc = st.columns([1, 1, 1, 6])
    mc[0].button("👍", key=f"up_{key}", on_click=log_feedback, args=(1, "resolution"))
    mc[1].button("👎", key=f"dn_{key}", on_click=log_feedback, args=(-1, "resolution"))
    if meta.get("summary") and mc[2].button("🔊", key=f"tts_{key}"):
        with st.spinner("Generating audio…"):
            r = get_task("tts").run(
                {"text": meta["summary"], "out_path": "./data/tts_out.wav"})
        if r["metrics"]["status"] == "ok":
            o = r["output"]
            st.audio(o["audio_path"])
            st.caption(
                f"{o['duration_s']}s · {o['chunks']} phrase"
                f"{'s' if o['chunks'] != 1 else ''} · presence "
                f"{o['presence_ratio'] * 100:.1f}% of spectrum"
                + (" · truncated to fit" if o.get("truncated") else "")
            )
        else:
            st.error(r["metrics"].get("error", "Speech synthesis failed."))


# --------------------------------------------------- top breadcrumb
theme.topbar("Overview", "Ask-AI")

# The transcript lives in its own container declared BEFORE the composer,
# so the composer stays pinned visually below the conversation and a
# newly streamed answer appears above it instead of underneath it.
chat_area = st.container()

with chat_area:
    # ---- empty state: hero + suggestion chips ----
    if not st.session_state.messages:
        theme.hero(config.PRODUCT_NAME)
        st.markdown("<div style='text-align:center;color:#7F7E7E;font-size:12px;"
                    "margin:14px 0 6px 0'>Try one of these</div>",
                    unsafe_allow_html=True)
        chips = st.container()   # styled via theme.chips_anchor()
        with chips:
            theme.chips_anchor()
            chip_cols = st.columns(2)
            for i, s in enumerate(SUGGESTIONS):
                if chip_cols[i % 2].button(s, key=f"sugg_{i}",
                                           use_container_width=True):
                    st.session_state._pending = s
                    st.rerun()

    # ---- replay chat history ----
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"],
                             avatar="🩺" if msg["role"] == "assistant"
                             else "🧑‍💻"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("meta"):
                render_assistant_meta(msg["meta"], key=f"hist{i}")

# ==================================== COMPOSER (ui/composer.py owns the box)
settings = composer_ui.render(
    asr_runner=lambda payload: get_task("asr").run(payload),
    asr_latency_ms=last_latency_ms,
    model_label=config.RESOLUTION_MODEL.split("/")[-1],
    groq_connected=bool(config.GROQ_API_KEY),
    default_top_k=config.TOP_K,
    default_threshold=config.SIMILARITY_THRESHOLD,
)
# apply the composer's Search settings to this run's retrieval
config.TOP_K = settings.top_k
config.SIMILARITY_THRESHOLD = settings.similarity_threshold

st.caption(
    f"Style: **{composer_ui.style_label(settings.style)}** · Evidence: "
    f"**{'Ticket KB (RAG)' if settings.use_retrieval else 'no retrieval'}** · "
    f"Triage: **{settings.variant}** · Top-K {settings.top_k} @ "
    f"{settings.similarity_threshold:.2f}"
)

# incoming message: from the composer (Enter/Send), a suggestion chip, or voice
incident_in = st.session_state.pop("_pending", None)

# --------------------------------------------------------- run a turn
if incident_in:
    # fold any staged attachment into the incident text
    attach = st.session_state.pending_attachment
    incident = merge_incident(incident_in, attach) if attach else incident_in
    st.session_state.pending_attachment = None

    # create the conversation lazily on first message, titled from it
    if not st.session_state.conversation_id:
        st.session_state.conversation_id = store.create(make_title(incident_in))
    cid = st.session_state.conversation_id
    store.add_message(cid, "user", incident)
    store.rename_if_default(cid, incident_in)
    st.session_state.messages.append({"role": "user", "content": incident})

    with chat_area:
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(incident)

        with st.chat_message("assistant", avatar="🩺"):
            meta: dict[str, Any] = {}
            with st.status("Working through the incident…",
                           expanded=False) as status:
                st.write("Classifying queue…")
                r_cls = get_task("classify", settings.variant).run(incident)
                meta["classify"] = r_cls
                category = (r_cls["output"] or {}).get("label")

                if settings.use_retrieval:
                    st.write("Retrieving similar resolved tickets…")
                    sims = get_task("retrieve").run(incident)["output"] or []
                else:
                    st.write("Retrieval disabled — SOP-only answer.")
                    sims = []
                meta["similar"] = sims
                status.update(label="Diagnosis ready — drafting resolution…",
                              state="running")

            if r_cls["output"]:
                o = r_cls["output"]
                st.markdown(
                    theme.status_badge(
                        f"{o['label']} · {o['confidence']:.0%} · "
                        f"{o['variant']} model", "info"),
                    unsafe_allow_html=True,
                )
            theme.source_cards(sims)

            st.markdown("**Recommended resolution**")
            payload = {"incident": incident, "similar": sims,
                       "category": category, "style": settings.style}
            full = st.write_stream(get_task("resolve").stream(payload))
            meta["grounded_on"] = [t["ticket_id"] for t in sims[:3]]
            meta["sop"] = category
            meta["style"] = settings.style

            if full and len(full) >= 40:
                meta["summary"] = get_task("summarize").run(full)["output"]
            else:
                meta["summary"] = full
            if meta["summary"]:
                st.info(f"**Handover summary:** {meta['summary']}")

    content = full or "No resolution available."
    store.add_message(cid, "assistant", content, meta=meta)
    st.session_state.messages.append(
        {"role": "assistant", "content": content, "meta": meta})
    st.rerun()