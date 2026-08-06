# pages/3_Monitor.py
"""OpsMedic /monitor — enterprise LLM-Ops console (9 tabs).

Overview · Golden Signals · LLM Gateway · Model Monitoring · Metrics ·
Fine-tune · System Health ·
Request Logs · App Details. Reuses the reference dashboard's visual
language (KPI tiles, Plotly donut gauges, status badges) over
OpsMedic's live SQLite metrics store — no MLflow/Prefect needed.
"""
from __future__ import annotations

import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from finetune.data import default_paths
from llmops.insights import (
    feedback_summary,
    headline_delta,
    per_class_table,
    training_curves,
)
from llmops.metrics import MetricsLogger
from llmops.system_stats import db_file_sizes, system_stats
from src import config
from ui import nav, theme
from ui import golden_signals, metrics_dashboard, model_card_panel

st.set_page_config(page_title="OpsMedic · Monitor", page_icon="📡", layout="wide")
theme.inject_css()
nav.render_page_chrome(active_label="📡  Monitor")

logger = MetricsLogger(config.METRICS_DB_PATH, price_table=config.PRICE_PER_MTOK)
summary = logger.summary()
rows = logger.recent(limit=1000)
per = logger.summary_by_subtask()
df = pd.DataFrame(rows)

with st.sidebar:
    st.caption(f"Metrics store: `{config.METRICS_DB_PATH}`")
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

page_tabs = st.tabs([
    "Overview", "Golden Signals", "LLM Gateway", "Model Monitoring",
    "Metrics", "Fine-tune", "System Health", "Request Logs", "App Details",
])
# Metrics and Fine-tune used to be separate sidebar pages
# (1_LLMOps_Dashboard.py and 2_Model_Card.py). They are tabs now so
# everything operational lives on one page; their bodies moved to
# ui/metrics_dashboard.py and ui/model_card_panel.py.
# NOTE: the old "Fine-tune" tab was removed - it duplicated
# pages/2_Model_Card.py, which owns model evaluation. This page answers
# "is it healthy right now?"; the Model Card answers "can I trust this
# model?". See the Monitor / Experiments / Model Card split in the report.

# ============================================================ OVERVIEW
with page_tabs[0]:
    theme.page_header(
        "Operations", "Overview",
        "Live health of the OpsMedic incident-copilot pipeline.",
    )
    if summary["requests"] == 0:
        st.info("No metrics yet — run a chat in the app or "
                "`python scripts/smoke_test.py`.")
    else:
        fb = feedback_summary(rows)
        c = st.columns(4)
        with c[0]:
            theme.tile("🧠", "Requests served", str(summary["requests"]),
                       "all sub-tasks")
        with c[1]:
            state = theme.rate_state(summary["error_rate"])
            theme.tile("✅", "Error rate",
                       f"{summary['error_rate'] * 100:.1f}%",
                       badge_html=theme.status_badge(
                           "healthy" if state == "ok" else "watch", state))
        with c[2]:
            theme.tile("⚡", "Latency p95",
                       f"{summary['latency_p95_ms'] / 1000:.2f}s",
                       "incl. cold start")
        with c[3]:
            theme.tile("💰", "Est. spend",
                       f"${summary['cost_usd']:.4f}", "Groq tokens")

        st.write("")
        c2 = st.columns(4)
        with c2[0]:
            theme.tile("🎯", "Cache hit rate",
                       f"{summary['cache_hit_rate'] * 100:.0f}%",
                       "cost optimisation")
        with c2[1]:
            theme.tile("📈", "Throughput",
                       f"{summary['throughput_rpm']:.1f}", "req/min")
        with c2[2]:
            theme.tile("🔤", "Tokens in/out",
                       f"{summary['tokens_in']}/{summary['tokens_out']}")
        with c2[3]:
            theme.tile("👍", "User feedback",
                       f"{fb['up']} / {fb['down']}", "up / down")

        st.divider()
        gcols = st.columns(3)
        with gcols[0]:
            st.plotly_chart(
                theme.donut_gauge(1 - summary["error_rate"], "success rate",
                                  theme.OK),
                use_container_width=True)
        with gcols[1]:
            st.plotly_chart(
                theme.donut_gauge(summary["cache_hit_rate"], "cache hit",
                                  theme.PRIMARY),
                use_container_width=True)
        with gcols[2]:
            fb_ratio = (fb["up"] / fb["total"]) if fb["total"] else 0.0
            st.plotly_chart(
                theme.donut_gauge(fb_ratio, "positive feedback", theme.VIOLET),
                use_container_width=True)

# ========================================================= LLM GATEWAY
with page_tabs[1]:
    golden_signals.render(logger)

with page_tabs[2]:
    theme.page_header(
        "LLM Gateway", "Model call metrics",
        "Per-sub-task and per-model latency, tokens, cost and errors — "
        "the LLM/SLM gateway view.",
    )
    if not per:
        st.info("No calls logged yet.")
    else:
        per_df = pd.DataFrame(per)
        show = per_df.rename(columns={
            "subtask": "Sub-task", "requests": "Calls",
            "latency_p50_ms": "p50 ms", "latency_p95_ms": "p95 ms",
            "tokens_in": "Tok in", "tokens_out": "Tok out",
            "cost_usd": "Cost $", "error_rate": "Err", "cache_hit_rate": "Cache",
        })
        st.dataframe(show, use_container_width=True, hide_index=True)

        cc = st.columns(2)
        with cc[0]:
            fig = px.bar(per_df, x="subtask", y="latency_p95_ms",
                         title="p95 latency by sub-task (ms)",
                         color="latency_p95_ms",
                         color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"])
            theme.style_plotly(fig)
            st.plotly_chart(fig, use_container_width=True)
        with cc[1]:
            tok = per_df.melt(id_vars="subtask",
                              value_vars=["tokens_in", "tokens_out"],
                              var_name="dir", value_name="tokens")
            fig2 = px.bar(tok, x="subtask", y="tokens", color="dir",
                          barmode="group", title="Token usage by sub-task")
            theme.style_plotly(fig2)
            st.plotly_chart(fig2, use_container_width=True)

        # per-MODEL rollup (gateway groups by model string incl. version stamp)
        if not df.empty:
            st.subheader("Per-model rollup")
            model_roll = (
                df[df["subtask"] != "feedback"]
                .groupby("model")
                .agg(calls=("id", "count"),
                     avg_ms=("latency_ms", "mean"),
                     cost=("cost_usd", "sum"),
                     errors=("status", lambda s: (s == "error").sum()))
                .reset_index()
            )
            model_roll["avg_ms"] = model_roll["avg_ms"].round(1)
            model_roll["cost"] = model_roll["cost"].round(6)
            st.dataframe(model_roll, use_container_width=True, hide_index=True)

# ==================================================== MODEL MONITORING
with page_tabs[3]:
    theme.page_header(
        "Model Monitoring", "Quality, drift & feedback",
        "Latency trend, request volume, and user-feedback signal over time.",
    )
    if df.empty:
        st.info("No data yet.")
    else:
        work = df[df["subtask"] != "feedback"].copy()
        work["time"] = pd.to_datetime(work["ts"], unit="s")

        mc = st.columns(2)
        with mc[0]:
            trend = work.sort_values("time").pivot_table(
                index="time", columns="subtask", values="latency_ms")
            fig = go.Figure()
            for col in trend.columns:
                fig.add_trace(go.Scatter(x=trend.index, y=trend[col],
                                         mode="lines+markers", name=col))
            fig.update_layout(title="Latency over time (ms)")
            theme.style_plotly(fig)
            st.plotly_chart(fig, use_container_width=True)
        with mc[1]:
            vol = (work.set_index("time")
                   .groupby("subtask")
                   .resample("1min").size()
                   .rename("calls").reset_index())
            fig2 = px.area(vol, x="time", y="calls", color="subtask",
                           title="Request volume per minute")
            theme.style_plotly(fig2)
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Feedback signal")
        fb = feedback_summary(rows)
        fcols = st.columns([1, 3])
        with fcols[0]:
            theme.tile("💬", "Total feedback", str(fb["total"]),
                       f"👍 {fb['up']} · 👎 {fb['down']}")
        with fcols[1]:
            if fb["total"]:
                fig3 = go.Figure(go.Bar(
                    x=["👍 helpful", "👎 not helpful"],
                    y=[fb["up"], fb["down"]],
                    marker_color=[theme.OK, theme.ERR]))
                fig3.update_layout(title="User feedback tally", height=240)
                theme.style_plotly(fig3)
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.caption("No 👍/👎 captured yet — use the buttons in chat.")

        st.caption(
            "Drift note: with a labelled stream, feedback + per-class F1 "
            "trends form the drift signal; monitored here as the volume and "
            "sentiment of thumbs feedback against latency stability."
        )

# ============================================================ FINE-TUNE
with page_tabs[4]:
    metrics_dashboard.render(logger)

with page_tabs[5]:
    model_card_panel.render()

with page_tabs[6]:
    theme.page_header(
        "System Health", "Host resources & data stores",
        "Local runtime stats and the sizes of the metrics/cache stores.",
    )
    s = system_stats()
    sc = st.columns(4)
    for col, (icon, label, val, vmax) in zip(sc, [
        ("🖥️", "CPU", s["cpu_percent"], 100),
        ("💾", "Memory", s["memory_percent"], 100),
        ("🗄️", "Disk", s["disk_percent"], 100),
        ("🐍", "Python", None, None),
    ]):
        with col:
            if val is None and label == "Python":
                theme.tile(icon, label, s["python"], s["platform"])
            elif val is None:
                theme.tile(icon, label, "n/a", "psutil not available")
            else:
                st.plotly_chart(
                    theme.donut_gauge(val, f"{label} %", theme.PRIMARY, vmax=100),
                    use_container_width=True)
    if not s["psutil"]:
        st.caption("Install `psutil` for live CPU/memory gauges "
                   "(already in requirements.txt).")

    st.subheader("Data stores")
    sizes = db_file_sizes({
        "Metrics DB": config.METRICS_DB_PATH,
        "Cache DB": config.CACHE_DB_PATH,
        "Tickets KB": config.TICKETS_CSV,
        "FAISS index": os.path.join(config.INDEX_DIR, "index.faiss"),
    })
    st.dataframe(pd.DataFrame(sizes), use_container_width=True, hide_index=True)

# ========================================================= REQUEST LOGS
with page_tabs[7]:
    theme.page_header(
        "Request Logs", "Raw metric rows",
        "Every metered call, newest first — filterable stream.",
    )
    if df.empty:
        st.info("No requests logged yet.")
    else:
        fcols = st.columns(3)
        subtasks = ["(all)"] + sorted(df["subtask"].unique().tolist())
        pick = fcols[0].selectbox("Sub-task", subtasks)
        status_pick = fcols[1].selectbox("Status", ["(all)", "ok", "error"])
        only_cached = fcols[2].checkbox("Cache hits only")

        view = df.copy()
        if pick != "(all)":
            view = view[view["subtask"] == pick]
        if status_pick != "(all)":
            view = view[view["status"] == status_pick]
        if only_cached:
            view = view[view["cache_hit"] == 1]

        view = view[[
            "id", "ts", "subtask", "model", "latency_ms", "tokens_in",
            "tokens_out", "cost_usd", "status", "cache_hit", "error",
        ]].copy()
        view["ts"] = pd.to_datetime(view["ts"], unit="s").dt.strftime(
            "%Y-%m-%d %H:%M:%S")
        st.dataframe(view.head(200), use_container_width=True, hide_index=True)
        st.caption(f"Showing {min(len(view), 200)} of {len(view)} matching rows. "
                   "Fine-tuned classifier rows carry an @timestamp model version.")

# ========================================================== APP DETAILS
with page_tabs[8]:
    theme.page_header(
        "Application Details", "Configuration & stack",
        "What this deployment is running.",
    )
    dc = st.columns(2)
    with dc[0]:
        st.markdown("**Sub-tasks & models**")
        st.table(pd.DataFrame([
            {"Sub-task": "Retrieval (NLP)", "Model": config.EMBEDDING_MODEL},
            {"Sub-task": "Resolution (NLP/GenAI)", "Model": config.RESOLUTION_MODEL},
            {"Sub-task": "Summarize (NLP)", "Model": config.SUMMARIZER_MODEL},
            {"Sub-task": "Classify (NLP, fine-tuned)",
             "Model": config.CLASSIFIER_BASE_MODEL + " → fine-tuned"},
            {"Sub-task": "ASR (Speech)", "Model": config.ASR_MODEL},
            {"Sub-task": "TTS (Speech)", "Model": config.TTS_MODEL},
        ]))
    with dc[1]:
        st.markdown("**Runtime config**")
        st.json({
            "product": config.PRODUCT_NAME,
            "org": config.ORG_NAME,
            "groq_key_set": bool(config.GROQ_API_KEY),
            "cache_enabled": config.CACHE_ENABLED,
            "cache_ttl_s": config.CACHE_TTL_SECONDS,
            "top_k": config.TOP_K,
            "similarity_threshold": config.SIMILARITY_THRESHOLD,
            "dataset": config.HF_DATASET,
        })
    st.caption("Stack: Streamlit · Hugging Face Transformers · FAISS · "
               "sentence-transformers · Groq (OpenAI-compatible) · SQLite · Plotly")
