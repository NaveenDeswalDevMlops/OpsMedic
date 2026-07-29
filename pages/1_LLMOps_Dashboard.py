# pages/1_LLMOps_Dashboard.py
"""LLMOps dashboard: live metrics from the SQLite store.

Displays (>=5 assignment metrics, all persisted):
  latency p50/p95 - tokens in/out - cost/request - error rate -
  cache-hit rate - throughput - user feedback - model versions used.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from llmops.insights import feedback_summary
from llmops.metrics import MetricsLogger
from src import config
from ui import nav, theme

st.set_page_config(page_title="LLMOps Dashboard", page_icon="📊", layout="wide")
theme.inject_css()
nav.render_page_chrome(active_label="📊  Metrics")
st.title("📊 LLMOps Dashboard")
st.caption(
    f"{config.PRODUCT_NAME} · metrics persisted in {config.METRICS_DB_PATH} · "
    "refresh the page after new requests"
)

logger = MetricsLogger(config.METRICS_DB_PATH, price_table=config.PRICE_PER_MTOK)
summary = logger.summary()
rows = logger.recent(limit=500)

if summary["requests"] == 0:
    st.info("No metrics yet — run a journey in the main app or "
            "`python scripts/smoke_test.py` first.")
    st.stop()

# ------------------------------------------------------ headline metrics
c = st.columns(7)
c[0].metric("Requests", summary["requests"])
c[1].metric("Latency p50", f"{summary['latency_p50_ms'] / 1000:.2f}s")
c[2].metric("Latency p95", f"{summary['latency_p95_ms'] / 1000:.2f}s",
            help="Includes first-run model downloads (cold start).")
c[3].metric("Tokens in/out", f"{summary['tokens_in']}/{summary['tokens_out']}")
c[4].metric("Est. cost", f"${summary['cost_usd']:.4f}",
            help="Groq calls priced per PRICE_PER_MTOK; local models are $0.")
c[5].metric("Error rate", f"{summary['error_rate'] * 100:.1f}%")
c[6].metric("Cache hit rate", f"{summary['cache_hit_rate'] * 100:.1f}%")

fb = feedback_summary(rows)
st.caption(
    f"Throughput: {summary['throughput_rpm']} req/min · "
    f"User feedback: 👍 {fb['up']} / 👎 {fb['down']}"
)

st.divider()
left, right = st.columns([3, 2])

# ------------------------------------------------------ per-subtask table
with left:
    st.subheader("Per-sub-task breakdown")
    per = pd.DataFrame(logger.summary_by_subtask())
    st.dataframe(per, use_container_width=True, hide_index=True)

# ------------------------------------------------------ latency over time
with right:
    st.subheader("Latency over time (last 500)")
    df = pd.DataFrame(rows)
    df = df[df["subtask"] != "feedback"]
    df["time"] = pd.to_datetime(df["ts"], unit="s")
    chart = (
        df.sort_values("time")[["time", "subtask", "latency_ms"]]
        .pivot_table(index="time", columns="subtask", values="latency_ms")
    )
    st.line_chart(chart, height=280)

st.divider()

# ------------------------------------------------------ recent requests
st.subheader("Recent requests (raw metric rows)")
show = pd.DataFrame(rows)[
    ["id", "ts", "subtask", "model", "latency_ms", "tokens_in",
     "tokens_out", "cost_usd", "status", "cache_hit", "error"]
].head(50)
show["ts"] = pd.to_datetime(show["ts"], unit="s").dt.strftime("%H:%M:%S")
st.dataframe(show, use_container_width=True, hide_index=True)
st.caption(
    "Note the `model` column: fine-tuned classifier entries carry an "
    "@timestamp version stamp — model-version tracking in action."
)
