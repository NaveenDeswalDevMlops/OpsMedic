# ui/metrics_dashboard.py
"""Metrics tab for the Monitor page (was pages/1_LLMOps_Dashboard.py).

Moved out of pages/ so it renders as a tab under Monitor rather than a
separate sidebar entry: everything operational now lives on one page.

Scope: the raw assignment metrics, persisted and unaggregated -
latency p50/p95, tokens in/out, cost per request, error rate, cache-hit
rate, throughput, user feedback and model versions. The Golden Signals
tab is the SRE view of the same store; this tab is the flat evidence
that each required metric is really being written.
"""
from __future__ import annotations

from typing import Any


def render(logger: Any) -> None:
    """Draw the Metrics tab. Requires Streamlit + pandas."""
    import pandas as pd
    import streamlit as st

    from llmops.insights import feedback_summary
    from src import config
    from ui import theme

    theme.page_header(
        "Operations", "Metrics",
        "Every metric the assignment requires, straight from the SQLite "
        "store with no smoothing or aggregation.",
    )
    st.caption(
        f"{config.PRODUCT_NAME} · persisted in {config.METRICS_DB_PATH} · "
        "re-run a journey and refresh to see new rows"
    )

    summary = logger.summary()
    rows = logger.recent(limit=500)

    if summary["requests"] == 0:
        st.info(
            "No metrics yet — run an incident through the Chat page, or "
            "`python scripts/smoke_test.py` first."
        )
        return

    # ------------------------------------------------ headline metrics
    c = st.columns(7)
    c[0].metric("Requests", summary["requests"])
    c[1].metric("Latency p50", f"{summary['latency_p50_ms'] / 1000:.2f}s")
    c[2].metric("Latency p95", f"{summary['latency_p95_ms'] / 1000:.2f}s",
                help="Includes first-run model downloads (cold start).")
    c[3].metric("Latency p99", f"{summary['latency_p99_ms'] / 1000:.2f}s")
    c[4].metric("Tokens in/out",
                f"{summary['tokens_in']}/{summary['tokens_out']}")
    c[5].metric("Est. cost", f"${summary['cost_usd']:.4f}",
                help="Groq calls priced per PRICE_PER_MTOK; local models $0.")
    c[6].metric("Error rate", f"{summary['error_rate'] * 100:.1f}%")

    fb = feedback_summary(rows)
    st.caption(
        f"Throughput: {summary['throughput_rpm']} req/min · "
        f"Cache-hit rate: {summary['cache_hit_rate'] * 100:.1f}% · "
        f"User feedback: 👍 {fb['up']} / 👎 {fb['down']}"
    )

    st.divider()
    left, right = st.columns([3, 2])

    # ------------------------------------------------ per-subtask table
    with left:
        st.subheader("Per-sub-task breakdown")
        per = pd.DataFrame(logger.summary_by_subtask())
        st.dataframe(per, use_container_width=True, hide_index=True)

    # ------------------------------------------------ latency over time
    with right:
        st.subheader("Latency over time (last 500)")
        df = pd.DataFrame(rows)
        df = df[df["subtask"] != "feedback"]
        if df.empty:
            st.caption("No non-feedback rows to plot yet.")
        else:
            df["time"] = pd.to_datetime(df["ts"], unit="s")
            chart = (
                df.sort_values("time")[["time", "subtask", "latency_ms"]]
                .pivot_table(index="time", columns="subtask",
                             values="latency_ms")
            )
            st.line_chart(chart, height=280)

    st.divider()

    # ------------------------------------------------ quality signals
    st.subheader("Model-quality signals recorded")
    st.caption(
        "Written by BaseSubTask.report_signals() into the `extra` column. "
        "Latency and cost say how cheap an answer was; these say whether "
        "it was any good."
    )
    signals = logger.signal_summary()
    if signals:
        st.dataframe(pd.DataFrame(signals), use_container_width=True,
                     hide_index=True)
    else:
        st.info(
            "No quality signals yet. Run a full incident journey "
            "(dictate → classify → retrieve → resolve → summarise → speak)."
        )

    st.divider()

    # ------------------------------------------------ raw rows
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
