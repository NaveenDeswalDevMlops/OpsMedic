# ui/golden_signals.py
"""Grafana-style "Golden Signals" panel for the Monitor page.

Google's SRE book defines four golden signals for any serving system:
latency, traffic, errors and saturation. This module renders those four
plus the model-quality signals that a pure infrastructure dashboard
would miss - retrieval confidence, classifier confidence, ASR real-time
factor, TTS degradation - because for an inference pipeline "fast and
up" is not the same as "answering correctly".

Every number here is read from the metrics SQLite store written by
llmops.metrics; nothing is simulated. The pure data-shaping helpers at
the top are unit-tested offline in tests/test_golden_signals_offline.py
so the layout code below is the only part that needs a browser.
"""
from __future__ import annotations

import time
from typing import Any

# ---------------------------------------------------------------- pure helpers
#: Human labels and formatting for the quality signals surfaced by
#: MetricsLogger.signal_summary(). Keys must match SIGNAL_FIELDS.
SIGNAL_LABELS: dict[str, tuple[str, str, str]] = {
    # key: (display label, unit/format, one-line interpretation)
    "retrieval_top_score": ("Retrieval top score", "{:.3f}",
                            "Cosine similarity of the best matching ticket"),
    "retrieval_mean_score": ("Retrieval mean score", "{:.3f}",
                             "Average similarity across kept evidence"),
    "evidence_count": ("Evidence per query", "{:.2f}",
                       "Tickets that cleared the similarity threshold"),
    "no_evidence": ("No-evidence rate", "{:.1%}",
                    "Queries where nothing cleared the threshold"),
    "classifier_confidence": ("Classifier confidence", "{:.3f}",
                              "Mean softmax score of the predicted queue"),
    "low_confidence": ("Low-confidence rate", "{:.1%}",
                       "Predictions below the escalation floor"),
    "asr_audio_s": ("Audio duration", "{:.2f} s",
                    "Mean length of dictated incidents"),
    "asr_rtf": ("ASR real-time factor", "{:.2f}x",
                "Decode time / audio duration. Under 1.0 is real time"),
    "tts_fallback": ("TTS fallback rate", "{:.1%}",
                     "Calls that degraded away from real synthesis"),
    "compression_ratio": ("Summary compression", "{:.2f}",
                          "Summary length / source length. Lower is tighter"),
    "ttft_ms": ("Time to first token", "{:.0f} ms",
                "Perceived responsiveness of streamed generation"),
    "tokens_per_sec": ("Throughput", "{:.1f} tok/s",
                       "Decode speed once streaming has started"),
}

#: Signals where a HIGHER number is worse, so the badge colour inverts.
LOWER_IS_BETTER: frozenset[str] = frozenset({
    "no_evidence", "low_confidence", "asr_rtf", "tts_fallback",
    "compression_ratio", "ttft_ms",
})


def format_signal(key: str, value: float) -> str:
    """Render a signal value using its declared format string."""
    spec = SIGNAL_LABELS.get(key)
    if spec is None:
        return f"{value:.3f}"
    try:
        return spec[1].format(value)
    except (ValueError, TypeError):
        return str(value)


def signal_label(key: str) -> str:
    spec = SIGNAL_LABELS.get(key)
    return spec[0] if spec else key.replace("_", " ").capitalize()


def signal_help(key: str) -> str:
    spec = SIGNAL_LABELS.get(key)
    return spec[2] if spec else ""


def build_heatmap(
    series: list[dict[str, Any]], value_key: str = "latency_p95_ms"
) -> tuple[list[str], list[str], list[list[float | None]]]:
    """Reshape timeseries rows into (x_labels, y_labels, z_matrix).

    x is the time bucket, y is the sub-task, z is the chosen metric.
    Buckets with no traffic for a sub-task become None so Plotly leaves
    the cell blank instead of drawing a misleading zero.
    """
    if not series:
        return [], [], []
    buckets = sorted({r["bucket_ts"] for r in series})
    subtasks = sorted({r["subtask"] for r in series})
    lookup = {(r["bucket_ts"], r["subtask"]): r.get(value_key) for r in series}
    x_labels = [time.strftime("%H:%M", time.localtime(b)) for b in buckets]
    z: list[list[float | None]] = [
        [lookup.get((b, s)) for b in buckets] for s in subtasks
    ]
    return x_labels, subtasks, z


def traffic_by_subtask(
    series: list[dict[str, Any]],
) -> tuple[list[str], dict[str, list[int]]]:
    """Reshape timeseries rows into stacked-area traffic series.

    Returns (x_labels, {subtask: [count per bucket]}) with zero-filled
    gaps, because a stacked area chart needs an equal-length series per
    band or the bands misalign.
    """
    if not series:
        return [], {}
    buckets = sorted({r["bucket_ts"] for r in series})
    subtasks = sorted({r["subtask"] for r in series})
    lookup = {(r["bucket_ts"], r["subtask"]): r["requests"] for r in series}
    x_labels = [time.strftime("%H:%M", time.localtime(b)) for b in buckets]
    out = {s: [int(lookup.get((b, s), 0)) for b in buckets] for s in subtasks}
    return x_labels, out


def budget_state(burn_pct: float) -> str:
    """Badge state for the error-budget gauge."""
    if burn_pct <= 50:
        return "ok"
    if burn_pct <= 100:
        return "warn"
    return "err"


def humanise_ms(ms: float) -> str:
    """Compact latency label: 842 ms, 4.2 s, 2.9 min."""
    if ms < 1000:
        return f"{ms:.0f} ms"
    if ms < 60_000:
        return f"{ms / 1000:.1f} s"
    return f"{ms / 60_000:.1f} min"


# ---------------------------------------------------------------- render
def render(metrics: Any, window_seconds: float = 3600.0,
           bucket_seconds: float = 60.0, slo: float = 0.99) -> None:
    """Draw the Golden Signals panel. Requires Streamlit + Plotly."""
    import plotly.graph_objects as go
    import streamlit as st

    from ui import theme

    theme.page_header(
        "Operations", "Golden Signals",
        "Latency, traffic, errors and saturation - the four SRE signals - "
        "plus the model-quality signals a pure infrastructure dashboard "
        "would miss.",
    )

    col_w, col_b, col_s = st.columns([2, 2, 3])
    with col_w:
        window_label = st.selectbox(
            "Window", ["Last 15 min", "Last hour", "Last 6 hours",
                       "Last 24 hours", "All time"], index=1,
        )
    windows = {"Last 15 min": 900.0, "Last hour": 3600.0,
               "Last 6 hours": 21600.0, "Last 24 hours": 86400.0,
               "All time": None}
    window_seconds = windows[window_label]
    with col_b:
        bucket_label = st.selectbox("Bucket", ["1 min", "5 min", "15 min"],
                                    index=1)
    bucket_seconds = {"1 min": 60.0, "5 min": 300.0,
                      "15 min": 900.0}[bucket_label]
    with col_s:
        slo = st.slider("Success-rate SLO", 0.90, 1.00, 0.99, 0.01,
                        help="Error budget = 1 - SLO. Burn above 100% "
                             "means the objective is breached.")

    summary = metrics.summary(window_seconds=window_seconds)
    budget = metrics.error_budget(slo=slo, window_seconds=window_seconds)
    series = metrics.timeseries(bucket_seconds=bucket_seconds,
                               window_seconds=window_seconds)
    signals = metrics.signal_summary(window_seconds=window_seconds)

    if summary["requests"] == 0:
        st.info(
            "No metered requests in this window. Run an incident through "
            "the chat page, or widen the window to 'All time'."
        )
        return

    # ---- the four golden signals as KPI tiles ----
    st.markdown("#### The four golden signals")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        theme.tile("⏱", "LATENCY p95", humanise_ms(summary["latency_p95_ms"]),
                   f"p50 {humanise_ms(summary['latency_p50_ms'])} · "
                   f"p99 {humanise_ms(summary['latency_p99_ms'])}",
                   theme.status_badge(
                       "slow" if summary["latency_p95_ms"] > 8000 else "ok",
                       "err" if summary["latency_p95_ms"] > 8000 else "ok"))
    with k2:
        theme.tile("📈", "TRAFFIC", f"{summary['throughput_rpm']:.1f} rpm",
                   f"{summary['requests']} requests in window")
    with k3:
        theme.tile("🚨", "ERRORS", f"{summary['error_rate']:.1%}",
                   f"{budget['failures']} of {budget['requests']} failed",
                   theme.status_badge(
                       f"{budget['burn_pct']:.0f}% budget",
                       budget_state(budget["burn_pct"])))
    with k4:
        theme.tile("💾", "SATURATION", f"{summary['cache_hit_rate']:.1%}",
                   f"cache hit rate · ${summary['cost_usd']:.5f} spend")

    st.divider()

    # ---- traffic (stacked area) ----
    left, right = st.columns([3, 2])
    with left:
        st.markdown("##### Traffic by sub-task")
        x_labels, bands = traffic_by_subtask(series)
        if bands:
            fig = go.Figure()
            for name, values in bands.items():
                fig.add_trace(go.Scatter(
                    x=x_labels, y=values, name=name, mode="lines",
                    stackgroup="one", line=dict(width=1.5),
                ))
            fig.update_layout(height=280, hovermode="x unified")
            theme.style_plotly(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Not enough buckets to plot.")

    # ---- error budget gauge ----
    with right:
        st.markdown("##### Error budget burn")
        gcolor = {"ok": theme.OK, "warn": theme.WARN,
                  "err": theme.ERR}[budget_state(budget["burn_pct"])]
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=min(budget["burn_pct"], 200.0),
            number={"suffix": "%", "font": {"size": 30}},
            gauge={
                "axis": {"range": [0, 200], "tickwidth": 1},
                "bar": {"color": gcolor, "thickness": 0.7},
                "steps": [
                    {"range": [0, 50], "color": "#DCFCE7"},
                    {"range": [50, 100], "color": "#FEF3C7"},
                    {"range": [100, 200], "color": "#FEE2E2"},
                ],
                "threshold": {"line": {"color": theme.ERR, "width": 3},
                              "thickness": 0.9, "value": 100},
            },
        ))
        gauge.update_layout(height=280, margin=dict(l=20, r=20, t=10, b=10),
                            paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(gauge, use_container_width=True)
        st.caption(
            f"SLO {slo:.0%} allows {budget['budget_pct']:.1f}% failures. "
            f"Observed success {budget['success_rate']:.2%}. "
            + ("**Breached.**" if budget["breached"] else "Within budget.")
        )

    st.divider()

    # ---- latency heatmap ----
    st.markdown("##### p95 latency by sub-task over time")
    st.caption(
        "Blank cells mean that sub-task received no traffic in that bucket. "
        "The first call of a session includes lazy model-load cost."
    )
    hx, hy, hz = build_heatmap(series, "latency_p95_ms")
    if hz:
        heat = go.Figure(go.Heatmap(
            x=hx, y=hy, z=hz, colorscale="YlOrRd", hoverongaps=False,
            colorbar=dict(title="ms"),
        ))
        heat.update_layout(height=260)
        theme.style_plotly(heat)
        st.plotly_chart(heat, use_container_width=True)
    else:
        st.caption("Not enough data for a heatmap.")

    st.divider()

    # ---- model-quality signals ----
    st.markdown("#### Model-quality signals")
    st.caption(
        "Latency and cost say how cheap an answer was. These say whether "
        "it was any good. Signals with no data in this window are omitted "
        "rather than shown as zero."
    )
    if not signals:
        st.info(
            "No quality signals recorded yet. These are written by "
            "BaseSubTask.report_signals(); run a full incident journey to "
            "populate them."
        )
        return

    for row in range(0, len(signals), 4):
        cols = st.columns(4)
        for col, sig in zip(cols, signals[row:row + 4]):
            key, value = sig["signal"], sig["value"]
            with col:
                inverted = key in LOWER_IS_BETTER
                state = theme.rate_state(
                    value if key in ("no_evidence", "low_confidence",
                                     "tts_fallback") else 0.0,
                    good_high=False,
                ) if inverted else "ok"
                theme.tile(
                    "🎯", signal_label(key).upper(),
                    format_signal(key, value),
                    f"n={sig['n']} · {sig['subtasks']}",
                    theme.status_badge(
                        "lower is better" if inverted else "higher is better",
                        state if inverted else "ok",
                    ),
                )
                st.caption(signal_help(key))
