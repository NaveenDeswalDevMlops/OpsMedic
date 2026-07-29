# pages/2_Finetune_Comparison.py
"""Fine-tuning evidence page: base vs fine-tuned DistilBERT.

Renders finetune/artifacts/.../compare.json and training_log.json:
headline metrics with delta, per-class F1, side-by-side sample
predictions, and the training curve - the assignment's before/after
demonstration, live inside the app.
"""
from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

from finetune.data import default_paths
from llmops.insights import headline_delta, per_class_table, training_curves
from src import config
from ui import nav, theme

st.set_page_config(page_title="Fine-tune Comparison", page_icon="🎯", layout="wide")
theme.inject_css()
nav.render_page_chrome(active_label="🎯  Fine-tune")
st.title("🎯 Fine-tuning: base vs fine-tuned")
st.caption(
    f"Model: {config.CLASSIFIER_BASE_MODEL} → fine-tuned on "
    "Tobi-Bueck/customer-support-tickets (DOI 10.57967/hf/6184), "
    "queue classification, 10 classes"
)

paths = default_paths()
if not os.path.isfile(paths["compare_json"]):
    st.warning(
        "No comparison found. Run:\n\n"
        "```\npython finetune/train.py\npython finetune/compare.py\n```"
    )
    st.stop()

with open(paths["compare_json"], "r", encoding="utf-8") as fh:
    compare = json.load(fh)

# ------------------------------------------------------------ headline
st.subheader("Held-out test metrics")
head = pd.DataFrame(headline_delta(compare))
c1, c2 = st.columns([2, 3])
with c1:
    st.dataframe(head, use_container_width=True, hide_index=True)
    st.caption(
        f"Evaluated on {compare['finetuned']['eval_rows']} held-out rows. "
        "'base' = pre-trained DistilBERT with an untrained classification "
        "head (no task knowledge); 'fine-tuned' = after training on the "
        "ticket dataset."
    )
with c2:
    melted = head.melt(id_vars="metric", value_vars=["base", "finetuned"])
    chart = melted.pivot_table(index="metric", columns="variable",
                               values="value")
    st.bar_chart(chart, height=220)

# ------------------------------------------------------------ per-class
st.subheader("Per-class F1 (fine-tuned)")
pc = pd.DataFrame(per_class_table(compare, "finetuned"))
if not pc.empty:
    st.bar_chart(pc.set_index("class")["f1"], height=260)
    st.dataframe(pc, use_container_width=True, hide_index=True)
    st.caption(
        "Distinct queues (Billing, Sales, Outages...) score well; residual "
        "confusion concentrates among semantically overlapping queues "
        "(IT/Technical/Product Support) — see samples below."
    )

# ------------------------------------------------------------ samples
st.subheader("Sample predictions (same tickets, both variants)")
samples = pd.DataFrame(compare.get("samples", []))
if not samples.empty:
    def _mark(row: pd.Series, col: str) -> str:
        return ("✅ " if row[col] == row["true"] else "❌ ") + str(row[col])

    view = samples.copy()
    view["base_pred"] = view.apply(lambda r: _mark(r, "base_pred"), axis=1)
    view["finetuned_pred"] = view.apply(
        lambda r: _mark(r, "finetuned_pred"), axis=1
    )
    st.dataframe(view, use_container_width=True, hide_index=True)

# ------------------------------------------------------------ curves
st.subheader("Training curve")
if os.path.isfile(paths["training_log"]):
    with open(paths["training_log"], "r", encoding="utf-8") as fh:
        tlog = json.load(fh)
    curves = training_curves(tlog.get("log_history", []))
    cc1, cc2 = st.columns(2)
    if curves["train"]:
        tdf = pd.DataFrame(curves["train"], columns=["epoch", "train_loss"])
        cc1.line_chart(tdf.set_index("epoch"), height=240)
        cc1.caption("Training loss")
    if curves["eval"]:
        edf = pd.DataFrame(
            curves["eval"],
            columns=["epoch", "accuracy", "f1_macro", "eval_loss"],
        )
        cc2.line_chart(edf.set_index("epoch")[["accuracy", "f1_macro"]],
                       height=240)
        cc2.caption("Held-out accuracy / macro-F1 per epoch")
    with st.expander("Hyperparameters & reproducibility"):
        st.json(tlog.get("hyperparameters", {}))
        st.caption(
            f"Train time: {tlog.get('train_seconds', 0) / 60:.1f} min · "
            "Seeded runs reproduce identical eval losses."
        )
else:
    st.info("training_log.json not found — retrain to regenerate curves.")
