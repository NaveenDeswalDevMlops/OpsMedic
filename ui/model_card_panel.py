# ui/model_card_panel.py
"""Fine-tune tab for the Monitor page (was pages/2_Model_Card.py).

Moved out of pages/ so it renders as a tab under Monitor instead of a
separate sidebar entry. The data preparation lives in ui/model_card.py,
which stays free of Streamlit so its arithmetic - the majority-class
baseline and the collapse detector - can be unit-tested offline.

This tab answers "can I trust this model?". Nothing here is live
traffic: it reads the artifacts written by finetune/train.py and
finetune/compare.py.
"""
from __future__ import annotations

import math
import os


def render() -> None:
    """Draw the Fine-tune / Model Card tab. Requires Streamlit + Plotly."""
    import pandas as pd
    import plotly.graph_objects as go
    import streamlit as st

    from src import config
    from ui import model_card as mc
    from ui import theme

    ARTIFACT_DIR = config.CLASSIFIER_FINETUNED_DIR
    CMP_PATH = os.path.join(ARTIFACT_DIR, "compare.json")
    LOG_PATH = os.path.join(ARTIFACT_DIR, "training_log.json")

    theme.page_header(
        "Evaluation", "Model Card",
        "Sub-task 2: ticket queue classification. Everything below is read "
        "from the training and comparison artifacts on disk — no live "
        "traffic, no simulated numbers.",
    )

    cmp_json = mc.load_json(CMP_PATH)
    log_json = mc.load_json(LOG_PATH)

    if cmp_json is None:
        st.warning(
            f"No comparison artifact at `{CMP_PATH}`.\n\n"
            "Run the fine-tune and comparison first:\n\n"
            "```\n"
            "python finetune/train.py --max-rows 0 --epochs 3 --balanced \\\n"
            "    --lr 2e-5 --batch 16\n"
            "python finetune/compare.py --max-rows 0\n"
            "```"
        )
        return

    supports = mc.supports_from_compare(cmp_json)
    maj = mc.majority_baseline(supports)
    collapse = mc.collapse_check(cmp_json)
    ft = cmp_json.get("finetuned", {})
    base = cmp_json.get("base", {})
    eval_rows = ft.get("eval_rows") or sum(supports.values())

    # ------------------------------------------------------------------ headline
    st.markdown("### 1 · Headline result")

    ft_acc = float(ft.get("accuracy", 0.0))
    ft_f1 = float(ft.get("f1_macro", 0.0))
    beats_majority = ft_f1 > maj["f1_macro"] and ft_acc > maj["accuracy"]
    ft_collapsed = collapse.get("finetuned", {}).get("collapsed")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        theme.tile("🎯", "MACRO F1", f"{ft_f1:.4f}",
                   f"vs {maj['f1_macro']:.4f} majority floor",
                   theme.status_badge(
                       f"{ft_f1 / maj['f1_macro']:.1f}x floor"
                       if maj["f1_macro"] else "n/a",
                       "ok" if beats_majority else "err"))
    with k2:
        theme.tile("✅", "ACCURACY", f"{ft_acc:.4f}",
                   f"vs {maj['accuracy']:.4f} majority floor",
                   theme.status_badge(
                       f"{ft_acc - maj['accuracy']:+.4f}",
                       "ok" if ft_acc > maj["accuracy"] else "err"))
    with k3:
        nz = collapse.get("finetuned", {}).get("nonzero_f1", 0)
        n_cls = collapse.get("finetuned", {}).get("n_classes", 0)
        theme.tile("🧭", "CLASSES PREDICTED", f"{nz} / {n_cls}",
                   "classes with non-zero F1",
                   theme.status_badge(
                       "collapsed" if ft_collapsed else "discriminating",
                       "err" if ft_collapsed else "ok"))
    with k4:
        theme.tile("🧾", "HELD-OUT ROWS", f"{eval_rows:,}",
                   f"{maj['n_classes']} classes · stratified split")

    if ft_collapsed:
        st.error(
            f"**Collapse detected.** The fine-tuned model has non-zero F1 on "
            f"only {nz} of {n_cls} classes and predicts "
            f"`{collapse['finetuned'].get('always_predicts')}` for effectively "
            f"every row. Accuracy equals that class's share of the split, so "
            f"the model has learned nothing. Do not report the delta against "
            f"the random-head base as an improvement — retrain before using "
            f"these numbers."
        )
    elif not beats_majority:
        st.warning(
            "The model does not beat the majority-class floor on both metrics. "
            "It is learning something, but not enough to be useful yet."
        )
    else:
        st.success(
            f"The model discriminates across {nz} of {n_cls} classes and beats "
            f"the majority-class floor on both metrics. Macro F1 is "
            f"{ft_f1 / maj['f1_macro']:.1f}x the collapse floor."
        )

    # --------------------------------------------------------------- baselines
    st.divider()
    st.markdown("### 2 · Compared against three baselines")
    st.caption(
        "The `base` variant is the pre-trained encoder with a **randomly "
        "initialised** classification head. That head is re-randomised on "
        "every construction, so its score — and therefore the delta against "
        "it — changes between runs. The majority-class floor is seed-independent "
        "and is the comparator to quote."
    )
    bt = pd.DataFrame(mc.baseline_table(cmp_json, supports))
    bt.columns = ["Comparator", "Accuracy", "Macro F1", "Reproducible?"]
    st.dataframe(bt, use_container_width=True, hide_index=True)

    cA, cB = st.columns(2)
    with cA:
        fig = go.Figure()
        rows = mc.baseline_table(cmp_json, supports)
        fig.add_trace(go.Bar(
            x=[r["comparator"].split(" - ")[0].split(" (")[0] for r in rows],
            y=[r["f1_macro"] for r in rows],
            marker_color=[theme.MUTED, theme.WARN, theme.OK],
            text=[f"{r['f1_macro']:.4f}" for r in rows], textposition="outside",
        ))
        fig.update_layout(height=300, title="Macro F1 by comparator",
                          yaxis_title="macro F1")
        theme.style_plotly(fig)
        st.plotly_chart(fig, use_container_width=True)
    with cB:
        st.markdown("**Why macro F1 is the headline here**")
        st.markdown(
            "- Training used `--balanced` (median downsampling), so the model "
            "deliberately forgoes the majority-class prior.\n"
            "- Evaluation uses the **true** class distribution.\n"
            "- Accuracy therefore understates the model, while macro F1 "
            "weights all ten queues equally.\n"
            f"- A model that always answered '{maj['label']}' would score "
            f"{maj['accuracy']:.3f} accuracy but only {maj['f1_macro']:.4f} "
            "macro F1. That gap is the collapse detector."
        )

    # --------------------------------------------------------------- per class
    st.divider()
    st.markdown("### 3 · Per-class performance")
    pc = mc.per_class_rows(cmp_json)
    if pc:
        df = pd.DataFrame(pc)
        df.columns = ["Class", "Support", "Precision", "Recall", "F1",
                      "Base F1", "Δ F1"]
        st.dataframe(df, use_container_width=True, hide_index=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=[r["class"] for r in pc],
                             y=[r["f1"] for r in pc],
                             name="Fine-tuned F1", marker_color=theme.OK))
        fig.add_trace(go.Bar(x=[r["class"] for r in pc],
                             y=[r["base_f1"] for r in pc],
                             name="Base F1", marker_color=theme.MUTED))
        fig.add_trace(go.Scatter(
            x=[r["class"] for r in pc],
            y=[r["support"] / max(1, max(x["support"] for x in pc)) for r in pc],
            name="Support (normalised)", yaxis="y2", mode="lines+markers",
            line=dict(color=theme.PRIMARY, dash="dot")))
        fig.update_layout(
            height=360, barmode="group", title="Per-class F1 with support overlay",
            yaxis=dict(title="F1"),
            yaxis2=dict(title="support (norm.)", overlaying="y", side="right",
                        range=[0, 1.05], showgrid=False),
        )
        theme.style_plotly(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Rare classes are the hard ones: with median downsampling the "
            "model sees at most 864 examples of the common queues and as few "
            "as 159 of the rarest, so low F1 on the tail is expected."
        )

    # ------------------------------------------------------------ training curve
    st.divider()
    st.markdown("### 4 · Training curve")
    if log_json is None:
        st.info(f"No training log at `{LOG_PATH}`.")
    else:
        curve = mc.training_curve(log_json)
        if curve:
            cdf = pd.DataFrame(curve).drop_duplicates(subset=["epoch"],
                                                      keep="last")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cdf["epoch"], y=cdf["eval_loss"],
                                     name="eval loss", mode="lines+markers",
                                     line=dict(color=theme.ERR)))
            fig.add_trace(go.Scatter(x=cdf["epoch"], y=cdf["accuracy"],
                                     name="accuracy", mode="lines+markers",
                                     yaxis="y2", line=dict(color=theme.PRIMARY)))
            fig.add_trace(go.Scatter(x=cdf["epoch"], y=cdf["f1_macro"],
                                     name="macro F1", mode="lines+markers",
                                     yaxis="y2", line=dict(color=theme.OK)))
            # chance-level reference line: ln(num_classes)
            chance = math.log(max(2, maj["n_classes"]))
            fig.add_hline(y=chance, line_dash="dash", line_color=theme.MUTED,
                          annotation_text=f"chance loss = ln({maj['n_classes']}) "
                                          f"= {chance:.3f}")
            fig.update_layout(height=340, title="Per-epoch evaluation",
                              xaxis_title="epoch",
                              yaxis=dict(title="eval loss"),
                              yaxis2=dict(title="accuracy / macro F1",
                                          overlaying="y", side="right",
                                          range=[0, 1], showgrid=False))
            theme.style_plotly(fig)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "The dashed line is chance-level cross-entropy. An eval loss "
                "sitting on it means the model outputs a near-uniform "
                "distribution and has not learned."
            )

            is_best = mc.saved_is_best(curve)
            best = mc.best_epoch(curve)
            if is_best is False and best:
                st.warning(
                    f"**The saved checkpoint is not the best epoch.** "
                    f"`train.py` does not pass `load_best_model_at_end`, so it "
                    f"saves the *final* epoch. Epoch {best['epoch']:.0f} scored "
                    f"macro F1 {best['f1_macro']:.4f}, better than the final "
                    f"epoch. Re-run with `load_best_model_at_end=True` to keep "
                    f"the stronger checkpoint."
                )
            elif is_best:
                st.success(
                    "The final epoch is also the best epoch, so the saved "
                    "checkpoint is the strongest one seen during training."
                )

        hp = log_json.get("hyperparameters") or {}
        st.markdown("**Hyperparameters actually used** (read from the artifact, "
                    "not from documentation)")
        st.dataframe(
            pd.DataFrame([{"setting": k, "value": str(v)} for k, v in hp.items()]),
            use_container_width=True, hide_index=True,
        )
        secs = log_json.get("train_seconds")
        if secs:
            st.caption(f"Wall-clock training time: {secs / 60:.1f} min "
                       f"on device `{hp.get('device', 'unknown')}`.")

    # ----------------------------------------------------------- error analysis
    st.divider()
    st.markdown("### 5 · Error analysis")
    misses = mc.confusion_pairs(cmp_json)
    st.caption(
        "compare.py stores a sample of predictions rather than the full "
        "prediction vector, so this is illustrative, not a complete confusion "
        "matrix."
    )
    if misses:
        mdf = pd.DataFrame(misses)
        mdf.columns = ["True queue", "Predicted queue", "Ticket text (truncated)"]
        st.dataframe(mdf, use_container_width=True, hide_index=True)
        st.markdown(
            "**Reading these errors.** Several of the ten queues overlap "
            "semantically — Technical Support, IT Support and Product Support "
            "describe adjacent work, and Returns and Exchanges sits inside "
            "Customer Service. Confusions between neighbouring queues are "
            "materially less costly than random misrouting, and a human "
            "annotator would disagree with the gold label on some of these "
            "rows too. This caps the achievable accuracy on this taxonomy."
        )
    else:
        st.success("No misclassifications among the sampled rows.")

    # ------------------------------------------------------ provenance + limits
    st.divider()
    st.markdown("### 6 · Dataset provenance and limitations")
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("**Provenance**")
        st.markdown(
            f"- Dataset: `{config.HF_DATASET}`\n"
            "- DOI: 10.57967/hf/6184 · Licence: CC-BY-NC-4.0\n"
            f"- Base model: `{config.CLASSIFIER_BASE_MODEL}`\n"
            f"- Artifact: `{ARTIFACT_DIR}`\n"
            f"- Held-out rows: {eval_rows:,} · classes: {maj['n_classes']}\n"
            "- Split: stratified, seeded, written by `finetune/data.py`"
        )
    with p2:
        st.markdown("**Stated limitations**")
        st.markdown(
            "- English-only rows; non-English tickets were dropped in prep.\n"
            "- Queue taxonomy is from one organisation and will not transfer.\n"
            "- `--balanced` downsamples common classes, so the model is "
            "calibrated for macro F1 rather than raw accuracy.\n"
            "- The `base` comparator has a random head and is **not** a tuned "
            "competitor; treat it as a floor, not a rival.\n"
            "- Trained on CPU (MPS is refused for DeBERTa-v3 — it produces NaN "
            "gradients on that backend), so epochs are limited by wall clock."
        )

    st.caption(
        "The Golden Signals and Metrics tabs answer 'is it healthy now?'. "
        "This tab answers 'can I trust the model?' — it reads training "
        "artifacts, not live traffic."
    )
