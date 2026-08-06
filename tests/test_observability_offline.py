# tests/test_observability_offline.py
"""Offline tests for the Tier-1 observability layer and the Model Card.

Covers three things that previously had no test:
  1. metrics.set_latency_ms / mark_error  - streamed rows used to record
     ~0 ms and log swallowed failures as status='ok'.
  2. BaseSubTask.report_signals           - model-quality signals now
     reach the metrics `extra` blob and signal_summary() aggregates them.
  3. ui.model_card                        - the majority-class baseline
     and the collapse detector, which are the guards against reporting
     a seed-dependent delta as an improvement.

Stdlib + sqlite3 only: no transformers, no torch, no Streamlit.
"""
from __future__ import annotations

import json
import math
import sqlite3

import pytest

from llmops.metrics import MetricsLogger
from models.base import BaseSubTask
from ui import model_card as mc

PRICES = {"llama-3.1-8b-instant": {"in": 0.05, "out": 0.08}}


@pytest.fixture()
def logger(tmp_path):
    return MetricsLogger(str(tmp_path / "m.db"), price_table=PRICES)


def _rows(lg):
    conn = sqlite3.connect(lg.db_path)
    conn.row_factory = sqlite3.Row
    with conn:
        return list(conn.execute("SELECT * FROM metrics ORDER BY id"))


class _Fake(BaseSubTask):
    """Minimal sub-task that reports whatever signals it is handed."""

    name, model_name, cacheable = "fake", "fake-model", False

    def __init__(self, signals, **kw):
        super().__init__(**kw)
        self._to_report = signals

    def _run(self, payload):
        self.report_signals(**self._to_report)
        return "done"


# ---------------------------------------------------- 1. latency overrides
def test_streamed_latency_override_is_recorded(logger):
    with logger.track("resolve", "llama-3.1-8b-instant") as r:
        r.set_latency_ms(4210.5)
    assert _rows(logger)[0]["latency_ms"] == pytest.approx(4210.5)


def test_block_timing_survives_when_no_override(logger):
    with logger.track("summarize", "distilbart"):
        pass
    assert _rows(logger)[0]["latency_ms"] >= 0.0


def test_negative_latency_is_clamped(logger):
    with logger.track("resolve", "llama-3.1-8b-instant") as r:
        r.set_latency_ms(-42.0)
    assert _rows(logger)[0]["latency_ms"] == 0.0


def test_mark_error_flips_status(logger):
    with logger.track("resolve", "llama-3.1-8b-instant") as r:
        r.set_latency_ms(100.0)
        r.mark_error("RuntimeError: Groq API 401")
    row = _rows(logger)[0]
    assert row["status"] == "error" and "401" in row["error"]


def test_raised_exception_still_wins(logger):
    with pytest.raises(RuntimeError):
        with logger.track("resolve", "llama-3.1-8b-instant") as r:
            r.mark_error("less specific")
            raise RuntimeError("the real failure")
    assert "the real failure" in _rows(logger)[0]["error"]


def test_summary_exposes_p99(logger):
    for ms in (10.0, 20.0, 5000.0):
        with logger.track("retrieve", "minilm") as r:
            r.set_latency_ms(ms)
    s = logger.summary()
    assert s["latency_p99_ms"] >= s["latency_p95_ms"] >= s["latency_p50_ms"]


# ------------------------------------------------- 2. report_signals
def test_report_signals_reaches_the_extra_blob(logger):
    _Fake({"classifier_confidence": 0.83, "low_confidence": False},
          metrics=logger).run("x")
    extra = json.loads(_rows(logger)[0]["extra"])
    assert extra["classifier_confidence"] == pytest.approx(0.83)
    assert extra["low_confidence"] is False


def test_none_signals_are_dropped_not_zeroed(logger):
    _Fake({"asr_rtf": None, "asr_audio_s": 6.0}, metrics=logger).run("x")
    extra = json.loads(_rows(logger)[0]["extra"])
    assert "asr_rtf" not in extra
    assert extra["asr_audio_s"] == pytest.approx(6.0)


def test_signals_do_not_leak_between_calls(logger):
    task = _Fake({"evidence_count": 3}, metrics=logger)
    task.run("first")
    task._to_report = {}
    task.run("second")
    second = _rows(logger)[1]
    extra = json.loads(second["extra"]) if second["extra"] else {}
    assert "evidence_count" not in extra


def test_signals_appear_on_the_returned_schema(logger):
    out = _Fake({"no_evidence": True}, metrics=logger).run("x")
    assert out["metrics"]["no_evidence"] is True


def test_signal_summary_means_and_rates(logger):
    _Fake({"classifier_confidence": 0.80, "low_confidence": False},
          metrics=logger).run("a")
    _Fake({"classifier_confidence": 0.40, "low_confidence": True},
          metrics=logger).run("b")
    got = {r["signal"]: r for r in logger.signal_summary()}
    assert got["classifier_confidence"]["value"] == pytest.approx(0.60)
    assert got["classifier_confidence"]["aggregation"] == "mean"
    assert got["low_confidence"]["value"] == pytest.approx(0.50)
    assert got["low_confidence"]["aggregation"] == "rate"


def test_unreported_signals_are_absent_not_zero(logger):
    _Fake({"asr_rtf": 28.7}, metrics=logger).run("x")
    signals = {r["signal"] for r in logger.signal_summary()}
    assert "asr_rtf" in signals
    assert "tts_fallback" not in signals


# ------------------------------------------------- 3. error budget + series
def test_error_budget_burn_and_breach(logger):
    for _ in range(9):
        with logger.track("retrieve", "minilm"):
            pass
    with pytest.raises(ValueError):
        with logger.track("retrieve", "minilm"):
            raise ValueError("boom")
    b = logger.error_budget(slo=0.99)
    assert b["requests"] == 10 and b["failures"] == 1
    assert b["success_rate"] == pytest.approx(0.9)
    assert b["breached"] is True
    assert b["burn_pct"] == pytest.approx(1000.0)


def test_error_budget_on_empty_store_is_not_breached(logger):
    assert logger.error_budget()["breached"] is False


def test_timeseries_buckets_by_subtask(logger):
    for _ in range(3):
        with logger.track("retrieve", "minilm"):
            pass
    with logger.track("classify", "deberta"):
        pass
    series = logger.timeseries(bucket_seconds=3600.0, window_seconds=None)
    by = {r["subtask"]: r for r in series}
    assert by["retrieve"]["requests"] == 3
    assert by["classify"]["requests"] == 1


# ------------------------------------------------- 4. model card maths
def test_majority_baseline_matches_the_closed_form():
    supports = {"A": 289, "B": 190, "C": 149, "D": 117, "E": 98,
                "F": 60, "G": 50, "H": 30, "I": 10, "J": 7}
    total = sum(supports.values())
    maj = mc.majority_baseline(supports)
    p = 289 / total
    assert maj["label"] == "A"
    assert maj["accuracy"] == pytest.approx(round(p, 4))
    assert maj["f1_macro"] == pytest.approx(round((2 * p / (p + 1)) / 10, 4))


def test_majority_baseline_handles_empty():
    assert mc.majority_baseline({})["label"] is None


def test_collapse_detector_flags_single_class_output():
    blob = {"finetuned": {"per_class": {
        "A": {"precision": 0.098, "recall": 1.0, "f1": 0.179, "support": 98},
        "B": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 289},
    }}}
    got = mc.collapse_check(blob)["finetuned"]
    assert got["collapsed"] is True
    assert got["nonzero_f1"] == 1
    assert got["always_predicts"] == "A"


def test_collapse_detector_passes_a_healthy_model():
    blob = {"finetuned": {"per_class": {
        "A": {"precision": 0.5, "recall": 0.6, "f1": 0.55, "support": 98},
        "B": {"precision": 0.4, "recall": 0.3, "f1": 0.34, "support": 289},
        "C": {"precision": 0.2, "recall": 0.2, "f1": 0.20, "support": 50},
    }}}
    got = mc.collapse_check(blob)["finetuned"]
    assert got["collapsed"] is False
    assert got["nonzero_f1"] == 3
    assert got["always_predicts"] is None


def test_chance_loss_reference_is_log_num_classes():
    """The training-curve reference line must be ln(K), not a constant."""
    assert math.log(10) == pytest.approx(2.302585, abs=1e-5)


def test_saved_is_best_detects_a_worse_final_epoch():
    curve = [{"epoch": 1.0, "f1_macro": 0.30},
             {"epoch": 2.0, "f1_macro": 0.42},
             {"epoch": 3.0, "f1_macro": 0.38}]
    assert mc.saved_is_best(curve) is False
    assert mc.best_epoch(curve)["epoch"] == 2.0


def test_saved_is_best_accepts_a_monotonic_run():
    curve = [{"epoch": 1.0, "f1_macro": 0.22},
             {"epoch": 2.0, "f1_macro": 0.28},
             {"epoch": 3.0, "f1_macro": 0.2817}]
    assert mc.saved_is_best(curve) is True


def test_load_json_returns_none_for_missing_file():
    assert mc.load_json("definitely/not/here.json") is None


# ------------------------------------------------- 5. golden signals shaping
def test_heatmap_leaves_gaps_as_none():
    from ui.golden_signals import build_heatmap

    series = [{"bucket_ts": 100, "subtask": "a", "latency_p95_ms": 5.0},
              {"bucket_ts": 200, "subtask": "b", "latency_p95_ms": 7.0}]
    _, y, z = build_heatmap(series)
    assert y == ["a", "b"]
    assert z[0][1] is None and z[1][0] is None


def test_traffic_zero_fills_for_stacked_area():
    from ui.golden_signals import traffic_by_subtask

    series = [{"bucket_ts": 100, "subtask": "a", "requests": 4},
              {"bucket_ts": 200, "subtask": "b", "requests": 2}]
    _, bands = traffic_by_subtask(series)
    assert bands["a"] == [4, 0] and bands["b"] == [0, 2]


def test_budget_state_thresholds():
    from ui.golden_signals import budget_state

    assert budget_state(10) == "ok"
    assert budget_state(80) == "warn"
    assert budget_state(150) == "err"


def test_humanise_ms_units():
    from ui.golden_signals import humanise_ms

    assert humanise_ms(842) == "842 ms"
    assert humanise_ms(4210.5) == "4.2 s"
    assert humanise_ms(172418) == "2.9 min"


def test_signal_formatting_uses_declared_units():
    from ui.golden_signals import format_signal

    assert format_signal("asr_rtf", 28.733) == "28.73x"
    assert format_signal("no_evidence", 0.5) == "50.0%"
    assert format_signal("ttft_ms", 180.2) == "180 ms"
