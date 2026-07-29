# tests/test_subtasks_offline.py
"""Offline tests for Phase-2B pure logic: retrieval filtering, resolution
prompt building + SOP linking, classifier label/variant resolution, and
ASR audio normalization. No model downloads or network required.

Run with pytest:   pytest tests/test_subtasks_offline.py -v
Or stdlib runner:  python tests/test_subtasks_offline.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from models.asr import TARGET_SR, to_mono_16k
from models.classifier import (
    DEFAULT_LABELS,
    artifact_version_stamp,
    pick_variant,
    resolve_labels,
)


def test_artifact_version_stamp():
    import json
    import tempfile

    d = tempfile.mkdtemp()
    assert artifact_version_stamp(d) == ""  # no labels.json yet
    with open(os.path.join(d, "labels.json"), "w") as fh:
        json.dump(["a", "b"], fh)
    stamp = artifact_version_stamp(d)
    assert stamp.startswith("@") and len(stamp) == 16  # @YYYYMMDD-HHMMSS
from models.resolution import SYSTEM_PROMPT, build_resolution_prompt, load_sop
from models.retrieval import filter_hits

META = [
    {"ticket_id": "OPM-00001", "title": "VPN down", "description": "d1" * 200,
     "resolution": "restart vpn", "category": "IT Support", "priority": "high"},
    {"ticket_id": "OPM-00002", "title": "Invoice wrong", "description": "d2",
     "resolution": "correct invoice", "category": "Billing and Payments",
     "priority": "low"},
    {"ticket_id": "OPM-00003", "title": "Printer jam", "description": "d3",
     "resolution": "clear jam", "category": "IT Support", "priority": "med"},
]


# ------------------------------------------------------------- retrieval
def test_filter_hits_threshold_padding_and_snippet():
    hits = filter_hits(
        scores=[0.91, 0.30, -1.0],
        indices=[0, 1, -1],  # -1 = FAISS padding for missing results
        meta_rows=META,
        threshold=0.35,
    )
    assert [h["ticket_id"] for h in hits] == ["OPM-00001"]  # 0.30 filtered
    assert len(hits[0]["description"]) <= 300  # snippet capped
    assert hits[0]["score"] == 0.91
    assert hits[0]["resolution"] == "restart vpn"


def test_filter_hits_keeps_order_and_all_above_threshold():
    hits = filter_hits([0.9, 0.8, 0.7], [2, 0, 1], META, threshold=0.5)
    assert [h["ticket_id"] for h in hits] == ["OPM-00003", "OPM-00001", "OPM-00002"]


# ------------------------------------------------------------ resolution
def test_build_prompt_grounds_on_similar_and_sop():
    similar = [
        {"ticket_id": "OPM-00042", "title": "Portal login fails",
         "description": "users cannot login", "resolution": "reset SSO cache",
         "score": 0.88},
    ]
    msgs = build_resolution_prompt("login portal down", similar, "IT Support")
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    user = msgs[1]["content"]
    assert "NEW INCIDENT:" in user and "login portal down" in user
    assert "[OPM-00042]" in user and "reset SSO cache" in user
    assert "STANDARD OPERATING PROCEDURE (IT Support)" in user
    assert "SOP — IT Support queue" in user  # actual file content included


def test_build_prompt_without_similar_or_category():
    msgs = build_resolution_prompt("mystery issue", [], None)
    user = msgs[1]["content"]
    assert "none above threshold" in user
    assert "STANDARD OPERATING PROCEDURE" not in user


def test_load_sop_missing_category_is_empty():
    assert load_sop(None) == ""
    assert load_sop("No Such Queue Ever") == ""
    assert "Triage" in load_sop("Technical Support")


# ------------------------------------------------------------ classifier
def test_pick_variant_rules():
    assert pick_variant("base") == "base"
    assert pick_variant("finetuned") == "finetuned"
    # 'auto' must match the actual artifact state of THIS environment:
    # 'finetuned' when the trained model exists, 'base' otherwise.
    from src import config as _cfg

    has_artifact = os.path.isfile(
        os.path.join(_cfg.CLASSIFIER_FINETUNED_DIR, "config.json")
    )
    assert pick_variant("auto") == ("finetuned" if has_artifact else "base")
    try:
        pick_variant("nope")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("invalid variant must raise")


def test_resolve_labels_sorted_and_nonempty():
    labels = resolve_labels()
    assert labels == sorted(labels)
    assert len(labels) >= 2
    # in this offline env there is no artifact/train csv -> defaults
    if not os.path.isfile("./data/finetune_train.csv"):
        assert labels == sorted(DEFAULT_LABELS)


# ------------------------------------------------------------------- asr
def test_to_mono_16k_downmixes_stereo():
    stereo = np.stack([np.ones(1600), np.zeros(1600)], axis=1)  # (n, 2)
    mono = to_mono_16k(stereo, TARGET_SR)
    assert mono.ndim == 1 and mono.shape[0] == 1600
    assert np.allclose(mono, 0.5)
    assert mono.dtype == np.float32


def test_to_mono_16k_passthrough_at_target_rate():
    x = np.random.RandomState(0).randn(3200).astype(np.float64)
    out = to_mono_16k(x, TARGET_SR)  # no librosa path needed
    assert out.shape == x.shape and out.dtype == np.float32


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
