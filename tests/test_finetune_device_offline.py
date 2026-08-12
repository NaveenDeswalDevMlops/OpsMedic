# tests/test_finetune_device_offline.py
"""Offline tests for the device/precision policy and the NaN tripwire.

These encode the failure that cost a 105-minute run: DeBERTa on the MPS
backend produced non-finite gradients, every weight became NaN, and the
model then predicted the alphabetically-first class for all 2060 held-out
rows (accuracy 0.0981 == 202/2060).

No torch, no GPU, no network required - the hardware probe is injected.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finetune.data import class_weights, split_stats
from finetune.device import (
    NonFiniteLoss,
    check_log_row,
    is_finite,
    is_fp16_unsafe,
    is_mps_unsafe,
    resolve_precision,
    resolve_training_device,
)
from finetune.train import PRESETS, choose_preset

APPLE = {"cuda": False, "mps": True, "bf16": False}
T4 = {"cuda": True, "mps": False, "bf16": False}
AMPERE = {"cuda": True, "mps": False, "bf16": True}
PLAIN_CPU = {"cuda": False, "mps": False, "bf16": False}


# ---------------------------------------------------------------- device
def test_deberta_is_never_auto_scheduled_onto_mps():
    device, notes = resolve_training_device(
        "microsoft/deberta-v3-large", "auto", probe=APPLE
    )
    assert device == "cpu"
    assert any("mps" in n.lower() for n in notes)


def test_explicit_mps_request_for_deberta_is_refused_with_an_explanation():
    device, notes = resolve_training_device(
        "microsoft/deberta-v3-base", "mps", probe=APPLE
    )
    assert device == "cpu"
    assert any("REFUSING" in n for n in notes)


def test_force_mps_is_honoured_but_warns():
    device, notes = resolve_training_device(
        "microsoft/deberta-v3-base", "mps", force_mps=True, probe=APPLE
    )
    assert device == "mps"
    assert any("WARNING" in n for n in notes)


def test_mps_safe_architecture_still_gets_the_gpu_on_apple():
    assert resolve_training_device("roberta-large", "auto", probe=APPLE)[0] == "mps"


def test_cuda_wins_over_everything_and_deberta_is_allowed_there():
    assert resolve_training_device(
        "microsoft/deberta-v3-large", "auto", probe=AMPERE
    )[0] == "cuda"
    assert resolve_training_device(
        "microsoft/deberta-v3-large", "auto", probe=T4
    )[0] == "cuda"


def test_requesting_absent_hardware_degrades_to_cpu_not_a_crash():
    assert resolve_training_device("roberta-large", "cuda", probe=APPLE)[0] == "cpu"
    assert resolve_training_device("roberta-large", "mps", probe=PLAIN_CPU)[0] == "cpu"


def test_cpu_is_always_available():
    assert resolve_training_device("anything", "cpu", probe=PLAIN_CPU)[0] == "cpu"
    assert resolve_training_device("anything", "auto", probe=PLAIN_CPU)[0] == "cpu"


def test_unknown_device_name_raises():
    try:
        resolve_training_device("roberta-large", "tpu", probe=PLAIN_CPU)
    except ValueError as exc:
        assert "tpu" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("must reject unknown devices")


def test_unsafe_lists_are_substring_matched_case_insensitively():
    assert is_mps_unsafe("microsoft/DeBERTa-v3-base")
    assert not is_mps_unsafe("roberta-large")
    assert is_fp16_unsafe("microsoft/deberta-v3-large")
    assert not is_fp16_unsafe("microsoft/deberta-v3-base")


# ------------------------------------------------------------- precision
def test_mixed_precision_is_never_used_off_cuda():
    for device in ("cpu", "mps"):
        fp16, bf16, notes = resolve_precision("roberta-large", device, probe=APPLE)
        assert (fp16, bf16) == (False, False)
        assert notes


def test_ampere_prefers_bf16_so_no_loss_scaling_is_needed():
    fp16, bf16, _ = resolve_precision(
        "microsoft/deberta-v3-large", "cuda", probe=AMPERE
    )
    assert (fp16, bf16) == (False, True)


def test_t4_uses_fp16_for_base_but_fp32_for_the_overflow_prone_large():
    assert resolve_precision("microsoft/deberta-v3-base", "cuda", probe=T4)[:2] == (
        True, False,
    )
    assert resolve_precision("microsoft/deberta-v3-large", "cuda", probe=T4)[:2] == (
        False, False,
    )


# -------------------------------------------------------------- tripwire
def test_is_finite_rejects_nan_inf_none_and_junk():
    assert is_finite(0.0) and is_finite(-2.5) and is_finite(3)
    for bad in (None, float("nan"), float("inf"), float("-inf"), "nan", "abc", True):
        assert not is_finite(bad), bad


def test_tripwire_passes_a_healthy_log_row():
    check_log_row({"loss": 2.31, "grad_norm": 1.4, "learning_rate": 2e-5, "step": 10})


def test_tripwire_fires_on_the_exact_row_that_started_the_real_failure():
    # verbatim shape of the first log line of the failed 105-minute run
    try:
        check_log_row({"loss": 502.3, "grad_norm": float("nan"), "epoch": 0.1299})
    except NonFiniteLoss as exc:
        assert "grad_norm" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("nan grad_norm must abort the run")


def test_tripwire_fires_on_nan_eval_loss():
    try:
        check_log_row({"eval_loss": float("nan"), "eval_accuracy": 0.09806})
    except NonFiniteLoss:
        pass
    else:  # pragma: no cover
        raise AssertionError("nan eval_loss must abort the run")


def test_tripwire_ignores_rows_that_simply_lack_the_keys():
    check_log_row({"train_runtime": 6195.0, "epoch": 3.0})


# --------------------------------------------------------- class weights
def test_class_weights_are_inverse_frequency_and_preserve_the_loss_scale():
    ids = [0] * 90 + [1] * 10
    w = class_weights(ids, 2)
    # the invariant is sum(n_c * w_c) == N, NOT mean(w) == 1
    assert abs(90 * w[0] + 10 * w[1] - 100) < 1e-9
    assert w[1] > w[0]
    assert abs(w[0] - 100 / (2 * 90)) < 1e-9
    assert abs(w[1] - 100 / (2 * 10)) < 1e-9


def test_balanced_data_gives_all_ones():
    assert class_weights([0, 1, 2] * 5, 3) == [1.0, 1.0, 1.0]


def test_unseen_class_gets_weight_one_not_a_division_by_zero():
    w = class_weights([0, 0, 1, 1], 3)
    assert len(w) == 3 and w[2] == 1.0
    assert all(is_finite(x) for x in w)


def test_class_weights_on_the_real_split_shape_favour_the_rare_queues():
    # actual data/finetune_train.csv distribution
    counts = [1146, 1738, 159, 242, 1368, 2215, 583, 348, 494, 3375]
    ids = [i for i, n in enumerate(counts) for _ in range(n)]
    w = class_weights(ids, 10)
    assert all(is_finite(x) for x in w)
    assert w[9] < 1.0 < w[2]          # Technical Support down, General Inquiry up
    assert w[2] == max(w)             # rarest class gets the largest weight
    assert w[9] == min(w)             # most common class gets the smallest


def test_class_weights_reject_bad_input():
    for bad in ((( [0], 0)), (([5], 3))):
        try:
            class_weights(*bad)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"must reject {bad}")


def test_split_stats_are_sorted_by_size_and_shares_sum_to_one():
    labels = ["a", "b", "c"]
    rows = split_stats([0, 0, 0, 1, 2, 2], labels)
    assert [r[0] for r in rows] == ["a", "c", "b"]
    assert [r[1] for r in rows] == [3, 2, 1]
    assert abs(sum(r[2] for r in rows) - 1.0) < 1e-9


# ----------------------------------------------------------- presets
def test_every_preset_is_well_formed():
    for name, (model, batch, accum, lr) in PRESETS.items():
        assert model and batch > 0 and accum > 0 and 0 < lr < 1e-3
        if name == "mps-safe":
            # mps-safe should be Apple-friendly, but a current DeBERTa model
            # may still be refused on MPS and safely fall back to CPU.
            assert resolve_training_device(model, "auto", probe=APPLE)[0] == "cpu"


def test_bigger_presets_use_a_lower_learning_rate_than_base():
    assert PRESETS["large"][3] < PRESETS["base"][3]


def test_explicit_preset_always_wins_over_hardware_sniffing():
    assert choose_preset("mps-safe") == "mps-safe"
    assert choose_preset("large") == "large"


# ------------------------------------------------------- stdlib runner
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
