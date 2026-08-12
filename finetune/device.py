# finetune/device.py
"""Device + precision policy for fine-tuning, and the NaN tripwire.

Pure stdlib logic (torch is imported lazily and optionally) so the whole
policy is unit-testable offline without a GPU.

Why this module exists
----------------------
DeBERTa-v2/v3 uses *disentangled attention*, which relies on gather /
bucketize kernels that are numerically broken on Apple's MPS backend in
current PyTorch builds. Training appears to start, then the first
backward pass produces non-finite gradients, one optimizer step writes
NaN into every weight, and the run silently continues for hours while
the model predicts label index 0 for every row.

That is exactly what happened on a 105-minute M4 run:
    loss 502.3 -> 0, grad_norm nan, eval_loss nan,
    eval_accuracy 0.0981 == 202/2060 == share of the alphabetically
    first class ("Billing and Payments").

So: MPS is refused for DeBERTa unless explicitly forced, fp16 is refused
where it is known to overflow, and a tripwire aborts within seconds if
the loss or grad-norm ever goes non-finite.
"""
from __future__ import annotations

import math

# Architectures whose MPS kernels are known to produce NaN in training.
# DeBERTa v3 is currently broken on the Apple MPS backend in the current
# PyTorch builds used by this project, even at the base size.
MPS_UNSAFE_SUBSTRINGS: tuple[str, ...] = (
    "deberta-v3-base",
    "deberta-v3-large",
    "deberta-v2-xlarge",
)

# Architectures that overflow fp16 at large scale (keep them in fp32/bf16).
FP16_UNSAFE_SUBSTRINGS: tuple[str, ...] = ("deberta-v3-large", "deberta-v2-xlarge")


def is_mps_unsafe(model_name: str) -> bool:
    """True if this architecture must not be trained on the MPS backend."""
    low = model_name.lower()
    return any(s in low for s in MPS_UNSAFE_SUBSTRINGS)


def is_fp16_unsafe(model_name: str) -> bool:
    """True if this architecture is known to overflow in fp16 training."""
    low = model_name.lower()
    return any(s in low for s in FP16_UNSAFE_SUBSTRINGS)


def _probe() -> dict[str, bool]:
    """Best-effort hardware probe; all False when torch is absent."""
    try:
        import torch
    except ImportError:
        return {"cuda": False, "mps": False, "bf16": False}
    cuda = bool(torch.cuda.is_available())
    bf16 = False
    if cuda:
        try:
            bf16 = bool(torch.cuda.is_bf16_supported())
        except Exception:  # noqa: BLE001 - old torch without the helper
            bf16 = False
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    return {"cuda": cuda, "mps": mps, "bf16": bf16}


def resolve_training_device(
    model_name: str,
    requested: str = "auto",
    force_mps: bool = False,
    probe: dict[str, bool] | None = None,
) -> tuple[str, list[str]]:
    """Pick a training device and explain every decision.

    Returns (device, notes). `device` is one of 'cuda' | 'mps' | 'cpu'.
    Preference order for 'auto' is cuda > mps > cpu, with MPS skipped for
    architectures listed in MPS_UNSAFE_SUBSTRINGS.
    """
    hw = probe if probe is not None else _probe()
    notes: list[str] = []
    req = (requested or "auto").strip().lower()

    if req not in ("auto", "cuda", "mps", "cpu"):
        raise ValueError(f"unknown device {requested!r}")

    if req == "cuda":
        if not hw["cuda"]:
            notes.append("cuda requested but unavailable -> falling back to cpu")
            return "cpu", notes
        return "cuda", notes

    if req == "mps":
        if not hw["mps"]:
            notes.append("mps requested but unavailable -> falling back to cpu")
            return "cpu", notes
        if is_mps_unsafe(model_name) and not force_mps:
            notes.append(
                f"REFUSING mps for '{model_name}': this architecture produces "
                "NaN gradients on the MPS backend. Using cpu instead "
                "(pass --force-mps to override)."
            )
            return "cpu", notes
        if is_mps_unsafe(model_name) and force_mps:
            notes.append(
                f"WARNING: --force-mps with '{model_name}'. NaN loss is likely."
            )
        return "mps", notes

    if req == "cpu":
        return "cpu", notes

    # auto
    if hw["cuda"]:
        return "cuda", notes
    if hw["mps"]:
        if is_mps_unsafe(model_name) and not force_mps:
            notes.append(
                f"mps detected but skipped for '{model_name}' (known NaN on the "
                "MPS backend). Using cpu. Train on a CUDA GPU (e.g. Kaggle) for "
                "a fast run, or pass --force-mps to override."
            )
            return "cpu", notes
        return "mps", notes
    return "cpu", notes


def resolve_precision(
    model_name: str,
    device: str,
    probe: dict[str, bool] | None = None,
) -> tuple[bool, bool, list[str]]:
    """Pick mixed precision. Returns (fp16, bf16, notes).

    Rules:
      - non-CUDA devices always train in fp32 (fp16 on CPU/MPS is slower
        and less stable, and Trainer's fp16 path assumes CUDA amp);
      - CUDA with bf16 support (Ampere+) prefers bf16, which has fp32's
        exponent range and therefore never needs loss scaling;
      - CUDA without bf16 (T4, P100) uses fp16, except for architectures
        in FP16_UNSAFE_SUBSTRINGS which stay in fp32.
    """
    hw = probe if probe is not None else _probe()
    notes: list[str] = []
    if device != "cuda":
        notes.append(f"precision: fp32 (mixed precision is CUDA-only; device={device})")
        return False, False, notes
    if hw["bf16"]:
        notes.append("precision: bf16 (Ampere or newer - no loss scaling needed)")
        return False, True, notes
    if is_fp16_unsafe(model_name):
        notes.append(
            f"precision: fp32 - '{model_name}' is known to overflow in fp16 and "
            "this GPU has no bf16 support"
        )
        return False, False, notes
    notes.append("precision: fp16 (no bf16 on this GPU)")
    return True, False, notes


def is_finite(value) -> bool:
    """True only for real, finite numbers. None / nan / inf / junk -> False."""
    if value is None or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


class NonFiniteLoss(RuntimeError):
    """Raised by the tripwire when training numerics have collapsed."""


def check_log_row(logs: dict) -> None:
    """Abort the run if a Trainer log row shows non-finite numerics.

    Watches 'loss', 'grad_norm' and 'eval_loss'. Keys that are absent are
    ignored; keys that are present but not finite raise NonFiniteLoss.
    """
    for key in ("loss", "grad_norm", "eval_loss"):
        if key in logs and not is_finite(logs[key]):
            raise NonFiniteLoss(
                f"non-finite {key}={logs[key]!r} at step "
                f"{logs.get('step', '?')} / epoch {logs.get('epoch', '?')}.\n"
                "Training numerics have collapsed - every weight is now NaN and "
                "the model would predict a single class forever.\n"
                "Most common causes: (1) DeBERTa on the MPS backend, "
                "(2) fp16 overflow on a large model, (3) learning rate far too "
                "high. Aborting now instead of wasting hours."
            )
