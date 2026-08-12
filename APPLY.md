# Fine-tuning fix — what to copy and what to run

## Files in this bundle

| File | Action |
|---|---|
| `finetune/device.py` | **new** |
| `finetune/train.py` | **replaces** yours |
| `finetune/data.py` | **replaces** yours (all existing functions unchanged, two added) |
| `tests/test_finetune_device_offline.py` | **new** |
| `.env.upgrade` | merge into your `.env` |

Nothing else is touched. `src/config.py`, the app, the UI and the TTS chain are
left alone — the model upgrades go through `.env` overrides.

## 1. Fix the failing test first (one line)

`tests/test_tts_offline.py::test_default_knobs_do_not_alter_synthesis` fails
because `TTS_SPEAKING_RATE` is `0.9` and the test asserts `1.0`. The test is the
source of truth: the approved audio sample was EQ over **default-rate**
synthesis. Set it back:

- if the value is env-driven in your `config.py`: `TTS_SPEAKING_RATE=1.0` in `.env`
- if it is hard-coded: change the literal `0.9` to `1.0` in `src/config.py`

Do **not** relax the assertion — that is the guard that stopped the tone
regression last time.

## 2. Copy the four files, then run the offline checks

```bash
python -m pytest tests -v
# expect: previous 158 passes + 26 new = 184 passed, 0 failed
```

## 3. Numerics check BEFORE any long run (about a minute)

```bash
python finetune/train.py --preflight-only
```

Healthy output ends with `[preflight] OK`. If it raises, the device/model pair is
broken and the real run must not start.

## 4. Train

On Kaggle / any CUDA GPU (recommended, full dataset):

```bash
python finetune/train.py --preset large --epochs 3
```

On the M4 MacBook Air:

```bash
python finetune/train.py --preset mps-safe --epochs 3
```

`--preset large` on a Kaggle T4 will train in fp32 (deberta-v3-large overflows
fp16 and the T4 has no bf16). If that is too slow, use `--preset base`, which
runs fp16 on a T4 and still beats roberta-large on this task.

## 5. Compare

```bash
python finetune/compare.py --max-rows 0
python finetune/evaluate.py --max-rows 0
```

## What changed and why

| Change | Reason |
|---|---|
| MPS refused for DeBERTa | its disentangled-attention kernels produce NaN gradients on MPS; this is what destroyed the 105-minute run |
| fp16 refused for `deberta-v3-large` on non-bf16 GPUs | attention logits overflow fp16 at that width |
| NaN tripwire on every log row | aborts in seconds instead of hours |
| `--preflight-only` | 30-step overfit check proves the forward/backward pass works before committing |
| refuse to save an artifact whose final eval is NaN | the old run happily wrote NaN weights to disk |
| `--max-rows 0` / `--eval-rows 0` default | full 11,668 / 2,060 rows |
| class-weighted loss replaces median downsampling | keeps all 11,668 rows instead of discarding 5,522, and still fights the 28.9% → 1.4% imbalance |
| dynamic padding + `group_by_length` | the old code padded every row to 256 tokens; most tickets are far shorter |
| `load_best_model_at_end` on `f1_macro` | saves the best epoch, not the last |
| lr 2e-5 / 8e-6 with 6% warmup, grad clip 1.0 | 1e-5 with no warmup was both too low for base and unclipped |
