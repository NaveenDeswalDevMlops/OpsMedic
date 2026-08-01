# OpsMedic — GenAI Incident Copilot

**BITS Pilani CCZG506 (API-driven Cloud Native Solutions) — Assignment II**

OpsMedic is an AI incident-resolution copilot for IT service desks:
report an incident by **voice or text**, and it **classifies** the queue
(fine-tuned DistilBERT), **retrieves similar historical tickets** with
their real resolutions (FAISS + MiniLM), **generates grounded resolution
steps** (Groq Llama RAG over past fixes + the linked SOP),
**summarizes** for handover (DistilBART), and **reads the fix aloud**
(MMS-TTS). Every call is metered by a built-in LLMOps layer.

- **Domain**: IT Incident Management (ITSM)
- **Categories**: NLP + Speech Recognition
- **Sub-tasks (6)**: semantic retrieval · grounded resolution generation ·
  summarization · ticket classification (fine-tuned) · ASR · TTS

## Architecture

```
 🎙 voice ─► [ASR whisper-tiny] ─► incident text ◄─ ⌨️ typed
                                        │
                        [classify: DistilBERT fine-tuned]──► queue
                                        │                      │
                 [retrieve: MiniLM + FAISS over 13.7k tickets] │
                                        │  similar tickets + real resolutions
                                        ▼                      ▼
                 [resolve: Groq llama-3.1-8b — RAG on tickets + SOP]
                                        │
                        [summarize: distilbart-cnn]
                            │                   │
                     handover text      [TTS mms-tts-eng] ─► 🔊
        every call ─► llmops (SQLite): latency p50/p95, tokens,
        cost, errors, cache hits, throughput, feedback, versions
```

## Quickstart

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
copy .env.example .env                              # add GROQ_API_KEY (free: console.groq.com)
# then drop the dataset CSV(s) into data/raw/ (see Dataset below)

make                # list every target
make all            # install -> data -> index -> fine-tune -> test -> run
```

`make` is the single entry point. The data, index and fine-tune steps are
real **file targets**, so make skips whatever is already up to date — no
hand-written "skip if exists" guards, and no accidental 2.5-hour retrain.

| Target | Does |
|---|---|
| `make verify` | print environment + artifact status (run this first if unsure) |
| `make install` | install pinned dependencies |
| `make data` | normalise `data/raw/*.csv` into `data/tickets.csv` |
| `make index` | build the FAISS index (~5 min CPU) |
| `make finetune` | fine-tune the classifier, **reusing** any existing artifact |
| `make retrain` | force a fresh fine-tune (~2.5 h CPU) |
| `make compare` | base vs fine-tuned evidence → `compare.json` |
| `make test` / `test-quick` | 173 offline tests (`-q` for a report screenshot) |
| `make smoke` | all six sub-tasks end to end |
| `make run` | launch the app (`make run PORT=8600`) |
| `make clean` / `distclean` | caches / caches + index — **never** touches your metrics DB |

Overrides: `make PYTHON=python3 test`, `make finetune FT_ROWS=2000 FT_EPOCHS=1`.

**Windows:** `make` is not installed by default — use `choco install make`,
`scoop install make`, Git Bash, or WSL. Every recipe uses python one-liners
instead of shell builtins so they run under `cmd.exe` too.

## The chat UI

One bordered composer holds the whole journey: a hint line, the message
field, staged-context chips, then a control row — pills on the left, a
dark circular send button flush right. Every control is a real Streamlit
widget bound to a real code path; none of them are decorative.

| Control | What it actually does |
|---|---|
| 📎 **Attach** | log · txt · md · csv · json · pdf · png/jpg → `llmops/attachments.extract_text` → merged into the incident context before retrieval |
| 🔍 **Search** | retrieval scope (**Ticket KB (RAG)** or **No retrieval**) plus Top-K and similarity threshold — a one-click grounded-vs-ungrounded comparison |
| ✍ **Writing Styles** | swaps the resolver's system prompt (see below) and selects the triage variant: `base` / `finetuned` / `auto` |
| 🎙 **Voice** | `st.audio_input` or an audio upload → whisper-tiny → the transcript lands in the message box, editable before sending |
| ⬆ **Send** | identical to pressing Enter |

The mic transcribes automatically on a new recording, fingerprinting the
audio bytes so a Streamlit rerun cannot re-run Whisper (or log a
duplicate metrics row) on the same clip.

The sidebar groups into collapsible sections — **Workspace**, **Recents**
(day-bucketed conversation history with delete), **Knowledge Base**
(indexed counts + Rebuild Index) — and every page renders the breadcrumb
topbar via a single `nav.render_page_chrome()` call, so no page can ship
without a header.

## Writing Styles = versioned prompts

The style control is an LLMOps experiment, not a cosmetic dropdown: each
style is a distinct prompt version, stamped onto every metrics row
(streamed, cache-hit and fallback alike), so the Monitor console can
group latency, tokens and cost by prompt.

| Style | `prompt_version` | Output shape |
|---|---|---|
| Step-by-step *(default)* | `resolve-v1-stepwise` | diagnosis · numbered steps · escalation condition |
| Concise | `resolve-v1-concise` | ≤ 4 bullets, no preamble |
| Customer-facing | `resolve-v1-customer` | plain-language reply to the requester + internal `Evidence:` line |
| Shift handover | `resolve-v1-handover` | IMPACT / DIAGNOSIS / ACTIONS TAKEN / NEXT STEPS FOR ON-CALL / ESCALATE IF |

All four are built from one shared `GROUNDING_RULE`, so no style can relax
the anti-hallucination contract — enforced by a test.

## Models (all free; local unless noted)

| Sub-task | Model | Why |
|---|---|---|
| Retrieval | sentence-transformers/all-MiniLM-L6-v2 | strong similarity, ~10 ms/query CPU |
| Resolution | llama-3.1-8b-instant (Groq API, free tier) | fluent grounded generation, near-zero cost |
| Summarize | facebook/bart-large-cnn | full BART beats the distilled variant on technical prose; drop-in |
| Classify | **microsoft/deberta-v3-base → fine-tuned** | strongest base-size encoder for short jargon-dense text; MIT |
| ASR | openai/whisper-medium | accuracy on accented/noisy input; slower on a fanless Air — override to whisper-small for a latency-critical demo |
| TTS | facebook/mms-tts-eng (default) · **hexgrad/Kokoro-82M** (opt-in) | mms needs no system packages; Kokoro adds 24 kHz + voice presets |

## Speech delivery (why the audio is shaped, not just generated)

A small VITS has no prosody or style conditioning, so "sounds like
reading" cannot be tuned out of the model. `models/tts_prosody.py` shapes
delivery around it instead, and the numbers below were measured on real
output rather than assumed:

| Problem measured | Fix |
|---|---|
| 34% of the clip was dead air, giving a halting rhythm | split into sentences/steps, synthesize each separately, trim per-chunk padding, rejoin with deliberate pauses (140 ms clause · 260 ms sentence · 420 ms before a step) and 8 ms anti-click fades |
| markdown and `1.` prefixes were spoken aloud | stripped; `1. Clear the cache` is spoken as "First, Clear the cache" |
| 83% of energy below 500 Hz, only **2.7%** in the 1.5–4 kHz consonant band → words arrived mushy | rumble cut + presence lift, raising that band to **4.5%** (1.7×) |
| peak at 0.86 full scale, harsh | normalised to 0.72 |

Note the counter-intuitive part: the output is *dull*, not harsh (0.13% of
energy above 4 kHz), so high-frequency softening is **off by default**
(`TTS_SOFTEN=0.0`) and exists only for a future brighter engine. All
knobs are in `src/config.py` (`TTS_PRESENCE`, `TTS_RUMBLE_CUT`,
`TTS_PEAK`, `TTS_SPEAKING_RATE`, `TTS_NOISE_SCALE_DURATION`, …) and
overridable per call in the TTS payload.

**Text normalisation is the other half.** `facebook/mms-tts-eng` has a
CHARACTER vocabulary with no digits and no pronunciation lexicon, and
`VitsTokenizer` silently drops characters it does not know — so `L2`
arrived as a bare `l` and came out as a garbled "llllll".
`models/tts_text.py` rewrites text before tokenization:

| Input | Spoken as |
|---|---|
| `L2` / `T3` | level two / tier three (ITSM support levels, not maths — pass `tiers=False` for L2-norm contexts) |
| `P1`, `SEV2` | priority one, severity two |
| `SSO`, `AD`, `MFA` | single sign on, active directory, multi factor authentication |
| `VPN`, `DNS` | vee pee en, dee en ess (no vowel to voice, so spelled) |
| `OPM-00042` | ticket oh pee em forty two (leading zeros dropped) |
| `10.0.0.1`, `14:30`, `40%` | ten point zero point zero point one, fourteen thirty, forty percent |

A final `scrub()` pass guarantees no digit and no out-of-vocabulary
character reaches the model, so terms the rules miss degrade to spelled
letters rather than to noise.

Two delivery modes, because they sound audibly different:

- `TTS_CHUNKED=1` (default) — sentence-by-sentence, removing the dead air,
  but the model's prosody contour restarts at every chunk;
- `TTS_CHUNKED=0` — one continuous pass with the EQ on top. This is the
  pipeline the tone above was validated against, so use it if the chunked
  version sounds less natural to you.

The VITS sampling knobs (`TTS_SPEAKING_RATE`, `TTS_NOISE_SCALE`,
`TTS_NOISE_SCALE_DURATION`) default to the model's OWN defaults — they
change the synthesis itself, not just the tone, so they are opt-in.

The render loop takes the synth function as an argument, so swapping in a
different engine (Piper, edge-tts, SpeechT5) reuses the whole layer.

## Compute device (CPU / Apple Silicon / CUDA)

`config.resolve_device()` picks `cuda` → `mps` → `cpu` and is passed to every
model, so the same code runs on a laptop and on a training GPU.

```bash
DEVICE=auto                     # or cpu / mps / cuda
PYTORCH_ENABLE_MPS_FALLBACK=1   # REQUIRED on Apple Silicon
```

The MPS fallback flag is not optional on a Mac: a few ops are still
unimplemented on that backend and hard-error without it. `fp16` is enabled
for training on CUDA only — on MPS it still produces NaNs in some models.

**Fine-tuning on a GPU** (Kaggle, ~30 h quota) uses the full dataset rather
than the CPU-sized 8k subset:

```bash
make train-gpu     # full dataset; batch and lr are derived from the base model
```

`FT_ROWS` defaults to **0 (the full prepared dataset)**. It was capped at 8,000
while training was CPU-bound — with a GPU, capping the data is the thing most
likely to make the fine-tune look worse than it is. On a CPU-only machine
override it: `make finetune FT_ROWS=8000`.

Batch size, learning rate and gradient accumulation are **not** hard-coded in
the Makefile: `finetune/train.py` derives them from the chosen base model
(`-base` → batch 32 / lr 2e-5; `-large` → batch 16 / lr 1e-5 / accum 2, with
fp16 auto-disabled because DeBERTa-large NaNs in half precision). Hard-coding
them would silently apply the wrong settings the moment the base model changes.

The learning rate matters: DeBERTa-v3 is LR-sensitive and diverges at the
`5e-5` that suited DistilBERT, so the default is now `2e-5` with 6% warmup
and 0.01 weight decay. On a fanless MacBook Air, prefer the smaller model of
each pair for a live demo — sustained inference thermal-throttles.

**Switching to the Kokoro voice:**

```bash
pip install kokoro && brew install espeak-ng     # macOS
# then in .env:
TTS_MODEL=hexgrad/Kokoro-82M
KOKORO_VOICE=bf_emma
TTS_PRESENCE=0          # see below
TTS_RUMBLE_CUT=0
```

Turning the EQ off is important. The presence lift and rumble cut were
measured against mms-tts-eng's specific defect (83% of energy below 500 Hz);
Kokoro does not share it, so leaving the correction on over-brightens it.

## Dataset (cited)

T. Bueck, *Customer Support Tickets*, Hugging Face. DOI:
[10.57967/hf/6184](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets)
· CC-BY-NC-4.0 · 61.8k tickets with agent resolutions (`answer`), queue,
priority, type, language. Kaggle mirror:
`tobiasbueck/multilingual-customer-support-tickets`. Download the CSVs
from the HF *Files* tab into `data/raw/` (org networks often block the
CLI; browser download works).

The RAG knowledge base, the fine-tuning train/test splits, and the
grounding evidence all come from this single source; per-queue SOPs in
`data/sops/` provide the linked-SOP grounding layer.

## Folder structure

```
app.py                  chat journey + composer
pages/                  1 LLMOps dashboard · 2 fine-tune comparison · 3 Monitor (7-tab console)
ui/                     theme (CSS kit) · composer (the chat input bar) · nav (sidebar + topbar chrome)
models/                 one wrapper per sub-task (uniform run() -> {output, metrics})
                        + tts_prosody (delivery) · tts_text (pronunciation)
llmops/                 metrics · cache · conversations · attachments · insights · system stats
finetune/               data · train · evaluate · compare (+ artifacts/)
scripts/                prepare_dataset · build_index · generate_tickets · smoke_test
tests/                  184 offline tests across 15 files (pytest or plain python)
data/                   raw/ · tickets.csv · sops/ · index/ · metrics + cache + conversations DBs
```

## LLMOps metrics

**Captured in real time** — one SQLite row written synchronously as each
sub-task call completes, so the dashboards reflect the run that just
happened with no batch job in between:

latency (wall-clock, per call) · **time-to-first-token** and
**tokens/sec** for streamed generation · tokens in/out (exact from the
Groq `usage` field, estimated only for local models) · est. cost/request
from a per-model price table · status + error text → error rate ·
cache-hit flag → cache-hit rate · throughput (requests/min, derived) ·
user 👍/👎 feedback · model-version and prompt-version stamps.

Aggregations (`summary`, `summary_by_subtask`) compute p50/p95 on read,
so any window is available without pre-aggregation.

**Not real time, by design** — fine-tuning metrics are offline artifacts,
since training is a batch job run once: `training_log.json` (per-epoch
loss, accuracy, macro-F1) and `compare.json` (base vs fine-tuned, per
class) are written by `finetune/train.py` and `finetune/compare.py`, then
*read* live by `pages/2_Finetune_Comparison.py`. System stats
(CPU/RAM/disk in `pages/3_Monitor.py`) are sampled point-in-time on page
load and not persisted, so there is no historical resource curve.

Practices: prompt versioning (writing styles), response caching (TTL,
keyed on the full payload incl. style), seeded and reproducible training,
graceful degradation, config-driven model registry.

## Tests

```bash
python -m pytest tests -v      # all 15 files, 184 tests
python -m pytest tests -q      # compact "184 passed"
python tests/test_llmops.py    # any single file also runs standalone
```

Everything runs offline — no network, no model downloads, no browser. The
suite covers config and dataset prep, all six sub-task wrappers, the
streaming resolver, the LLMOps metrics/cache layer, attachments,
conversation history, insights, fine-tune data handling, and the UI:

- `test_composer_offline.py` — composer state machine and style registry
- `test_theme_selectors_offline.py` — runs the real CSS selectors against
  a replica of Streamlit's DOM via **soupsieve**, so a rule that leaks
  onto the wrong elements fails the build instead of the demo
- `test_nav_chrome_offline.py` — renders the nav chrome against a
  recording Streamlit stub and asserts the topbar is emitted
- `test_tts_prosody_offline.py` — measures real spectral energy to prove
  the presence lift and rumble cut do what they claim
- `test_tts_offline.py` — drives the whole TTSTask with a fake
  `soundfile` and a stubbed synth, so chunking, tone overrides, the
  char budget and the error paths are covered without a 140 MB download

`beautifulsoup4` (ships soupsieve) is a test-only pin; that one suite
skips itself without it, so a short count of 169 instead of 184 means it is
missing, not that something broke.

## Troubleshooting

- **`operator torchvision::nms does not exist`** — a stale torchvision in
  a shared venv; `pip uninstall torchvision -y` (unused here) or install
  `torchvision==0.19.1` to match torch 2.4.1.
- **Hugging Face blocked by org network** — download dataset CSVs and
  models via browser/hotspot; place CSVs in `data/raw/`.
- **Groq key missing** — the app degrades gracefully (shows the most
  similar ticket's real resolution) but generation needs
  `GROQ_API_KEY` in `.env`. Metrics rows for the fallback carry
  `extra.error = "GROQ_API_KEY not set (streamed fallback)"`, so a
  dashboard full of 0.0 ms resolve latencies means the key is absent.
- **Mic records but nothing transcribes** — `st.audio_input` needs
  Streamlit ≥ 1.39 (pinned 1.40.1) and browser microphone permission;
  decoding needs `soundfile`. Non-WAV uploads depend on the libsndfile
  build, so prefer WAV if an upload fails.
- **Composer or chips look unstyled** — the CSS scopes itself with
  `:has()`, which needs Chrome 105+ / Safari 15.4+ / Firefox 121+. Also
  check browser zoom is at 100%; heavy zoom exaggerates truncation.
- **First run is slow** — one-time model downloads (~600 MB); the p95
  latency metric includes this cold start.
