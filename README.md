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
pip install -r requirements.txt
copy .env.example .env                              # add GROQ_API_KEY (free: console.groq.com)

# data: drop the dataset CSV(s) into data/raw/ (see Dataset below), then
python scripts/prepare_dataset.py     # normalize + fine-tune split
python scripts/build_index.py         # FAISS index (~5 min CPU)
python finetune/train.py --max-rows 8000 --epochs 3 --balanced   # ~2.5h CPU
python finetune/compare.py            # base vs fine-tuned evidence

python -m pytest tests -v             # 100 offline tests, 12 files
python scripts/smoke_test.py          # all 6 sub-tasks end-to-end
streamlit run app.py                  # chat + composer, plus /monitor pages
```

Or one command after placing the dataset: `run_all.bat` (Windows) /
`./run_all.sh` (Linux/macOS).

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
| Summarize | sshleifer/distilbart-cnn-12-6 | near-BART ROUGE at 2× CPU speed |
| Classify | distilbert-base-uncased → fine-tuned | minutes-scale fine-tuning, strong for size |
| ASR | openai/whisper-tiny | real-time CPU transcription |
| TTS | facebook/mms-tts-eng | simple local VITS, no vocoder setup |

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
llmops/                 metrics · cache · conversations · attachments · insights · system stats
finetune/               data · train · evaluate · compare (+ artifacts/)
scripts/                prepare_dataset · build_index · generate_tickets · smoke_test
tests/                  100 offline tests across 12 files (pytest or plain python)
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
python -m pytest tests -v      # all 12 files, 100 tests
python -m pytest tests -q      # compact "100 passed"
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

`beautifulsoup4` (ships soupsieve) is a test-only pin; that one suite
skips itself without it, so a short count of 85 instead of 100 means it is
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
