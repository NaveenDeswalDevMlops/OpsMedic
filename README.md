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

pytest tests -v                       # 36 offline tests
python scripts/smoke_test.py          # all 6 sub-tasks end-to-end
streamlit run app.py                  # ChatGPT-style chat (history sidebar) + /monitor
```

Or one command after placing the dataset: `run_all.bat` (Windows) /
`./run_all.sh` (Linux/macOS).

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
app.py                  Streamlit UI (the unified journey)
pages/                  LLMOps dashboard · fine-tune comparison · Monitor (7-tab LLM-Ops console)
ui/                     enterprise UI kit (CSS, KPI tiles, Plotly gauges)
models/                 one wrapper per sub-task (uniform run() -> {output, metrics})
llmops/                 metrics · cache · conversations (chat history) · dashboard helpers · system stats
finetune/               data · train · evaluate · compare (+ artifacts/)
scripts/                prepare_dataset · build_index · smoke_test
tests/                  36 offline unit tests (pytest or plain python)
data/                   raw/ · tickets.csv · sops/ · index/ · metrics DBs
```

## LLMOps metrics (persisted + dashboarded)

latency p50/p95 · tokens in/out · est. cost/request · error rate ·
cache-hit rate · throughput · user 👍/👎 feedback · model-version stamps.
Practices: prompt versioning, response caching (TTL), seeded/reproducible
training, graceful degradation, config-driven model registry.

## Troubleshooting

- **`operator torchvision::nms does not exist`** — a stale torchvision in
  a shared venv; `pip uninstall torchvision -y` (unused here) or install
  `torchvision==0.19.1` to match torch 2.4.1.
- **Hugging Face blocked by org network** — download dataset CSVs and
  models via browser/hotspot; place CSVs in `data/raw/`.
- **Groq key missing** — the app degrades gracefully (shows the most
  similar ticket's real resolution) but generation needs
  `GROQ_API_KEY` in `.env`.
- **First run is slow** — one-time model downloads (~600 MB); the p95
  latency metric includes this cold start.
