# Makefile — OpsMedic (CCZG506 Assignment II)
# Single entry point for setup, training, tests and the app.
#
# Quick start (after placing the dataset CSVs in data/raw/ — see README):
#     make            # show available targets
#     make all        # install -> data -> index -> fine-tune -> test -> run
#
# Notes:
# - data / index / fine-tune are real file targets, so make skips completed steps.
# - run does not remove artifacts. Use make clean / make distclean only when you want to rebuild.

PYTHON    ?= python
PORT      ?= 8501
FT_ROWS   ?= 0
FT_EPOCHS ?= 3

TICKETS  := data/tickets.csv
GENERATED_TICKETS := data/generated_tickets.csv
INDEX    := data/index/index.faiss
FT_STAMP := finetune/artifacts/.trained
FT_LABELS := $(wildcard finetune/artifacts/*/labels.json)

.DEFAULT_GOAL := help
.PHONY: help all install data prepare-data generate-data index finetune retrain train-gpu compare test test-quick \
        smoke run verify clean clean-index clean-metrics distclean

help: ## Show this help
	@$(PYTHON) -c "import re;print('OpsMedic make targets:');[print(f'  {m.group(1):<13s} {m.group(2)}') for l in open('Makefile',encoding='utf-8') for m in [re.match(r'^([a-zA-Z][a-zA-Z0-9_-]*):.*?## (.*)$$',l)] if m]"

all: install data index finetune test run ## Full bootstrap, then launch the app

install: ## Install pinned dependencies
	$(PYTHON) -m pip install -r requirements.txt

prepare-data: $(TICKETS) ## Prepare the real KB dataset from data/raw/*.csv into data/tickets.csv
$(TICKETS): scripts/prepare_dataset.py
	@$(PYTHON) -c "import glob,sys;sys.exit(None if glob.glob('data/raw/*.csv') else 'ERROR: no CSV in data/raw/ - download the dataset first (README > Dataset)')"
	$(PYTHON) scripts/prepare_dataset.py --raw-csv data/raw/aa_dataset-tickets-multi-lang-5-2-50-version.csv --top-queues 0

generate-data: $(GENERATED_TICKETS) ## Generate a synthetic sample ticket file into data/generated_tickets.csv
$(GENERATED_TICKETS): scripts/generate_tickets.py
	$(PYTHON) scripts/generate_tickets.py --out $(GENERATED_TICKETS)

data: prepare-data ## Alias for the real KB preparation step

index: $(INDEX) ## Build the FAISS retrieval index (~5 min CPU)
$(INDEX): $(TICKETS) scripts/build_index.py
	$(PYTHON) scripts/build_index.py

finetune: $(FT_STAMP) ## Fine-tune the classifier, reusing any existing artifact
$(FT_STAMP): $(TICKETS) finetune/train.py finetune/data.py
ifeq ($(FT_LABELS),)
	@echo "[finetune] running fine-tune..."
	@$(PYTHON) finetune/train.py --max-rows $(FT_ROWS) --epochs $(FT_EPOCHS) --balanced --device cpu || exit 1
	@$(PYTHON) finetune/compare.py
else
	@$(PYTHON) -c "print('  reusing artifact: $(firstword $(FT_LABELS))   (make retrain to force)')"
endif
	@$(PYTHON) -c "import pathlib;p=pathlib.Path('$(FT_STAMP)');p.parent.mkdir(parents=True,exist_ok=True);p.touch()"

train-gpu: ## Fine-tune on the FULL dataset with GPU settings (Kaggle/CUDA)
	$(PYTHON) finetune/train.py --max-rows 0 --epochs 3 --batch 32 --lr 2e-5 --balanced --device cuda
	$(PYTHON) finetune/compare.py

train-mps: ## Fine-tune on Apple MPS explicitly
	$(PYTHON) finetune/train.py --max-rows 0 --epochs 3 --batch 8 --lr 1e-5 --balanced --device mps
	$(PYTHON) finetune/compare.py

retrain: ## Force a fresh fine-tune even if an artifact exists (~2.5 h CPU)
	$(PYTHON) finetune/train.py --max-rows $(FT_ROWS) --epochs $(FT_EPOCHS) --balanced
	$(PYTHON) finetune/compare.py

compare: ## Base vs fine-tuned evaluation -> finetune/artifacts/compare.json
	$(PYTHON) finetune/compare.py

test: ## Run the full offline test suite
	$(PYTHON) -m pytest tests -v

test-quick: ## Compact test output, good for a report screenshot
	$(PYTHON) -m pytest tests -q

smoke: ## Exercise all six sub-tasks end to end (downloads models on first run)
	$(PYTHON) scripts/smoke_test.py

run: ## Launch the Streamlit app (override with PORT=8600)
	$(PYTHON) -m streamlit run app.py --server.port $(PORT)

verify: ## Print environment and artifact status
	@$(PYTHON) -c "import sys,glob,os;print('python       ',sys.version.split()[0]);print('raw csv      ',len(glob.glob('data/raw/*.csv')),'file(s)');print('tickets.csv  ',os.path.exists('data/tickets.csv'));print('generated tickets ',os.path.exists('data/generated_tickets.csv'));print('faiss index  ',os.path.exists('data/index/index.faiss'));print('ft artifacts ',glob.glob('finetune/artifacts/*/labels.json') or 'NONE - run make finetune');print('compare.json ',any(glob.glob('finetune/artifacts/**/compare.json',recursive=True)));print('metrics db   ',os.path.exists('data/llmops_metrics.db'));print('GROQ_API_KEY ','set in env' if os.environ.get('GROQ_API_KEY') else 'not in env (may still be in .env)')"

clean: ## Remove __pycache__ and .pyc (keeps data, models and metrics)
	@$(PYTHON) -c "import pathlib,shutil;[shutil.rmtree(p,ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')];[p.unlink() for p in pathlib.Path('.').rglob('*.pyc') if p.exists()];print('caches removed')"

clean-index: ## Delete the FAISS index so the next build regenerates it
	@$(PYTHON) -c "import shutil;shutil.rmtree('data/index',ignore_errors=True);print('index removed')"

clean-metrics: ## DESTRUCTIVE: wipe the metrics DB (your LLMOps evidence)
	@$(PYTHON) -c "import sys;sys.exit('Refusing: this deletes your LLMOps evidence. Run: make clean-metrics CONFIRM=yes') if '$(CONFIRM)' \!= 'yes' else None"
	@$(PYTHON) -c "import os;[os.remove(f) for f in ['data/llmops_metrics.db','data/response_cache.db'] if os.path.exists(f)];print('metrics + cache DBs removed')"

distclean: clean clean-index ## Caches + index (keeps raw data, artifacts, metrics)
	@$(PYTHON) -c "print('distclean done - raw data, fine-tune artifacts and metrics kept')"
