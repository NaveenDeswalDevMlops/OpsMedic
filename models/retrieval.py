# models/retrieval.py
"""Sub-task 1 (NLP): semantic search over historical tickets.

Given a new incident description, returns the most similar resolved
tickets from the FAISS index (built by scripts/build_index.py), each
with its real resolution text - the "we have fixed this before"
evidence that grounds the resolution sub-task.

Model: sentence-transformers/all-MiniLM-L6-v2 (SLM, 384-dim, ~80 MB).
Chosen for: strong semantic-similarity quality at CPU-friendly latency
(~10 ms/query) with zero API cost.
"""
from __future__ import annotations

import os
from typing import Any

from models.base import BaseSubTask, resolve_device
from src import config

SNIPPET_CHARS = 300  # description preview length in results


def filter_hits(
    scores: list[float],
    indices: list[int],
    meta_rows: list[dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    """Pure post-processing of a FAISS search: threshold + shape results.

    Separated from I/O so it is unit-testable without faiss installed.
    `meta_rows` is the full KB as a list of dicts; `indices` index into it.
    """
    hits: list[dict[str, Any]] = []
    for score, idx in zip(scores, indices):
        if idx < 0 or score < threshold:
            continue
        row = meta_rows[idx]
        hits.append(
            {
                "ticket_id": row["ticket_id"],
                "title": row["title"],
                "description": str(row["description"])[:SNIPPET_CHARS],
                "resolution": row["resolution"],
                "category": row["category"],
                "priority": row.get("priority", ""),
                "score": round(float(score), 4),
            }
        )
    return hits


class RetrievalTask(BaseSubTask):
    name = "retrieve"
    category = "NLP"
    cacheable = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model_name = config.EMBEDDING_MODEL
        self._encoder = None
        self._index = None
        self._meta_rows: list[dict[str, Any]] | None = None

    # -- lazy loading ---------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._index is not None:
            return
        import faiss  # lazy heavy imports
        import pandas as pd
        from sentence_transformers import SentenceTransformer

        index_path = os.path.join(config.INDEX_DIR, "index.faiss")
        meta_path = os.path.join(config.INDEX_DIR, "meta.csv")
        if not (os.path.isfile(index_path) and os.path.isfile(meta_path)):
            raise FileNotFoundError(
                f"Index not found in {config.INDEX_DIR}. "
                "Run: python scripts/build_index.py"
            )
        self._index = faiss.read_index(index_path)
        self._meta_rows = pd.read_csv(meta_path).to_dict(orient="records")
        self._encoder = SentenceTransformer(self.model_name, device=resolve_device())

    # -- execution --------------------------------------------------------
    def _run(self, payload: Any) -> list[dict[str, Any]]:
        """payload: incident text (str). Returns list of similar tickets."""
        query = str(payload).strip()
        if not query:
            raise ValueError("empty query")
        self._ensure_loaded()
        vec = self._encoder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        scores, idxs = self._index.search(vec, config.TOP_K)
        hits = filter_hits(
            scores[0].tolist(),
            idxs[0].tolist(),
            self._meta_rows,
            config.SIMILARITY_THRESHOLD,
        )
        # Retrieval confidence: how strong was the best match, and did
        # anything clear the threshold at all? A high no_evidence rate
        # means the generator is being asked to answer ungrounded.
        kept = [h["score"] for h in hits]
        self.report_signals(
            retrieval_top_score=round(max(kept), 4) if kept else 0.0,
            retrieval_mean_score=(
                round(sum(kept) / len(kept), 4) if kept else 0.0
            ),
            evidence_count=len(hits),
            no_evidence=not kept,
            similarity_threshold=config.SIMILARITY_THRESHOLD,
        )
        return hits
