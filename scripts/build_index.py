# scripts/build_index.py
"""Build the FAISS retrieval index over data/tickets.csv.

Embeds "title. description" per ticket with sentence-transformers
(all-MiniLM-L6-v2, 384-dim), L2-normalizes, and stores an inner-product
FAISS index (== cosine similarity) plus row metadata.

Outputs (in INDEX_DIR):
    index.faiss   - the vector index
    meta.csv      - rows aligned 1:1 with index vectors

Usage:
    python scripts/build_index.py                # uses config paths
    python scripts/build_index.py --batch 128
Run time: ~1-2 min for 5k rows on CPU; ~10-20 min for the full corpus.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config  # noqa: E402

DESC_CHARS = 1200  # cap description length fed to the embedder


def embed_texts(texts: list[str], model_name: str, batch: int):
    """Encode texts to L2-normalized float32 vectors (lazy heavy imports)."""
    from sentence_transformers import SentenceTransformer  # lazy

    model = SentenceTransformer(model_name)
    return model.encode(
        texts,
        batch_size=batch,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickets", default=config.TICKETS_CSV)
    parser.add_argument("--index-dir", default=config.INDEX_DIR)
    parser.add_argument("--model", default=config.EMBEDDING_MODEL)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    if not os.path.isfile(args.tickets):
        sys.exit(f"ERROR: {args.tickets} not found. Run prepare_dataset.py first.")

    df = pd.read_csv(args.tickets)
    texts = (
        df["title"].fillna("").astype(str)
        + ". "
        + df["description"].fillna("").astype(str).str.slice(0, DESC_CHARS)
    ).tolist()
    print(f"Embedding {len(texts)} tickets with {args.model} ...")

    t0 = time.perf_counter()
    vecs = embed_texts(texts, args.model, args.batch)
    print(f"Embedded in {time.perf_counter() - t0:.1f}s -> shape {vecs.shape}")

    import faiss  # lazy

    index = faiss.IndexFlatIP(vecs.shape[1])  # inner product on unit vectors
    index.add(vecs)

    os.makedirs(args.index_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(args.index_dir, "index.faiss"))
    df.to_csv(os.path.join(args.index_dir, "meta.csv"), index=False)
    print(f"Wrote {index.ntotal} vectors -> {args.index_dir}/index.faiss")
    print(f"Wrote metadata          -> {args.index_dir}/meta.csv")
    print("Done. Next: python scripts/smoke_test.py")


if __name__ == "__main__":
    main()
