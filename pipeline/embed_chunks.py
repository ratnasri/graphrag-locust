"""
Embed page-level bulletin chunks with all-MiniLM-L6-v2.
Reads data/graph/chunks.json, writes data/graph/chunk_embeddings.npy.
"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = Path("data/graph/chunks.json")
OUT_FILE    = Path("data/graph/chunk_embeddings.npy")
MODEL_NAME  = "all-MiniLM-L6-v2"
BATCH_SIZE  = 64


def main():
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    texts = [c["text"] for c in chunks]

    model = SentenceTransformer(MODEL_NAME)
    print(f"Embedding {len(texts)} chunks (batch size {BATCH_SIZE})...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    np.save(OUT_FILE, embeddings)
    print(f"Saved {embeddings.shape} embeddings -> {OUT_FILE}")


if __name__ == "__main__":
    main()
