"""
Phase 1 - Embeddings
Encode every chunk in data/processed/chunks.jsonl with all-MiniLM-L6-v2
and save embeddings (numpy) + aligned metadata (jsonl) to disk.

Usage:
    python src/embeddings/generate_embeddings.py
"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

INPUT_PATH = Path("data/processed/chunks.jsonl")
OUTPUT_EMB_PATH = Path("data/processed/embeddings.npy")
OUTPUT_META_PATH = Path("data/processed/metadata.jsonl")

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 256


def load_chunks(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    records = load_chunks(INPUT_PATH)
    texts = [r["text"]for r in records]
    print(f"Loaded {len(texts)} chunks from {INPUT_PATH}")

    model = SentenceTransformer(MODEL_NAME)
    print(f"Embedding with {MODEL_NAME} (dim={model.get_sentence_embedding_dimension()})...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so cosine similarity == dot product (matches Chroma's default)
    )

    np.save(OUTPUT_EMB_PATH, embeddings)
    with open(OUTPUT_META_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved embeddings {embeddings.shape} -> {OUTPUT_EMB_PATH}")
    print(f"Saved metadata -> {OUTPUT_META_PATH}")


if __name__ == "__main__":
    main()
