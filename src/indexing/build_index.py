import json
from pathlib import Path

import chromadb
import numpy as np

EMBEDDINGS_PATH = Path("data/processed/embeddings.npy")
METADATA_PATH = Path("data/processed/metadata.jsonl")
CHROMA_DB_PATH = Path("data/chroma_db")
COLLECTION_NAME = "wikipedia_chunks"

BATCH_SIZE = 5000

HNSW_CONFIG = {
    "hnsw:space": "cosine",
    "hnsw:construction_ef": 200,
    "hnsw:M": 16,
    "hnsw:search_ef": 100,
}


def load_metadata(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_index(recreate: bool = True):
    embeddings = np.load(EMBEDDINGS_PATH)
    metadata = load_metadata(METADATA_PATH)
    assert len(embeddings) == len(metadata), (
        f"Embeddings ({len(embeddings)}) and metadata ({len(metadata)}) row counts don't match"
    )

    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

    if recreate:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata=HNSW_CONFIG,
    )

    print(f"Indexing {len(embeddings)} vectors into Chroma (dim={embeddings.shape[1]})...")
    for start in range(0, len(embeddings), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(embeddings))
        batch_meta = metadata[start:end]

        collection.add(
            ids=[m["chunk_id"] for m in batch_meta],
            embeddings=embeddings[start:end].tolist(),
            documents=[m["text"] for m in batch_meta],
            metadatas=[
                {"title": m["title"], "article_id": m["article_id"], "chunk_index": m["chunk_index"]}
                for m in batch_meta
            ],
        )
        print(f"  added rows {start}-{end}")

    print(f"Index built. Collection '{COLLECTION_NAME}' has {collection.count()} vectors.")
    print(f"HNSW config: {HNSW_CONFIG}")
    return collection


if __name__ == "__main__":
    build_index()