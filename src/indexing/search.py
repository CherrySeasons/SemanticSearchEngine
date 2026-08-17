from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DB_PATH = Path("data/chroma_db")
COLLECTION_NAME = "wikipedia_chunks"
MODEL_NAME = "all-MiniLM-L6-v2"


class Retriever:
    def __init__(self, db_path=CHROMA_DB_PATH, collection_name=COLLECTION_NAME,
                model_name=MODEL_NAME, search_ef: int | None = None):
        
        self.client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.client.get_collection(collection_name)
        self.model = SentenceTransformer(model_name)

        # search_ef can be overridden per-instance without rebuilding the index
        # (it's a query-time HNSW parameter, unlike construction_ef/M, which
        # are baked into the graph at build time).
        # NOTE: Chroma's collection.modify(metadata=...) raises an error if
        # you re-send hnsw:space/M/construction_ef, even unchanged, since
        # those are immutable post-build. Only pass the one mutable key —
        # the underlying HNSW graph is untouched, just the query-time
        # candidate-list size.
        
        if search_ef is not None:
            self.collection.modify(metadata={"hnsw:search_ef": search_ef})

    def embed_query(self, query: str):
        return self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0].tolist()

    def search(self, query: str, top_k: int = 5):
        query_embedding = self.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "title": results["metadatas"][0][i]["title"],
                # Chroma returns cosine *distance* (1 - cosine_similarity)
                # surface similarity since it's the more intuitive "higher=better" number.
                "score": 1 - results["distances"][0][i],
            })
        return hits


if __name__ == "__main__":
    retriever = Retriever()
    for hit in retriever.search("How do plants make food?", top_k=5):
        print(f"[{hit['score']:.3f}] {hit['title']} :: {hit['text'][:100]}...")
