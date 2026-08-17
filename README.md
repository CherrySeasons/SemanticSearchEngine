# Semantic Search Engine — Simple English Wikipedia

A semantic search engine built from scratch over the full Simple English Wikipedia corpus (~351K chunks): a data pipeline that chunks and embeds raw article text, an ANN-indexed retrieval core, and a FastAPI service with caching wrapped around it.

Built as a portfolio project to demonstrate both the ML side (embeddings, vector search, retrieval evaluation) and the SDE side (API design, caching, containerization) of building a real search system — deliberately scoped for clarity over architectural sophistication.

## How it works

```
Raw Wikipedia dump
      │
      ▼
┌─────────────────┐     paragraph-aware chunking (~200 words, 40-word overlap)
│   Ingestion      │ ──────────────────────────────────────────────────────►  chunks.jsonl
└─────────────────┘
      │
      ▼
┌─────────────────┐     all-MiniLM-L6-v2, L2-normalized
│   Embeddings     │ ──────────────────────────────────────────────────────►  embeddings.npy
└─────────────────┘
      │
      ▼
┌─────────────────┐     ChromaDB + HNSW (cosine, M=16, ef_construction=200)
│   Indexing       │ ──────────────────────────────────────────────────────►  chroma_db/
└─────────────────┘
      │
      ▼
┌─────────────────┐     /search → embed query → top-k ANN lookup → ranked results
│   FastAPI + LRU  │
│   cache layer    │
└─────────────────┘
```

## Tech stack

| Layer | Choice |
|---|---|
| Dataset | HuggingFace `wikimedia/wikipedia` (`20231101.simple`) |
| Chunking | Paragraph-aware sliding window, ~200 words, 40-word overlap |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Vector store | ChromaDB, persistent HNSW index |
| API | FastAPI + Pydantic v2 |
| Caching | Hand-rolled thread-safe LRU (exposes hit/miss stats via `/cache/stats`) |
| Containerization | Docker / docker-compose |

## Project structure

```
SemanticSearchEngine/
├── src/
│   ├── ingestion/         # download, clean, chunk raw Wikipedia dump
│   ├── embeddings/        # encode chunks with all-MiniLM-L6-v2
│   ├── indexing/          # build & query the Chroma HNSW index
│   ├── api/                # FastAPI app, schemas, LRU cache
│   └── evaluation/         # Recall@k / MRR eval harness
├── eval/
│   └── queries.json       # hand-labeled query set for evaluation
├── data/                  # generated locally, not committed (see Setup)
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Setup

```bash
git clone https://github.com/CherrySeasons/SemanticSearchEngine.git
cd SemanticSearchEngine
pip install -r requirements.txt
```

Data isn't committed to the repo (the full embedded index is several hundred MB), so it's rebuilt locally in three steps:

```bash
# 1. Download, clean, and chunk the corpus
python src/ingestion/load_and_chunk.py

# 2. Generate embeddings for every chunk
python src/embeddings/generate_embeddings.py

# 3. Build the HNSW vector index
python src/indexing/build_index.py
```

## Usage

**Interactive CLI** — sanity-check retrieval by hand:
```bash
python src/indexing/cli.py --top-k 5
```

**API server:**
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/search \
     -H "Content-Type: application/json" \
     -d '{"query": "How do plants make food?", "top_k": 5}'
```

**Docker:**
```bash
docker compose up --build
```

## API reference

| Endpoint | Method | Description |
|---|---|---|
| `/search` | POST | `{"query": str, "top_k": int}` → ranked chunk results |
| `/health` | GET | Service status + indexed collection size |
| `/cache/stats` | GET | LRU cache hit/miss stats |

## Design notes

- **Chunking**: paragraph-aware with overlap, rather than fixed-token windows, to avoid severing sentences mid-thought at chunk boundaries.
- **HNSW tuning**: `M=16` and `construction_ef=200` are set at build time and are immutable afterward; `search_ef` (the query-time candidate-list size) can be tuned live without rebuilding the index — the main recall/latency knob exposed at query time.
- **Caching**: a custom LRU rather than `functools.lru_cache`, so cache hit/miss rates are inspectable via `/cache/stats` rather than opaque.

## Evaluation

Retrieval quality is measured with Recall@k and Mean Reciprocal Rank (MRR) against a hand-labeled query set (`eval/queries.json`). Run it with:

```bash
python src/evaluation/run_eval.py --k-values 1 3 5 10
```

*Results table to be added here once the labeled query set and evaluation run are complete.*

## Roadmap

- [x] Data ingestion & chunking pipeline
- [x] HNSW-indexed vector retrieval
- [x] FastAPI serving layer with caching
- [ ] Cloud deployment

## License

MIT — see [LICENSE](LICENSE).
