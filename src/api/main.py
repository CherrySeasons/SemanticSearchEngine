"""
Phase 3 - API
FastAPI service wrapping the Phase 2 Retriever, with request validation,
an in-memory LRU cache, and basic error handling.

Run locally:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

Then:
    curl -X POST http://localhost:8000/search \
        -H "Content-Type: application/json" \
        -d '{"query": "How do plants make food?", "top_k": 5}'
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.api.cache import LRUCache
from src.api.schemas import HealthResponse, SearchRequest, SearchResponse, SearchResult
from src.indexing.search import Retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("semantic_search_api")

CACHE_MAXSIZE = 512

# Populated at startup (see lifespan below) so the model + Chroma client
# load exactly once per process, not per-request.
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading retriever (model + Chroma index)...")
    state["retriever"] = Retriever()
    state["cache"] = LRUCache(maxsize=CACHE_MAXSIZE)
    logger.info("Retriever loaded. Collection size: %d", state["retriever"].collection.count())
    # <<< ========================== SETUP (Code above yield) ========================== <<<
    
    yield
    
    # >>> ========================== CLEANUP (Code after yield) ========================== >>>
    state.clear()


app = FastAPI(title="Semantic Search API", version="1.0.0", lifespan=lifespan)

@app.get("/health", response_model=HealthResponse)
def health():
    retriever = state.get("retriever")  
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not yet initialized")
    return HealthResponse(status="ok", collection_size=retriever.collection.count())


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    retriever = state.get("retriever")
    cache = state.get("cache")
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not yet initialized")

    cached_results = cache.get(request.query, request.top_k)
    if cached_results is not None:
        return SearchResponse(
            query=request.query,
            top_k=request.top_k,
            cached=True,
            results=[SearchResult(**r) for r in cached_results],
        )

    try:
        hits = retriever.search(request.query, top_k=request.top_k)
    except Exception:
        logger.exception("Search failed for query=%r", request.query)
        raise HTTPException(status_code=500, detail="Search failed. Please try again.")

    cache.set(request.query, request.top_k, hits)

    return SearchResponse(
        query=request.query,
        top_k=request.top_k,
        cached=False,
        results=[SearchResult(**r) for r in hits],
    )


@app.get("/cache/stats")
def cache_stats():
    cache = state.get("cache")
    if cache is None:
        raise HTTPException(status_code=503, detail="Cache not yet initialized")
    return cache.stats()