"""
Phase 3 - Schemas
Pydantic request/response models for the /search endpoint.
"""
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500,
                        description="Natural language search query")
    top_k: int = Field(default=5, ge=1, le=50,
                        description="Number of results to return")


class SearchResult(BaseModel):
    chunk_id: str
    title: str
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    top_k: int
    cached: bool
    results: list[SearchResult]


class HealthResponse(BaseModel):
    status: str
    collection_size: int