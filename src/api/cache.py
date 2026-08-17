"""
Phase 3 - Caching
A small, dependency-free LRU cache for repeated (query, top_k) lookups.

Why a custom class instead of functools.lru_cache:
- We want cache hit/miss stats exposed via an endpoint (a decorator alone
    doesn't give you that without extra plumbing).
- We want a strict max memory footprint via maxsize + eviction, not just
    "cache everything forever" (lru_cache's default with maxsize=None).
- It's a clearer talking point for an interview: you can explain the
    eviction policy, not just cite a stdlib decorator.
"""
import threading
from collections import OrderedDict
from typing import Any, Optional


class LRUCache:
    def __init__(self, maxsize: int = 256):
        self.maxsize = maxsize
        self._store: "OrderedDict[str, Any]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _make_key(self, query: str, top_k: int) -> str:
        # Normalize so "Cats" / "cats " / "cats" share a cache entry.
        return f"{query.strip().lower()}::{top_k}"

    def get(self, query: str, top_k: int) -> Optional[Any]:
        key = self._make_key(query, top_k)
        with self._lock:
            if key not in self._store:
                self.misses += 1
                return None
            self._store.move_to_end(key)  # mark as most-recently-used
            self.hits += 1
            return self._store[key]

    def set(self, query: str, top_k: int, value: Any) -> None:
        key = self._make_key(query, top_k)
        with self._lock:
            self._store[key] = value
            self._store.move_to_end(key)
            if len(self._store) > self.maxsize:
                self._store.popitem(last=False)  # evict least-recently-used

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total else 0.0
            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(hit_rate, 3),
                "size": len(self._store),
                "maxsize": self.maxsize,
            }