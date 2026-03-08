from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CacheEntry:
    """
    One semantic cache entry.

    We store the original query string, its L2-normalised embedding, the
    dominant cluster id we routed it to, and the pre-computed result payload
    that the API will return on a cache hit.
    """

    query: str
    embedding: np.ndarray  # shape: (d,)
    dominant_cluster: int
    result: str


class SemanticCache:
    """
    Simple in-memory semantic cache.

    Design:
    - Cache entries are bucketed by their dominant cluster id, as given by the
      Gaussian Mixture Model; this keeps lookups fast even as the cache grows.
    - When a new query comes in, we:
        1) compute its cluster probability vector,
        2) choose the dominant cluster,
        3) search only within that bucket for the most similar past query
           using cosine similarity on normalised embeddings.
    - If the best similarity is above a configurable threshold, we treat this
      as a cache hit; otherwise we record a miss and store a new entry.
    """

    def __init__(self, similarity_threshold: float = 0.85) -> None:
        self.similarity_threshold = similarity_threshold
        self.entries_by_cluster: Dict[int, List[CacheEntry]] = {}
        self.hit_count: int = 0
        self.miss_count: int = 0

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        # Embeddings are expected to be L2-normalised, but we still guard
        # against numerical edge cases.
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def lookup(
        self,
        query_embedding: np.ndarray,
        cluster_probs: np.ndarray,
    ) -> Tuple[bool, Optional[CacheEntry], float, int]:
        """
        Try to find a cached result for this query.

        Returns:
            (hit, entry, similarity, dominant_cluster)
        """
        dominant_cluster = int(np.argmax(cluster_probs))
        bucket = self.entries_by_cluster.get(dominant_cluster, [])

        if not bucket:
            self.miss_count += 1
            return False, None, 0.0, dominant_cluster

        best_entry: Optional[CacheEntry] = None
        best_sim: float = -1.0

        for entry in bucket:
            sim = self._cosine_similarity(query_embedding, entry.embedding)
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_entry is not None and best_sim >= self.similarity_threshold:
            self.hit_count += 1
            return True, best_entry, best_sim, dominant_cluster

        self.miss_count += 1
        return False, None, best_sim if best_sim > 0 else 0.0, dominant_cluster

    def store(
        self,
        query: str,
        embedding: np.ndarray,
        dominant_cluster: int,
        result: str,
    ) -> None:
        """
        Insert a new cache entry into the bucket for the given cluster.

        The embedding is assumed to be L2-normalised already.
        """
        entry = CacheEntry(
            query=query,
            embedding=embedding.astype(np.float32),
            dominant_cluster=dominant_cluster,
            result=result,
        )
        self.entries_by_cluster.setdefault(dominant_cluster, []).append(entry)

    def stats(self) -> Dict[str, float]:
        total_entries = sum(len(v) for v in self.entries_by_cluster.values())
        total_lookups = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_lookups) if total_lookups > 0 else 0.0
        return {
            "total_entries": total_entries,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
        }

    def clear(self) -> None:
        self.entries_by_cluster.clear()
        self.hit_count = 0
        self.miss_count = 0

