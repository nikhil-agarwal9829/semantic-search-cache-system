from __future__ import annotations

import json
from typing import Optional

from fastapi import FastAPI

from .cache import SemanticCache
from .core import embed_query, infer_cluster_probs, semantic_search
from .models import CacheStats, QueryRequest, QueryResponse


app = FastAPI(title="Trademarkia Semantic Search & Cache Demo")

# Global in-memory semantic cache shared across requests.
# The similarity_threshold is the key tunable parameter: higher values mean
# fewer but stricter cache hits; lower values increase hit rate but risk
# reusing results for less similar queries.
semantic_cache = SemanticCache(similarity_threshold=0.85)


@app.post("/query", response_model=QueryResponse)
def query_endpoint(payload: QueryRequest) -> QueryResponse:
    """
    Accept a natural language query, embed it, route it through the semantic
    cache, and either reuse a cached result or compute a fresh semantic search.
    """
    query = payload.query.strip()
    if not query:
        # Simple guard to avoid empty queries; in a production service you
        # might want more detailed error handling.
        return QueryResponse(
            query=query,
            cache_hit=False,
            matched_query=None,
            similarity_score=None,
            result="Empty query.",
            dominant_cluster=-1,
        )

    q_embedding = embed_query(query)
    cluster_probs = infer_cluster_probs(q_embedding)

    hit, entry, sim, dominant_cluster = semantic_cache.lookup(
        query_embedding=q_embedding,
        cluster_probs=cluster_probs,
    )

    if hit and entry is not None:
        return QueryResponse(
            query=query,
            cache_hit=True,
            matched_query=entry.query,
            similarity_score=round(sim, 4),
            result=entry.result,
            dominant_cluster=dominant_cluster,
        )

    # Cache miss: run semantic search over the corpus and store the result.
    search_results, _ = semantic_search(q_embedding, top_k=5)
    result_str = json.dumps(search_results, ensure_ascii=False, indent=2)

    semantic_cache.store(
        query=query,
        embedding=q_embedding,
        dominant_cluster=dominant_cluster,
        result=result_str,
    )

    return QueryResponse(
        query=query,
        cache_hit=False,
        matched_query=None,
        similarity_score=None,
        result=result_str,
        dominant_cluster=dominant_cluster,
    )


@app.get("/cache/stats", response_model=CacheStats)
def cache_stats_endpoint() -> CacheStats:
    stats = semantic_cache.stats()
    return CacheStats(
        total_entries=stats["total_entries"],
        hit_count=stats["hit_count"],
        miss_count=stats["miss_count"],
        hit_rate=stats["hit_rate"],
    )


@app.delete("/cache")
def cache_clear_endpoint() -> dict:
    semantic_cache.clear()
    return {"status": "ok", "message": "Cache cleared and statistics reset."}

