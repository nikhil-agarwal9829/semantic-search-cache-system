from typing import Optional

from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: Optional[str]
    similarity_score: Optional[float]
    result: str
    dominant_cluster: int


class CacheStats(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float

