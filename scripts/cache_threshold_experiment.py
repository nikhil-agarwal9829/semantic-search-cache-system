import json
import random
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DOCUMENTS_PATH = DATA_DIR / "documents.jsonl"
GMM_MODEL_PATH = DATA_DIR / "gmm_model.pkl"

OUT_DIR = ROOT / "analysis_outputs"


def load_texts(limit: int = 400) -> List[str]:
    texts: List[str] = []
    with DOCUMENTS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
            if len(texts) >= limit:
                break
    return texts


def make_variants(text: str) -> List[str]:
    """
    Create cheap, deterministic "paraphrase-like" variants without any external LLM:
    - short extract
    - short extract with some words removed
    - same with a short prompt prefix
    These are not true paraphrases, but they are semantically close enough to
    stress-test the cache similarity threshold in a controlled way.
    """
    words = text.split()
    base = " ".join(words[:22])
    drop = " ".join([w for i, w in enumerate(words[:28]) if (i % 3) != 0])
    prefix = "please explain: " + base
    return [base, drop, prefix]


def embed(model: SentenceTransformer, s: str) -> np.ndarray:
    v = model.encode([s], convert_to_numpy=True, normalize_embeddings=False)[0]
    v = normalize(v.reshape(1, -1), norm="l2")[0]
    return v.astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def run_simulation(threshold: float, queries: List[str], gmm, model: SentenceTransformer) -> Dict[str, float]:
    """
    Simulate the API cache behavior with a cluster-bucketed semantic cache.
    We store (embedding, query) per dominant cluster and reuse the most similar
    query if similarity >= threshold.
    """
    buckets: Dict[int, List[Dict[str, object]]] = {}
    hit = 0
    miss = 0

    for q in queries:
        v = embed(model, q)
        probs = gmm.predict_proba(v.reshape(1, -1))[0]
        c = int(np.argmax(probs))

        bucket = buckets.get(c, [])
        if not bucket:
            miss += 1
            buckets.setdefault(c, []).append({"query": q, "vec": v})
            continue

        best_sim = -1.0
        best = None
        for item in bucket:
            sim = cosine(v, item["vec"])  # type: ignore[arg-type]
            if sim > best_sim:
                best_sim = sim
                best = item

        if best is not None and best_sim >= threshold:
            hit += 1
        else:
            miss += 1
            buckets.setdefault(c, []).append({"query": q, "vec": v})

    total = hit + miss
    return {
        "threshold": threshold,
        "total_queries": total,
        "hit_count": hit,
        "miss_count": miss,
        "hit_rate": (hit / total) if total else 0.0,
        "final_cache_entries": sum(len(v) for v in buckets.values()),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    texts = load_texts(limit=450)
    variant_queries: List[str] = []
    for t in texts:
        variant_queries.extend(make_variants(t))

    random.seed(42)
    random.shuffle(variant_queries)

    gmm = joblib.load(GMM_MODEL_PATH)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    thresholds = [0.70, 0.80, 0.85, 0.90, 0.95]
    results = [run_simulation(t, variant_queries, gmm, model) for t in thresholds]

    out_path = OUT_DIR / "cache_threshold_experiment.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "n_base_documents_used": len(texts),
                "queries_per_document": 3,
                "total_queries": len(variant_queries),
                "thresholds_tested": thresholds,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

