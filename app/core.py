from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

DOCUMENTS_PATH = DATA_DIR / "documents.jsonl"
DOC_INDEX_PATH = DATA_DIR / "doc_index.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
GMM_MODEL_PATH = DATA_DIR / "gmm_model.pkl"
NN_INDEX_PATH = DATA_DIR / "nn_index.pkl"


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Load the sentence-transformer model once and cache it in memory.

    Using lru_cache ensures FastAPI workers reuse the same model instance
    instead of reloading weights on every request.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return SentenceTransformer(model_name)


@lru_cache(maxsize=1)
def get_corpus_embeddings() -> np.ndarray:
    return np.load(EMBEDDINGS_PATH)


@lru_cache(maxsize=1)
def get_doc_index() -> Dict[str, List[str]]:
    with DOC_INDEX_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def get_doc_texts() -> List[str]:
    """
    Load the full cleaned document texts into memory in the same order as the
    embeddings. This allows us to return short snippets for semantic search
    results without hitting disk on every query.
    """
    texts: List[str] = []
    with DOCUMENTS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
    return texts


@lru_cache(maxsize=1)
def get_gmm_model():
    return joblib.load(GMM_MODEL_PATH)


@lru_cache(maxsize=1)
def get_nn_index():
    return joblib.load(NN_INDEX_PATH)


def embed_query(text: str) -> np.ndarray:
    """
    Embed a single query string and L2-normalise it to match the corpus
    embeddings used for nearest-neighbour search and clustering.
    """
    model = get_embedding_model()
    vec = model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=False,
    )[0]
    vec = normalize(vec.reshape(1, -1), norm="l2")[0]
    return vec.astype(np.float32)


def infer_cluster_probs(query_embedding: np.ndarray) -> np.ndarray:
    """
    Compute the soft cluster assignment p(cluster | query) using the GMM.
    """
    gmm = get_gmm_model()
    probs = gmm.predict_proba(query_embedding.reshape(1, -1))[0]
    return probs


def semantic_search(
    query_embedding: np.ndarray,
    top_k: int = 5,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Run a nearest-neighbour search in embedding space and return a small list
    of top-k matching documents with ids, labels, scores, and short snippets.
    """
    nn = get_nn_index()
    embeddings = get_corpus_embeddings()
    doc_index = get_doc_index()
    doc_texts = get_doc_texts()

    distances, indices = nn.kneighbors(
        query_embedding.reshape(1, -1),
        n_neighbors=top_k,
        return_distance=True,
    )

    results: List[Dict[str, Any]] = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        # For cosine distance, similarity = 1 - distance.
        similarity = float(1.0 - dist)
        doc_id = doc_index["ids"][idx]
        label = doc_index["labels"][idx]
        text = doc_texts[idx]
        snippet = " ".join(text.split())[:300]

        results.append(
            {
                "rank": rank,
                "id": doc_id,
                "label": label,
                "similarity": round(similarity, 4),
                "snippet": snippet,
            }
        )

    # For convenience we also return the index of the single best document.
    best_doc_idx = int(indices[0][0])
    return results, best_doc_idx

