import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

DOCUMENTS_PATH = DATA_DIR / "documents.jsonl"
DOC_INDEX_PATH = DATA_DIR / "doc_index.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
GMM_MODEL_PATH = DATA_DIR / "gmm_model.pkl"

OUT_DIR = ROOT / "analysis_outputs"


def load_docs() -> Tuple[List[str], List[str], List[str]]:
    ids: List[str] = []
    labels: List[str] = []
    texts: List[str] = []
    with DOCUMENTS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            ids.append(obj["id"])
            labels.append(obj["label"])
            texts.append(obj["text"])
    return ids, labels, texts


def entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    return float(-(p * np.log(p)).sum())


def top_terms_by_cluster(
    texts: List[str],
    cluster_assignments: np.ndarray,
    n_clusters: int,
    top_n: int = 12,
) -> Dict[int, List[str]]:
    """
    Use TF-IDF to surface interpretable words per cluster.
    We compute the mean TF-IDF vector across docs in each cluster bucket
    (using hard assignment by argmax of soft membership for interpretability).
    """
    vectorizer = TfidfVectorizer(
        max_features=50000,
        stop_words="english",
        min_df=3,
    )
    X = vectorizer.fit_transform(texts)
    vocab = np.array(vectorizer.get_feature_names_out())

    out: Dict[int, List[str]] = {}
    for c in range(n_clusters):
        idx = np.where(cluster_assignments == c)[0]
        if idx.size == 0:
            out[c] = []
            continue
        mean_vec = X[idx].mean(axis=0)
        mean_vec = np.asarray(mean_vec).ravel()
        top_idx = np.argsort(mean_vec)[::-1][:top_n]
        out[c] = vocab[top_idx].tolist()
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ids, labels, texts = load_docs()
    embeddings = np.load(EMBEDDINGS_PATH)
    gmm = joblib.load(GMM_MODEL_PATH)

    probs = gmm.predict_proba(embeddings)
    dominant = probs.argmax(axis=1)
    n_clusters = probs.shape[1]
    maxp = probs.max(axis=1)

    # Cluster size distribution
    sizes = {int(c): int((dominant == c).sum()) for c in range(n_clusters)}

    # Find boundary / uncertain docs:
    # - Highest entropy: most mixed memberships
    # - Lowest max-probability: least confident dominant assignment
    ent = np.array([entropy(p) for p in probs])
    uncertain_by_entropy_idx = np.argsort(ent)[::-1][:15].tolist()
    uncertain_by_lowmax_idx = np.argsort(maxp)[:15].tolist()

    def build_uncertain_examples(indices: List[int]) -> List[dict]:
        examples = []
        for i in indices:
            p = probs[i]
            top3 = np.argsort(p)[::-1][:3]
            examples.append(
                {
                    "id": ids[i],
                    "label": labels[i],
                    "dominant_cluster": int(dominant[i]),
                    "max_prob": float(maxp[i]),
                    "entropy": float(ent[i]),
                    "top_clusters": [
                        {"cluster": int(k), "prob": float(p[k])} for k in top3
                    ],
                    "snippet": " ".join(texts[i].split())[:280],
                }
            )
        return examples

    uncertain_examples = {
        "by_entropy": build_uncertain_examples(uncertain_by_entropy_idx),
        "by_low_max_prob": build_uncertain_examples(uncertain_by_lowmax_idx),
    }

    # Confident examples: highest max-probability within each cluster
    confident_examples = {}
    for c in range(n_clusters):
        idx = np.where(dominant == c)[0]
        if idx.size == 0:
            confident_examples[int(c)] = []
            continue
        top_idx = idx[np.argsort(maxp[idx])[::-1][:5]]
        confident_examples[int(c)] = [
            {
                "id": ids[i],
                "label": labels[i],
                "max_prob": float(maxp[i]),
                "snippet": " ".join(texts[i].split())[:280],
            }
            for i in top_idx
        ]

    # Interpretable top terms per cluster
    terms = top_terms_by_cluster(texts, dominant, n_clusters, top_n=12)

    summary = {
        "n_documents": len(texts),
        "embedding_shape": list(embeddings.shape),
        "n_clusters": int(n_clusters),
        "cluster_sizes": sizes,
        "top_terms_by_cluster": terms,
        "confident_examples_by_cluster": confident_examples,
        "uncertain_boundary_examples": uncertain_examples,
    }

    out_path = OUT_DIR / "cluster_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

