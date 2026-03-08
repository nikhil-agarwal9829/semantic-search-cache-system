import os
import json
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


DATASET_ROOT = Path(__file__).resolve().parents[1] / "twenty+newsgroups" / "20_newsgroups"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"
DOCS_PATH = OUTPUT_DIR / "documents.jsonl"
EMBEDDINGS_PATH = OUTPUT_DIR / "embeddings.npy"
DOC_INDEX_PATH = OUTPUT_DIR / "doc_index.json"
GMM_MODEL_PATH = OUTPUT_DIR / "gmm_model.pkl"
GMM_MODEL_SELECTION_PATH = OUTPUT_DIR / "gmm_model_selection.json"
NN_INDEX_PATH = OUTPUT_DIR / "nn_index.pkl"


def read_raw_document(path: Path) -> str:
    """
    Read a single raw 20 Newsgroups file as text.

    We open with latin-1 to be robust to mixed encodings in this 1990s-era corpus.
    """
    with path.open("r", encoding="latin-1", errors="ignore") as f:
        return f.read()


def strip_headers_and_quotes(text: str) -> str:
    """
    Basic cleaning tailored to this dataset:
    - Drop the RFC822-style header block (everything before the first blank line),
      because it mostly contains routing metadata (From, Path, Organization, etc.)
      that dominates the vocabulary but adds little semantic meaning for topics.
    - Remove lines starting with '>' which are usually quoted replies, to focus on
      the author's own message instead of duplicated context.
    """
    # Remove header block (up to first blank line)
    parts = text.split("\n\n", 1)
    body = parts[1] if len(parts) == 2 else text

    # Drop simple quoted lines
    cleaned_lines = []
    for line in body.splitlines():
        if line.strip().startswith(">"):
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()
    return cleaned


def collect_documents():
    """
    Walk the 20_newsgroups directory and build a simple list of documents:
    each with an id, cleaned text, original newsgroup label, and file path.

    Result is written to data/documents.jsonl so later steps (embeddings,
    clustering, FastAPI service) can reuse the same corpus snapshot.
    """
    assert DATASET_ROOT.exists(), f"Dataset root not found at {DATASET_ROOT}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    count = 0

    with DOCS_PATH.open("w", encoding="utf-8") as out_f:
        for group_dir, _, files in os.walk(DATASET_ROOT):
            group_dir = Path(group_dir)
            if group_dir == DATASET_ROOT:
                # top-level container, skip
                continue

            label = group_dir.name  # e.g. 'alt.atheism'

            for fname in files:
                file_path = group_dir / fname
                raw = read_raw_document(file_path)
                cleaned = strip_headers_and_quotes(raw)

                # Skip extremely short or empty documents; they add noise but
                # almost no semantic information to the embedding space.
                if len(cleaned) < 50:
                    continue

                doc = {
                    "id": f"{label}/{fname}",
                    "label": label,
                    "path": str(file_path),
                    "text": cleaned,
                }
                out_f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                count += 1

    print(f"Wrote {count} cleaned documents to {DOCS_PATH}")


def load_documents() -> Tuple[List[str], List[str], List[str]]:
    """
    Load the cleaned documents.jsonl file into parallel lists.

    We keep ids, labels, and texts aligned by index so the same order can be
    used for embeddings and later for semantic search.
    """
    ids: List[str] = []
    labels: List[str] = []
    texts: List[str] = []

    with DOCS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            ids.append(obj["id"])
            labels.append(obj["label"])
            texts.append(obj["text"])

    return ids, labels, texts


def embed_documents(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    """
    Compute sentence-transformer embeddings for all cleaned documents.

    Model choice: all-MiniLM-L6-v2 (384-dim, lightweight, good semantic quality).
    We L2-normalize the resulting matrix so that cosine similarity reduces to a
    simple dot product, which makes later nearest-neighbour search fast.
    """
    ids, labels, texts = load_documents()

    print(f"Loaded {len(texts)} cleaned documents from {DOCS_PATH}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save a compact index file so we can later map embedding rows back to docs
    with DOC_INDEX_PATH.open("w", encoding="utf-8") as f:
        json.dump({"ids": ids, "labels": labels}, f, ensure_ascii=False, indent=2)

    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print("Computing embeddings (this may take a few minutes)...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    # L2-normalise so each vector has unit length
    embeddings = normalize(embeddings, norm="l2")
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Saved embeddings to {EMBEDDINGS_PATH} with shape {embeddings.shape}")

    return embeddings


def select_gmm_components(embeddings: np.ndarray, candidates=(10, 15, 20, 25, 30)) -> int:
    """
    Fit Gaussian Mixture Models with different numbers of components on a
    subset of the data and choose the best K by BIC (lower is better).

    This is our evidence-based way of deciding the number of clusters, instead
    of hard-coding 20 just because the dataset has 20 labels.
    """
    n_samples = embeddings.shape[0]
    subset_size = min(5000, n_samples)
    rng = np.random.default_rng(42)
    subset_idx = rng.choice(n_samples, size=subset_size, replace=False)
    subset = embeddings[subset_idx]

    results = []
    best_k = None
    best_bic = None

    for k in candidates:
        print(f"Fitting GMM with {k} components on subset of size {subset_size}...")
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            max_iter=200,
            random_state=42,
        )
        gmm.fit(subset)
        bic = gmm.bic(subset)
        results.append({"k": k, "bic": float(bic)})
        print(f"k={k}, BIC={bic:.2f}")

        if best_bic is None or bic < best_bic:
            best_bic = bic
            best_k = k

    with GMM_MODEL_SELECTION_PATH.open("w", encoding="utf-8") as f:
        json.dump({"results": results, "best_k": best_k}, f, indent=2)

    print(f"Best number of components by BIC: k={best_k}")
    return int(best_k)


def fit_full_gmm(embeddings: np.ndarray, n_components: int) -> GaussianMixture:
    """
    Fit a Gaussian Mixture Model on the full embedding matrix with the chosen
    number of components, then persist it for use at query time.

    The GMM gives us a soft assignment p(cluster | document) for each point,
    which we will later use both for analysis and for routing queries into
    cache buckets.
    """
    print(f"Fitting full GMM with {n_components} components on {embeddings.shape[0]} documents...")
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        max_iter=200,
        random_state=42,
    )
    gmm.fit(embeddings)
    joblib.dump(gmm, GMM_MODEL_PATH)
    print(f"Saved GMM model to {GMM_MODEL_PATH}")
    return gmm


def build_nearest_neighbors_index(embeddings: np.ndarray) -> None:
    """
    Build a simple nearest-neighbours index over the embeddings using
    scikit-learn's NearestNeighbors as our lightweight 'vector database'.

    We use cosine distance and the brute-force algorithm, which is perfectly
    adequate at this scale (~20k documents) and keeps dependencies simple.
    """
    print("Fitting NearestNeighbors index over embeddings...")
    nn = NearestNeighbors(
        n_neighbors=20,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    nn.fit(embeddings)
    joblib.dump(nn, NN_INDEX_PATH)
    print(f"Saved NearestNeighbors index to {NN_INDEX_PATH}")


def main():
    # 1) Collect and clean raw documents if needed.
    if not DOCS_PATH.exists():
        print("No cleaned documents found, collecting from raw dataset...")
        collect_documents()
    else:
        print(f"Using existing cleaned documents at {DOCS_PATH}")

    # 2) Compute or load embeddings.
    if EMBEDDINGS_PATH.exists():
        print(f"Loading existing embeddings from {EMBEDDINGS_PATH}")
        embeddings = np.load(EMBEDDINGS_PATH)
    else:
        embeddings = embed_documents()

    # 3) Choose number of GMM components and fit full model.
    if GMM_MODEL_PATH.exists() and GMM_MODEL_SELECTION_PATH.exists():
        print(f"GMM model already exists at {GMM_MODEL_PATH}, skipping refit.")
    else:
        best_k = select_gmm_components(embeddings)
        fit_full_gmm(embeddings, best_k)

    # 4) Build nearest-neighbour index.
    if NN_INDEX_PATH.exists():
        print(f"Nearest-neighbour index already exists at {NN_INDEX_PATH}, skipping refit.")
    else:
        build_nearest_neighbors_index(embeddings)


if __name__ == "__main__":
    main()

