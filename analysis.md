## Analysis (clustering + cache)

This file documents the key design decisions and provides **evidence** that:
- the learned clusters are semantically meaningful, and
- the semantic cache behavior is controlled by one tunable parameter (the similarity threshold).

All numbers below were generated locally from the saved artifacts in `data/` using:
- `python scripts/analyze.py` → `analysis_outputs/cluster_summary.json`
- `python scripts/cache_threshold_experiment.py` → `analysis_outputs/cache_threshold_experiment.json`

---

## 1) Corpus preparation choices

The raw 20 Newsgroups files contain:
- RFC822-like **headers** (routing, From/Path/Organization, etc.)
- quoted reply content (lines starting with `>`)

In `scripts/prepare_data.py` we:
- drop the header block (everything before the first blank line),
- remove `>` quote lines,
- skip very short cleaned documents (< 50 chars).

**Reasoning**: headers/quotes dominate token frequency but contribute little topic meaning, so removing them makes embeddings focus on the actual message content.

Result: **19,868** cleaned documents.

---

## 2) Embeddings + vector retrieval

### Model choice

We use `sentence-transformers/all-MiniLM-L6-v2` because it is:
- lightweight and fast on CPU,
- high quality for semantic similarity,
- a good fit for ~20k documents.

### Persisted artifacts

After running `python scripts/prepare_data.py`, we have:
- `data/embeddings.npy`: shape **(19868, 384)** (L2-normalised)
- `data/nn_index.pkl`: cosine NearestNeighbors index (our “vector DB”)

Nearest-neighbour results are returned as top-k document ids + labels + similarity + a short snippet.

---

## 3) Fuzzy clustering (soft memberships)

### Why Gaussian Mixture Model (GMM)

The requirement is “distribution, not label”. A GMM provides:
\[
p(\text{cluster}=k \mid \text{document})
\]
for every document, which is a direct soft assignment.

### Choosing number of clusters (evidence, not convenience)

We tested candidate component counts on a 5k subset using BIC:
- K ∈ {10, 15, 20, 25, 30}

The best (lowest) BIC was at **K=10** for this embedding space and preprocessing setup, so we fit a full GMM with **10 components** on all documents.

### What “lives” in clusters

We extracted interpretability hints by computing top TF‑IDF terms for documents whose dominant cluster is k.
Examples (from `analysis_outputs/cluster_summary.json`):

- **Cluster 0**: `car`, `bike`, `cars`, `engine`, `bmw` … (autos / motorcycles themes)
- **Cluster 4**: `drive`, `card`, `scsi`, `windows`, `mac`, `disk`, `dos` … (hardware / storage / OS)
- **Cluster 6**: `game`, `team`, `hockey`, `baseball`, `season`, `players` … (sports)
- **Cluster 7**: `sale`, `price`, `shipping`, `offer`, `sell` … (marketplace / misc.forsale)
- **Cluster 8**: `encryption`, `key`, `clipper`, `nsa`, `escrow`, `public` … (cryptography)
- **Cluster 9**: `god`, `jesus`, `bible`, `christian`, `church`, `believe` … (religion)

These term clusters align with obvious semantic “centers of gravity”, even though the original dataset labels overlap.

### Boundary / uncertainty behavior (important observation)

We tried to surface boundary documents using:
- high entropy \(H(p)\)
- low max probability \(\max_k p_k\)

In this run, the GMM posterior distributions are generally **very peaky** (even the “least confident” examples still had dominant probabilities around ~0.95+).

**What this reveals**:
- In this embedding space, a full‑covariance GMM can become highly confident.
- That doesn’t invalidate fuzzy clustering (we still have a distribution), but it means “boundary cases” are rarer/less ambiguous than expected.

If you want more visibly “mixed membership” documents, two practical knobs are:
- increase K and re-check BIC/qualitative coherence, or
- regularise the mixture (e.g. different covariance type), then re-run the same analysis script.

---

## 4) Semantic cache (first principles)

### Core mechanism

Traditional caches match exact strings. We instead:
- embed the query,
- compute its GMM cluster probabilities,
- route it to a dominant cluster bucket,
- compare only against cached embeddings in that bucket using cosine similarity,
- declare a hit if similarity ≥ **threshold τ**.

This uses clustering to keep cache lookup efficient as it grows, because we avoid scanning all entries globally.

### The key tunable decision: similarity threshold \( \tau \)

We ran a controlled simulation that generates 3 “paraphrase-like” variants per base text snippet (450 base docs → 1350 queries), and evaluated different thresholds.

Results:

| τ (threshold) | hit_rate | final_cache_entries |
|---:|---:|---:|
| 0.70 | 0.499 | 677 |
| 0.80 | 0.407 | 800 |
| 0.85 | 0.368 | 853 |
| 0.90 | 0.318 | 921 |
| 0.95 | 0.252 | 1010 |

**What this reveals**:
- Lower τ increases reuse (higher hit rate) and keeps the cache smaller, but risks matching less-related queries.
- Higher τ is stricter (lower hit rate), growing the cache faster, but increases precision of reuse.

In the live API, τ is set in `app/main.py`:
- `SemanticCache(similarity_threshold=0.85)`

---

## 5) How to reproduce the evidence

```powershell
.venv\Scripts\Activate.ps1
python scripts\prepare_data.py
python scripts\analyze.py
python scripts\cache_threshold_experiment.py
```

Outputs:
- `analysis_outputs/cluster_summary.json`
- `analysis_outputs/cache_threshold_experiment.json`

