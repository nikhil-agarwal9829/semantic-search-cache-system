## Trademarkia – AI/ML Engineer Task (20 Newsgroups)

This project implements a lightweight semantic search system over the **20 Newsgroups** corpus with:
- **Vector embeddings + vector retrieval**
- **Fuzzy clustering** (soft membership distributions, not hard labels)
- **Semantic cache** built from first principles (no Redis/Memcached)
- **FastAPI** service exposing the cache as a live API

The dataset is expected to be present at `twenty+newsgroups/20_newsgroups/`.
Note: the raw dataset folder is **ignored by git** (`.gitignore`) to keep the repository small. Download/unpack it locally before running `scripts/prepare_data.py`.

### Repo structure

- **`scripts/prepare_data.py`**: offline pipeline (clean → embed → cluster → build vector index)
- **`data/`**: persisted artifacts used by the API at runtime
  - `documents.jsonl` (cleaned corpus)
  - `doc_index.json` (ids/labels aligned with embedding rows)
  - `embeddings.npy`
  - `gmm_model.pkl`
  - `gmm_model_selection.json`
  - `nn_index.pkl`
- **`app/`**: FastAPI service + cache
  - `main.py` endpoints
  - `cache.py` semantic cache implementation
  - `core.py` model loading + embedding + search utilities
- **`process_flow.txt`**: step-by-step log (what/why/result)
- **`analysis.md`**: clustering + cache behavior analysis (evidence-based)

---

## Local setup (venv)

### 1) Create and activate venv

PowerShell (Windows):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

### 3) Prepare artifacts (one-time)

This creates `data/` artifacts used by the API.

Prerequisite: download and unpack the dataset into:
- `twenty+newsgroups/20_newsgroups/`

```powershell
python scripts\prepare_data.py
```

### 4) Start the API

```powershell
uvicorn app.main:app --reload
```

Open Swagger UI:
- `http://127.0.0.1:8000/docs`

---

## API endpoints

### POST `/query`

Request body:

```json
{ "query": "How do graphics cards work in PCs?" }
```

Response:
- `cache_hit`: whether a semantic cache match was found
- `matched_query`: the stored query we matched against (on hit)
- `similarity_score`: cosine similarity to matched query (on hit)
- `result`: JSON string of top-k retrieved docs (id/label/similarity/snippet)
- `dominant_cluster`: the cluster used for cache routing

### GET `/cache/stats`

Returns cache state:

```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}
```

### DELETE `/cache`

Clears cache and resets stats.

---

## Docker (optional)

### Build image

```bash
docker build -t trademarkia-semantic-search .
```

### Run container

```bash
docker run -p 8000:8000 trademarkia-semantic-search
```

Then open:
- `http://127.0.0.1:8000/docs`
1. Sending a Query (POST /query)

Submitting a natural language query to the semantic search API.

<img width="1894" height="819" alt="Sending query request" src="https://github.com/user-attachments/assets/e5252b05-dabb-4e50-b7f5-e71dba25afb7" />
2. Server Response

The server processes the query and returns the most relevant documents along with similarity scores and cluster information.

<img width="1812" height="859" alt="Server response with semantic results" src="https://github.com/user-attachments/assets/98d23c13-84b8-49dd-a54c-307f06f087e4" />
3. Viewing Cache Statistics (GET /cache/stats)

Retrieving cache statistics to monitor cache entries, hits, misses, and overall hit rate.

<img width="1817" height="853" alt="Cache statistics endpoint" src="https://github.com/user-attachments/assets/7e0c49b1-fd08-44fe-a5a5-f69b7033d24e" />
4. Clearing the Cache (DELETE /cache)

Resetting the semantic cache and clearing all stored queries.

<img width="1839" height="837" alt="Cache clear endpoint" src="https://github.com/user-attachments/assets/9375e30e-3dbd-49f1-a9c1-b3b6528383ed" />

