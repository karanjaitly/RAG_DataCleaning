# GTD RetClean (Weeks 3-4)

This repository now includes the week 3-4 tuple-based indexing workflow described in `instructions.md`:

1. Data prep: split GTD into known (`gname` available) and unknown (`gname` missing/Unknown) attacks.
2. Elasticsearch index: lexical retrieval over known attack summaries.
3. Faiss index: dense retrieval over known attack summary embeddings.
4. Retrieval test: run basic query tests for unknown attacks.

## Project Structure

- `data/` input datasets (place `gtd_6_month.xlsx` here)
- `indices/` saved Faiss artifacts
- `outputs/` prepared data and retrieval outputs
- `src/gtd_retclean/` reusable pipeline modules
- `scripts/` runnable week 3-4 task scripts

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Start Elasticsearch locally (default expected at `http://localhost:9200`).

## Run Weeks 3-4 Pipeline

```bash
python scripts/01_prepare_data.py
python scripts/02_build_elasticsearch_index.py
python scripts/03_build_faiss_index.py
python scripts/04_test_retrieval.py --limit 10 --top-k 3
```

Faiss-only mode (no Elasticsearch required):

```bash
python scripts/01_prepare_data.py
python scripts/03_build_faiss_index.py
python scripts/04_test_retrieval.py --faiss-only --limit 10 --top-k 3
```

Outputs:

- `outputs/known_attacks.csv`
- `outputs/unknown_attacks.csv`
- `indices/gtd_summary_faiss.index`
- `indices/gtd_summary_faiss_metadata.csv`
- `outputs/retrieval_preview.json`
