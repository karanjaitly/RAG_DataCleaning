# GTD RetClean (Weeks 1-6)

This repository now covers the first three proposal milestones:

1. Week 1-2 EDA: profile GTD missingness and summarize the unknown `gname` population.
2. Week 3-4 tuple-based indexing: split known vs. unknown attacks, then build Elasticsearch and Faiss retrieval layers.
3. Week 5-6 reranking: merge lexical/vector candidates and rerank them with a ColBERT-style late interaction module.

## Project Structure

- `data/` input datasets (place `gtd_6_month.xlsx` here)
- `indices/` saved Faiss artifacts
- `outputs/` prepared data and retrieval outputs
- `src/gtd_retclean/` reusable pipeline modules
- `scripts/` runnable week-by-week task scripts

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Start Elasticsearch locally (default expected at `http://localhost:9200`).

## Run Weeks 1-6 Pipeline

Week 1-2 EDA:

```bash
python scripts/00_profile_data.py
```

Weeks 3-4 retrieval pipeline:

```bash
python scripts/01_prepare_data.py
python scripts/02_build_elasticsearch_index.py
python scripts/03_build_faiss_index.py
python scripts/04_test_retrieval.py --limit 10 --top-k 5 --candidate-pool-size 8
```

Faiss-only mode (no Elasticsearch required):

```bash
python scripts/01_prepare_data.py
python scripts/03_build_faiss_index.py
python scripts/04_test_retrieval.py --faiss-only --limit 10 --top-k 5 --candidate-pool-size 8
```

Week 5-6 reranking:

```bash
python scripts/05_rerank_candidates.py --faiss-only --limit 10 --retrieve-top-k 5 --rerank-top-k 3
```

Rerank a previously saved retrieval file:

```bash
python scripts/05_rerank_candidates.py --retrieval-path outputs/retrieval_preview.json --backend token_overlap
```

Verify previous milestones:

```bash
python scripts/06_verify_previous_work.py
```

Outputs:

- `outputs/eda_summary.json`
- `outputs/missing_value_profile.csv`
- `outputs/known_attacks.csv`
- `outputs/unknown_attacks.csv`
- `outputs/retrieval_preview.json`
- `outputs/reranked_preview.json`
- `outputs/milestone_verification.json`
- `indices/gtd_summary_faiss.index`
- `indices/gtd_summary_faiss_metadata.csv`

## Notes

- The reranker is modular: it currently supports a ColBERT-style `late_interaction` backend and a lightweight `token_overlap` fallback for offline checks.
- Future week 7-8 reasoner work can consume `candidate_pool` or `reranked_candidates` directly without changing the retrieval scripts.
