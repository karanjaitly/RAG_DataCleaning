# GTD RetClean (Weeks 1-8)

This repository now covers the first four proposal milestones:

1. Week 1-2 EDA: profile GTD missingness and summarize the unknown `gname` population.
2. Week 3-4 tuple-based indexing: split known vs. unknown attacks, then build Elasticsearch and Faiss retrieval layers.
3. Week 5-6 reranking: merge lexical/vector candidates and rerank them with a ColBERT-style late interaction module.
4. Week 7-8 reasoner: evaluate reranked candidate tuples with a local matcher/extractor stack and infer the missing `gname`.
5. Week 9-10 evaluation: hold out known incidents, run the Faiss-only pipeline end to end, and compare against a majority-class baseline.

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

Download the local model cache used by retrieval and the week 7-8 reasoner:

```bash
python scripts/download_reasoner_models.py
```

Start Elasticsearch locally (default expected at `http://localhost:9200`).

## Run Weeks 1-8 Pipeline

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

Week 7-8 reasoner smoke test:

```bash
python scripts/07_reason_about_candidates.py --reranked-path outputs/reranked_preview.json --limit 1 --save-every 1
```

Week 7-8 full local-model reasoner run:

```bash
python scripts/07_reason_about_candidates.py --reranked-path outputs/reranked_preview.json --save-every 1
```

Resume an interrupted reasoner run:

```bash
python scripts/07_reason_about_candidates.py --reranked-path outputs/reranked_preview.json --resume --save-every 1
```

Week 9-10 evaluation on a practical validation sample:

```bash
python scripts/08_evaluate_pipeline.py --sample-size 100 --save-every 1
```

Fast smoke test of the evaluation flow:

```bash
python scripts/08_evaluate_pipeline.py --sample-size 10 --reranker-backend token_overlap --matcher-backend field_weighted --extractor-backend group_vote
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
- `outputs/reasoned_preview.json`
- `outputs/milestone_verification.json`
- `outputs/evaluation_predictions.json`
- `outputs/evaluation_summary.json`
- `indices/gtd_summary_faiss.index`
- `indices/gtd_summary_faiss_metadata.csv`

## Notes

- The reranker is modular: it currently supports a ColBERT-style `late_interaction` backend and a lightweight `token_overlap` fallback for offline checks.
- The practical week 7-8 default stack is a local `cross-encoder/ms-marco-MiniLM-L-6-v2` matcher plus a local `TinyLlama/TinyLlama-1.1B-Chat-v1.0` extractor cached under `.cache/hf_models/`.
- The reasoner script now supports `--limit`, `--start-index`, `--save-every`, and `--resume` so testing can use small smoke runs before expensive full-batch evaluation.
- The evaluation script rebuilds a temporary Faiss index from the training split only, so held-out validation incidents do not leak back into retrieval.
- The evaluation defaults to a sample-sized validation run and can trim the sample further if projected local-model runtime exceeds the configured budget.
- Unit tests validate local pipeline logic, scoring behavior, serialization, and model integration boundaries, but they do not replace a practical end-to-end run on real GTD rows.
