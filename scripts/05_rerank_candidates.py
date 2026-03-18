from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gtd_retclean.config import (
    DEFAULT_CANDIDATE_POOL_SIZE,
    DEFAULT_ES_INDEX,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    DEFAULT_RERANKER_BACKEND,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANK_TOP_K,
    DEFAULT_RETRIEVAL_TOP_K,
    RERANKED_PREVIEW_FILE,
    UNKNOWN_ATTACKS_FILE,
)
from gtd_retclean.es_indexer import create_client
from gtd_retclean.retrieval import run_combined_retrieval
from gtd_retclean.reranker import create_reranker, rerank_retrieval_results
from gtd_retclean.serialization import to_json_ready


def main() -> None:
    parser = argparse.ArgumentParser(description="Run week 5-6 reranking over retrieved GTD candidates.")
    parser.add_argument("--es-host", default="http://localhost:9200", help="Elasticsearch host URL")
    parser.add_argument("--index", default=DEFAULT_ES_INDEX, help="Elasticsearch index name")
    parser.add_argument(
        "--faiss-only",
        action="store_true",
        help="Skip Elasticsearch and rerank Faiss retrieval results only.",
    )
    parser.add_argument(
        "--retrieval-path",
        default=None,
        help="Optional path to an existing retrieval JSON file. If provided, retrieval is skipped.",
    )
    parser.add_argument("--unknown-path", default=str(UNKNOWN_ATTACKS_FILE), help="Path to unknown attacks CSV")
    parser.add_argument("--limit", type=int, default=10, help="Number of unknown rows to rerank")
    parser.add_argument(
        "--retrieve-top-k",
        type=int,
        default=DEFAULT_RETRIEVAL_TOP_K,
        help="Top candidates to retrieve from each engine before reranking.",
    )
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=DEFAULT_CANDIDATE_POOL_SIZE,
        help="Maximum merged candidates kept before reranking.",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=DEFAULT_RERANK_TOP_K,
        help="Final number of candidates to keep after reranking.",
    )
    parser.add_argument(
        "--backend",
        default=DEFAULT_RERANKER_BACKEND,
        choices=["late_interaction", "token_overlap"],
        help="Reranker backend. Use late_interaction to evaluate a ColBERT-style scorer.",
    )
    parser.add_argument("--model", default=DEFAULT_RERANKER_MODEL, help="Encoder model for late interaction")
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_SEQUENCE_LENGTH,
        help="Max token length for the reranker encoder.",
    )
    parser.add_argument(
        "--output-path",
        default=str(RERANKED_PREVIEW_FILE),
        help="Where to save reranked output JSON",
    )
    args = parser.parse_args()

    if args.retrieval_path:
        retrieval_results = json.loads(Path(args.retrieval_path).read_text(encoding="utf-8"))
    else:
        unknown_df = pd.read_csv(args.unknown_path).head(args.limit)
        use_elasticsearch = not args.faiss_only
        client = create_client(args.es_host) if use_elasticsearch else None
        retrieval_results = run_combined_retrieval(
            query_df=unknown_df,
            es_client=client,
            es_index_name=args.index,
            top_k=args.retrieve_top_k,
            use_elasticsearch=use_elasticsearch,
            candidate_pool_size=args.candidate_pool_size,
        )
    reranker = create_reranker(
        backend=args.backend,
        model_name=args.model,
        max_length=args.max_length,
    )
    reranked_results = rerank_retrieval_results(
        retrieval_results=retrieval_results,
        reranker=reranker,
        top_k=args.rerank_top_k,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(to_json_ready(reranked_results), indent=2), encoding="utf-8")

    print(f"Reranker backend: {args.backend}")
    print(f"Queries reranked: {len(reranked_results)}")
    if args.retrieval_path:
        print(f"Loaded retrieval input: {args.retrieval_path}")
    else:
        print(f"Retrieved candidates per engine: {args.retrieve_top_k}")
    print(f"Reranked top-k: {args.rerank_top_k}")
    print(f"Saved reranked output: {output_path}")


if __name__ == "__main__":
    main()
