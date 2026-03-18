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
    RETRIEVAL_PREVIEW_FILE,
    UNKNOWN_ATTACKS_FILE,
)
from gtd_retclean.es_indexer import create_client
from gtd_retclean.retrieval import run_combined_retrieval
from gtd_retclean.serialization import to_json_ready


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval tests against unknown GTD records.")
    parser.add_argument("--es-host", default="http://localhost:9200", help="Elasticsearch host URL")
    parser.add_argument("--index", default=DEFAULT_ES_INDEX, help="Elasticsearch index name")
    parser.add_argument(
        "--faiss-only",
        action="store_true",
        help="Skip Elasticsearch and run vector retrieval only.",
    )
    parser.add_argument("--unknown-path", default=str(UNKNOWN_ATTACKS_FILE), help="Path to unknown attacks CSV")
    parser.add_argument("--limit", type=int, default=10, help="Number of unknown rows to test")
    parser.add_argument("--top-k", type=int, default=3, help="Top candidates per retrieval engine")
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=DEFAULT_CANDIDATE_POOL_SIZE,
        help="Maximum merged candidates to keep for reranking.",
    )
    parser.add_argument(
        "--output-path",
        default=str(RETRIEVAL_PREVIEW_FILE),
        help="Where to save retrieval output JSON",
    )
    args = parser.parse_args()

    unknown_df = pd.read_csv(args.unknown_path).head(args.limit)
    use_elasticsearch = not args.faiss_only
    client = create_client(args.es_host) if use_elasticsearch else None
    results = run_combined_retrieval(
        query_df=unknown_df,
        es_client=client,
        es_index_name=args.index,
        top_k=args.top_k,
        use_elasticsearch=use_elasticsearch,
        candidate_pool_size=args.candidate_pool_size,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(to_json_ready(results), indent=2), encoding="utf-8")

    mode = "faiss_only" if args.faiss_only else "hybrid"
    print(f"Retrieval mode: {mode}")
    print(f"Unknown queries tested: {len(results)}")
    print(f"Saved retrieval output: {output_path}")
    print(f"Candidate pool size: {args.candidate_pool_size}")


if __name__ == "__main__":
    main()
