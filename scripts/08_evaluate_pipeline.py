from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gtd_retclean.config import (
    DEFAULT_CANDIDATE_POOL_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    DEFAULT_REASONER_CANDIDATE_LIMIT,
    DEFAULT_REASONER_EXTRACTOR_BACKEND,
    DEFAULT_REASONER_EXTRACTOR_MODEL,
    DEFAULT_REASONER_MATCHER_BACKEND,
    DEFAULT_REASONER_MATCHER_MODEL,
    DEFAULT_RERANKER_BACKEND,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANK_TOP_K,
    DEFAULT_RETRIEVAL_TOP_K,
    EVALUATION_PREDICTIONS_FILE,
    EVALUATION_SUMMARY_FILE,
    KNOWN_ATTACKS_FILE,
)
from gtd_retclean.evaluation import evaluate_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the Faiss-only GTD RetClean pipeline on held-out known incidents.",
    )
    parser.add_argument("--known-path", default=str(KNOWN_ATTACKS_FILE), help="Path to known attacks CSV.")
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Fraction of each eligible group to hold out for validation.",
    )
    parser.add_argument(
        "--min-group-count",
        type=int,
        default=2,
        help="Minimum examples required for a group to contribute validation rows.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Maximum validation rows to score. Use 0 or a negative value for the full eligible validation split.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the train/validation split.")
    parser.add_argument(
        "--retrieve-top-k",
        type=int,
        default=DEFAULT_RETRIEVAL_TOP_K,
        help="Top Faiss candidates to retrieve before reranking.",
    )
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=DEFAULT_CANDIDATE_POOL_SIZE,
        help="Maximum merged candidate pool size before reranking.",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=DEFAULT_RERANK_TOP_K,
        help="Final number of candidates kept after reranking.",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=DEFAULT_REASONER_CANDIDATE_LIMIT,
        help="Maximum reranked candidates passed into the reasoner per query.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="SentenceTransformer model used for Faiss retrieval.",
    )
    parser.add_argument(
        "--reranker-backend",
        default=DEFAULT_RERANKER_BACKEND,
        choices=["late_interaction", "token_overlap"],
        help="Reranker backend. late_interaction is the main local-model option.",
    )
    parser.add_argument(
        "--reranker-model",
        default=DEFAULT_RERANKER_MODEL,
        help="Encoder model name or local path for the late-interaction reranker.",
    )
    parser.add_argument(
        "--matcher-backend",
        default=DEFAULT_REASONER_MATCHER_BACKEND,
        choices=["field_weighted", "cross_encoder_matcher", "roberta_matcher"],
        help="Matcher backend used by the week 7-8 reasoner.",
    )
    parser.add_argument(
        "--matcher-model",
        default=DEFAULT_REASONER_MATCHER_MODEL,
        help="Matcher model name or local path for transformer-based backends.",
    )
    parser.add_argument(
        "--extractor-backend",
        default=DEFAULT_REASONER_EXTRACTOR_BACKEND,
        choices=["group_vote", "llama_extractor"],
        help="Extractor backend used by the week 7-8 reasoner.",
    )
    parser.add_argument(
        "--extractor-model",
        default=DEFAULT_REASONER_EXTRACTOR_MODEL,
        help="Extractor model name or local path for local LLaMA-backed runs.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_SEQUENCE_LENGTH,
        help="Token limit for transformer-based reranker and matcher backends.",
    )
    parser.add_argument(
        "--runtime-budget-minutes",
        type=float,
        default=30.0,
        help="Soft runtime budget used to trim the sample if local models are too slow.",
    )
    parser.add_argument(
        "--runtime-check-after-rows",
        type=int,
        default=3,
        help="How many completed rows to observe before estimating runtime.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save prediction checkpoints every N new rows. Set to 0 to only save at the end.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing predictions file for the same split/config.",
    )
    parser.add_argument(
        "--predictions-path",
        default=str(EVALUATION_PREDICTIONS_FILE),
        help="Where to save per-row evaluation predictions.",
    )
    parser.add_argument(
        "--summary-path",
        default=str(EVALUATION_SUMMARY_FILE),
        help="Where to save evaluation summary metrics.",
    )
    args = parser.parse_args()

    known_df = pd.read_csv(args.known_path)
    summary = evaluate_pipeline(
        known_df=known_df,
        predictions_path=Path(args.predictions_path),
        summary_path=Path(args.summary_path),
        validation_fraction=args.validation_fraction,
        min_group_count=args.min_group_count,
        sample_size=args.sample_size,
        random_seed=args.seed,
        retrieval_top_k=args.retrieve_top_k,
        candidate_pool_size=args.candidate_pool_size,
        rerank_top_k=args.rerank_top_k,
        candidate_limit=args.candidate_limit,
        embedding_model_name=args.embedding_model,
        reranker_backend=args.reranker_backend,
        reranker_model_name=args.reranker_model,
        matcher_backend=args.matcher_backend,
        matcher_model_name=args.matcher_model,
        extractor_backend=args.extractor_backend,
        extractor_model_name=args.extractor_model,
        max_length=args.max_length,
        save_every=args.save_every,
        resume=args.resume,
        runtime_budget_minutes=args.runtime_budget_minutes,
        runtime_check_after_rows=args.runtime_check_after_rows,
        progress_callback=lambda message: print(message, flush=True),
    )

    pipeline_metrics = summary["metrics"]["pipeline"]
    majority_metrics = summary["metrics"]["majority_baseline"]
    print(f"Evaluation status: {summary['status']}")
    print(f"Pipeline accuracy: {pipeline_metrics['accuracy']:.4f}")
    print(f"Pipeline macro-F1: {pipeline_metrics['macro_f1']:.4f}")
    print(f"Majority baseline ({majority_metrics['predicted_gname']}): {majority_metrics['accuracy']:.4f}")
    print(f"Accuracy lift: {summary['metrics']['accuracy_lift']:.4f}")
    print(f"Coverage: {pipeline_metrics['coverage']:.4f}")
    print(f"Evaluated rows: {summary['runtime']['evaluated_rows']}")
    print(f"Elapsed minutes: {summary['runtime']['elapsed_minutes']:.2f}")
    print(f"Saved predictions: {args.predictions_path}")
    print(f"Saved summary: {args.summary_path}")


if __name__ == "__main__":
    main()
