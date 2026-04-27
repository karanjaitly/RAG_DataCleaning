from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gtd_retclean.config import (
    DEFAULT_MAX_SEQUENCE_LENGTH,
    DEFAULT_REASONER_CANDIDATE_LIMIT,
    DEFAULT_REASONER_EXTRACTOR_BACKEND,
    DEFAULT_REASONER_EXTRACTOR_MODEL,
    DEFAULT_REASONER_MATCHER_BACKEND,
    DEFAULT_REASONER_MATCHER_MODEL,
    REASONED_PREVIEW_FILE,
    RERANKED_PREVIEW_FILE,
)
from gtd_retclean.reasoner import create_extractor, create_matcher, reason_single_result
from gtd_retclean.serialization import to_json_ready


def _load_existing_results(output_path: Path, resume: bool) -> list[dict]:
    if not resume or not output_path.exists():
        return []
    return json.loads(output_path.read_text(encoding="utf-8"))


def _persist_results(output_path: Path, results: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(to_json_ready(results), indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the week 7-8 reasoner over reranked GTD candidates.")
    parser.add_argument(
        "--reranked-path",
        default=str(RERANKED_PREVIEW_FILE),
        help="Path to reranked JSON produced by the week 5-6 pipeline.",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=DEFAULT_REASONER_CANDIDATE_LIMIT,
        help="Maximum reranked candidates to pass into the reasoner per query.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of query rows to reason. Useful for short smoke tests.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start offset into the reranked JSON payload. Useful for batched testing.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Persist partial progress every N records. Set to 0 to only save at the end.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output file by skipping already saved results.",
    )
    parser.add_argument(
        "--matcher-backend",
        default=DEFAULT_REASONER_MATCHER_BACKEND,
        choices=["field_weighted", "cross_encoder_matcher", "roberta_matcher"],
        help="Matcher backend. cross_encoder_matcher is the practical local default for this repo.",
    )
    parser.add_argument(
        "--matcher-model",
        default=DEFAULT_REASONER_MATCHER_MODEL,
        help="Optional matcher model name or local checkpoint path.",
    )
    parser.add_argument(
        "--extractor-backend",
        default=DEFAULT_REASONER_EXTRACTOR_BACKEND,
        choices=["group_vote", "llama_extractor"],
        help="Extractor backend. llama_extractor is the practical local TinyLlama-backed default.",
    )
    parser.add_argument(
        "--extractor-model",
        default=DEFAULT_REASONER_EXTRACTOR_MODEL,
        help="Optional extractor model name or local checkpoint path.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_SEQUENCE_LENGTH,
        help="Token length used by transformer-based matcher backends.",
    )
    parser.add_argument(
        "--output-path",
        default=str(REASONED_PREVIEW_FILE),
        help="Where to save the reasoner output JSON.",
    )
    args = parser.parse_args()

    reranked_results = json.loads(Path(args.reranked_path).read_text(encoding="utf-8"))
    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")
    if args.save_every < 0:
        raise ValueError("--save-every must be >= 0")
    selected_results = reranked_results[args.start_index :]
    if args.limit is not None:
        selected_results = selected_results[: args.limit]

    output_path = Path(args.output_path)
    existing_results = _load_existing_results(output_path, args.resume)
    already_completed = len(existing_results)
    remaining_results = selected_results[already_completed:]

    matcher = create_matcher(
        backend=args.matcher_backend,
        model_name=args.matcher_model,
        max_length=args.max_length,
    )
    extractor = create_extractor(
        backend=args.extractor_backend,
        model_name=args.extractor_model,
    )

    reasoned_results = list(existing_results)
    for index, result in enumerate(remaining_results, start=1):
        reasoned_results.append(
            reason_single_result(
                result=result,
                matcher=matcher,
                extractor=extractor,
                candidate_limit=args.candidate_limit,
            )
        )
        if args.save_every and index % args.save_every == 0:
            _persist_results(output_path, reasoned_results)
            print(
                f"Checkpoint saved after {len(reasoned_results)} record(s): {output_path}",
                flush=True,
            )

    _persist_results(output_path, reasoned_results)

    predicted_count = sum(1 for item in reasoned_results if item.get("predicted_gname"))
    print(f"Matcher backend: {args.matcher_backend}")
    print(f"Extractor backend: {args.extractor_backend}")
    print(f"Queries reasoned: {len(reasoned_results)}")
    print(f"Predictions emitted: {predicted_count}")
    if args.limit is not None:
        print(f"Requested limit: {args.limit}")
    if args.resume:
        print(f"Resumed from existing results: {already_completed}")
    print(f"Saved reasoner output: {output_path}")


if __name__ == "__main__":
    main()
