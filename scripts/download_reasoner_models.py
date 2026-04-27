from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gtd_retclean.config import (
    DEFAULT_EMBEDDING_MODEL_DIR,
    DEFAULT_EMBEDDING_MODEL_REPO,
    DEFAULT_REASONER_EXTRACTOR_MODEL_DIR,
    DEFAULT_REASONER_EXTRACTOR_REPO,
    DEFAULT_REASONER_MATCHER_MODEL_DIR,
    DEFAULT_REASONER_MATCHER_REPO,
    ensure_project_dirs,
)


def _download_model(repo_id: str, local_dir: Path, force_download: bool = False) -> Path:
    from huggingface_hub import snapshot_download

    local_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        force_download=force_download,
    )
    return local_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the practical local retrieval and reasoner models into the project cache."
    )
    parser.add_argument(
        "--embedding-repo",
        default=DEFAULT_EMBEDDING_MODEL_REPO,
        help="Hugging Face repo for the local embedding model used by retrieval and reranking.",
    )
    parser.add_argument(
        "--embedding-dir",
        default=str(DEFAULT_EMBEDDING_MODEL_DIR),
        help="Local directory where the embedding model should be cached.",
    )
    parser.add_argument(
        "--matcher-repo",
        default=DEFAULT_REASONER_MATCHER_REPO,
        help="Hugging Face repo for the cross-encoder matcher.",
    )
    parser.add_argument(
        "--matcher-dir",
        default=str(DEFAULT_REASONER_MATCHER_MODEL_DIR),
        help="Local directory where the matcher should be cached.",
    )
    parser.add_argument(
        "--extractor-repo",
        default=DEFAULT_REASONER_EXTRACTOR_REPO,
        help="Hugging Face repo for the TinyLlama extractor.",
    )
    parser.add_argument(
        "--extractor-dir",
        default=str(DEFAULT_REASONER_EXTRACTOR_MODEL_DIR),
        help="Local directory where the extractor should be cached.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload files even if they already exist locally.",
    )
    args = parser.parse_args()

    ensure_project_dirs()
    embedding_dir = _download_model(
        repo_id=args.embedding_repo,
        local_dir=Path(args.embedding_dir),
        force_download=args.force_download,
    )
    matcher_dir = _download_model(
        repo_id=args.matcher_repo,
        local_dir=Path(args.matcher_dir),
        force_download=args.force_download,
    )
    extractor_dir = _download_model(
        repo_id=args.extractor_repo,
        local_dir=Path(args.extractor_dir),
        force_download=args.force_download,
    )

    print(f"Embedding repo: {args.embedding_repo}")
    print(f"Embedding dir: {embedding_dir}")
    print(f"Matcher repo: {args.matcher_repo}")
    print(f"Matcher dir: {matcher_dir}")
    print(f"Extractor repo: {args.extractor_repo}")
    print(f"Extractor dir: {extractor_dir}")


if __name__ == "__main__":
    main()
