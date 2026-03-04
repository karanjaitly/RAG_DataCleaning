from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gtd_retclean.config import (
    DEFAULT_EMBEDDING_MODEL,
    FAISS_INDEX_FILE,
    FAISS_METADATA_FILE,
    KNOWN_ATTACKS_FILE,
)
from gtd_retclean.faiss_indexer import build_embeddings, build_faiss_index, persist_faiss_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Faiss index from known GTD summaries.")
    parser.add_argument("--known-path", default=str(KNOWN_ATTACKS_FILE), help="Path to known attacks CSV")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL, help="SentenceTransformer model name")
    parser.add_argument("--index-path", default=str(FAISS_INDEX_FILE), help="Output Faiss index path")
    parser.add_argument("--metadata-path", default=str(FAISS_METADATA_FILE), help="Output metadata CSV path")
    args = parser.parse_args()

    known_df = pd.read_csv(args.known_path)
    if "summary" not in known_df.columns:
        raise ValueError("Known attacks CSV must include a 'summary' column")

    known_df["summary"] = known_df["summary"].fillna("").astype(str)
    known_df = known_df[known_df["summary"].str.strip() != ""].reset_index(drop=True)

    embeddings = build_embeddings(known_df["summary"].tolist(), model_name=args.model)
    index = build_faiss_index(embeddings)
    persist_faiss_artifacts(index, known_df, Path(args.index_path), Path(args.metadata_path))

    print(f"Vector rows indexed: {index.ntotal}")
    print(f"Faiss index: {args.index_path}")
    print(f"Metadata: {args.metadata_path}")


if __name__ == "__main__":
    main()
