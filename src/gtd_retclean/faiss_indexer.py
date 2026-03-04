from __future__ import annotations

from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .config import DEFAULT_EMBEDDING_MODEL


def build_embeddings(texts: list[str], model_name: str = DEFAULT_EMBEDDING_MODEL) -> np.ndarray:
    """Convert summaries into dense vectors for semantic retrieval."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embeddings.astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Create an inner-product index over normalized embeddings."""
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def persist_faiss_artifacts(
    index: faiss.Index,
    metadata_df: pd.DataFrame,
    index_path: Path,
    metadata_path: Path,
) -> None:
    """Persist vector index plus row metadata mapping for lookup."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    metadata_df.to_csv(metadata_path, index=False)


def load_faiss_artifacts(index_path: Path, metadata_path: Path) -> tuple[faiss.Index, pd.DataFrame]:
    """Load saved Faiss index and metadata mapping."""
    index = faiss.read_index(str(index_path))
    metadata = pd.read_csv(metadata_path)
    return index, metadata


def search_faiss(
    query_text: str,
    index: faiss.Index,
    metadata: pd.DataFrame,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Run vector similarity search and return top matching records."""
    model = SentenceTransformer(model_name)
    q = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
    distances, ids = index.search(q.astype("float32"), top_k)

    results: list[dict[str, Any]] = []
    for score, row_id in zip(distances[0], ids[0]):
        if row_id < 0:
            continue
        row = metadata.iloc[int(row_id)].to_dict()
        row["score"] = float(score)
        results.append(row)
    return results
