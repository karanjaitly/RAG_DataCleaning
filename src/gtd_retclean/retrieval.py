from __future__ import annotations

from typing import Any

import pandas as pd

from .config import ColumnConfig, DEFAULT_EMBEDDING_MODEL, FAISS_INDEX_FILE, FAISS_METADATA_FILE
from .es_indexer import search_by_summary
from .faiss_indexer import load_faiss_artifacts, search_faiss


def _hit_to_record(hit: dict[str, Any]) -> dict[str, Any]:
    source = hit.get("_source", {})
    source["score"] = float(hit.get("_score", 0.0))
    return source


def run_combined_retrieval(
    query_df: pd.DataFrame,
    es_client=None,
    es_index_name: str | None = None,
    top_k: int = 3,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    use_elasticsearch: bool = True,
) -> list[dict[str, Any]]:
    """Run lexical + vector retrieval for unknown attacks.

    RAG step: retrieve candidate historical attacks for each dirty tuple.
    """
    if use_elasticsearch and (es_client is None or not es_index_name):
        raise ValueError(
            "Elasticsearch is enabled but client/index was not provided. "
            "Pass use_elasticsearch=False for Faiss-only mode."
        )

    cols = ColumnConfig()
    index, metadata = load_faiss_artifacts(FAISS_INDEX_FILE, FAISS_METADATA_FILE)

    outputs: list[dict[str, Any]] = []
    for _, row in query_df.iterrows():
        query_text = str(row.get(cols.summary, ""))
        if not query_text:
            continue

        es_hits = []
        if use_elasticsearch:
            es_hits = search_by_summary(es_client, es_index_name, query_text, top_k=top_k)

        vec_hits = search_faiss(
            query_text=query_text,
            index=index,
            metadata=metadata,
            model_name=model_name,
            top_k=top_k,
        )

        outputs.append(
            {
                "eventid": row.get(cols.event_id),
                "query_summary": query_text,
                "es_candidates": [_hit_to_record(hit) for hit in es_hits],
                "faiss_candidates": vec_hits,
                "mode": "hybrid" if use_elasticsearch else "faiss_only",
            }
        )
    return outputs
