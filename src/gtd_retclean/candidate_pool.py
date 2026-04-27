from __future__ import annotations

from typing import Any

import pandas as pd


def _candidate_key(candidate: dict[str, Any]) -> str:
    event_id = candidate.get("eventid")
    if event_id is not None and not pd.isna(event_id) and str(event_id).strip():
        return f"eventid::{str(event_id).strip()}"

    summary = str(candidate.get("summary", "")).strip().lower()
    return f"summary::{summary}"


def build_candidate_pool(
    es_candidates: list[dict[str, Any]],
    faiss_candidates: list[dict[str, Any]],
    candidate_pool_size: int | None = None,
) -> list[dict[str, Any]]:
    """Merge lexical/vector candidates into a deduplicated pool for reranking."""
    merged: dict[str, dict[str, Any]] = {}

    for engine_name, candidates in (("elasticsearch", es_candidates), ("faiss", faiss_candidates)):
        for rank, candidate in enumerate(candidates, start=1):
            key = _candidate_key(candidate)
            if key not in merged:
                merged[key] = dict(candidate)
                merged[key]["source_engines"] = []
                merged[key]["source_scores"] = {}
                merged[key]["source_ranks"] = {}
                merged[key]["reciprocal_rank_score"] = 0.0

            pooled = merged[key]
            pooled["source_engines"].append(engine_name)
            pooled["source_scores"][engine_name] = float(candidate.get("score", 0.0))
            pooled["source_ranks"][engine_name] = rank
            pooled["reciprocal_rank_score"] += 1.0 / rank

    ranked = sorted(
        merged.values(),
        key=lambda record: (
            -len(record["source_engines"]),
            -record["reciprocal_rank_score"],
            str(record.get("eventid", "")),
        ),
    )

    for rank, record in enumerate(ranked, start=1):
        record["candidate_rank"] = rank
        record["source_count"] = len(record["source_engines"])

    if candidate_pool_size is not None:
        return ranked[:candidate_pool_size]
    return ranked
