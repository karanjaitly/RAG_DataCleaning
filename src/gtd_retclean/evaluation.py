from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from .candidate_pool import build_candidate_pool
from .config import (
    ColumnConfig,
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
    ensure_project_dirs,
)
from .data_prep import UNKNOWN_GNAME_MARKERS
from .faiss_indexer import build_embeddings, build_faiss_index, search_faiss
from .reasoner import BaseExtractor, BaseMatcher, create_extractor, create_matcher, reason_single_result
from .reranker import BaseReranker, create_reranker
from .serialization import to_json_ready


def _normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in UNKNOWN_GNAME_MARKERS:
        return None
    return text


def _progress(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _persist_predictions(predictions: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(to_json_ready(predictions), indent=2), encoding="utf-8")


def _load_existing_predictions(output_path: Path, resume: bool) -> list[dict[str, Any]]:
    if not resume or not output_path.exists():
        return []
    return json.loads(output_path.read_text(encoding="utf-8"))


def build_validation_split(
    known_df: pd.DataFrame,
    validation_fraction: float = 0.2,
    min_group_count: int = 2,
    max_validation_rows: int | None = 100,
    random_seed: int = 42,
    columns: ColumnConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create a train/validation split with at least one training row per validation group."""
    if validation_fraction <= 0.0 or validation_fraction >= 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if min_group_count < 2:
        raise ValueError("min_group_count must be at least 2.")

    cols = columns or ColumnConfig()
    cleaned = known_df.copy().reset_index(drop=True)
    cleaned["_split_order"] = cleaned.index
    cleaned[cols.summary] = cleaned[cols.summary].fillna("").astype(str).str.strip()
    cleaned[cols.gname] = cleaned[cols.gname].fillna("").astype(str).str.strip()
    cleaned = cleaned[
        (cleaned[cols.summary] != "")
        & ~cleaned[cols.gname].str.lower().isin(UNKNOWN_GNAME_MARKERS)
    ].reset_index(drop=True)

    if cleaned.empty:
        raise ValueError("No known GTD rows with both summary text and a usable group label were available.")

    group_counts = cleaned[cols.gname].value_counts()
    eligible_groups = set(group_counts[group_counts >= min_group_count].index.tolist())

    validation_parts: list[pd.DataFrame] = []
    train_parts: list[pd.DataFrame] = []
    eligible_rows = cleaned[cleaned[cols.gname].isin(eligible_groups)]
    ineligible_rows = cleaned[~cleaned[cols.gname].isin(eligible_groups)]
    if not ineligible_rows.empty:
        train_parts.append(ineligible_rows)

    for offset, group_name in enumerate(sorted(eligible_groups)):
        group_df = eligible_rows[eligible_rows[cols.gname] == group_name].sample(
            frac=1.0,
            random_state=random_seed + offset,
        )
        holdout_count = max(1, int(round(len(group_df) * validation_fraction)))
        holdout_count = min(holdout_count, len(group_df) - 1)
        validation_parts.append(group_df.iloc[:holdout_count])
        train_parts.append(group_df.iloc[holdout_count:])

    train_df = pd.concat(train_parts, ignore_index=False).sort_values("_split_order")
    validation_df = pd.concat(validation_parts, ignore_index=False).sort_values("_split_order")

    uncapped_validation_rows = int(len(validation_df))
    sample_capped = False
    if max_validation_rows is not None and max_validation_rows > 0 and len(validation_df) > max_validation_rows:
        sample_capped = True
        selected_validation = validation_df.sample(n=max_validation_rows, random_state=random_seed)
        returned_to_train = validation_df.drop(selected_validation.index)
        if not returned_to_train.empty:
            train_df = pd.concat([train_df, returned_to_train], ignore_index=False).sort_values("_split_order")
        validation_df = selected_validation.sort_values("_split_order")

    train_df = train_df.drop(columns="_split_order").reset_index(drop=True)
    validation_df = validation_df.drop(columns="_split_order")
    validation_df = validation_df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    split_summary = {
        "known_rows": int(len(cleaned)),
        "eligible_groups_for_validation": int(len(eligible_groups)),
        "excluded_groups_from_validation": int((group_counts < min_group_count).sum()),
        "eligible_validation_rows_before_cap": uncapped_validation_rows,
        "excluded_validation_rows": int(len(ineligible_rows)),
        "training_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "validation_fraction": float(validation_fraction),
        "min_group_count": int(min_group_count),
        "sample_capped": sample_capped,
        "max_validation_rows": None if max_validation_rows is None else int(max_validation_rows),
    }
    return train_df, validation_df, split_summary


def compute_prediction_metrics(
    actual_labels: list[Any],
    predicted_labels: list[Any],
) -> dict[str, float | int]:
    """Compute accuracy-style metrics without adding sklearn as a dependency."""
    if len(actual_labels) != len(predicted_labels):
        raise ValueError("actual_labels and predicted_labels must have the same length.")

    normalized_actual = [_normalize_label(label) for label in actual_labels]
    normalized_predicted = [_normalize_label(label) for label in predicted_labels]
    total = len(normalized_actual)
    if total == 0:
        return {
            "rows": 0,
            "accuracy": 0.0,
            "coverage": 0.0,
            "covered_accuracy": 0.0,
            "macro_f1": 0.0,
        }

    covered_mask = [prediction is not None for prediction in normalized_predicted]
    correct_mask = [
        actual is not None and predicted is not None and actual == predicted
        for actual, predicted in zip(normalized_actual, normalized_predicted)
    ]

    covered_rows = sum(covered_mask)
    correct_rows = sum(correct_mask)
    covered_correct = sum(correct and covered for correct, covered in zip(correct_mask, covered_mask))

    labels = sorted(
        {label for label in normalized_actual if label is not None}
        | {label for label in normalized_predicted if label is not None}
    )
    if not labels:
        macro_f1 = 0.0
    else:
        label_f1_scores: list[float] = []
        for label in labels:
            true_positive = sum(
                actual == label and predicted == label
                for actual, predicted in zip(normalized_actual, normalized_predicted)
            )
            false_positive = sum(
                actual != label and predicted == label
                for actual, predicted in zip(normalized_actual, normalized_predicted)
            )
            false_negative = sum(
                actual == label and predicted != label
                for actual, predicted in zip(normalized_actual, normalized_predicted)
            )
            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
            if precision + recall == 0.0:
                label_f1_scores.append(0.0)
            else:
                label_f1_scores.append((2.0 * precision * recall) / (precision + recall))
        macro_f1 = sum(label_f1_scores) / len(label_f1_scores)

    return {
        "rows": int(total),
        "accuracy": float(correct_rows / total),
        "coverage": float(covered_rows / total),
        "covered_accuracy": float(covered_correct / covered_rows) if covered_rows else 0.0,
        "macro_f1": float(macro_f1),
    }


def summarize_evaluation_results(
    predictions: list[dict[str, Any]],
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    split_summary: dict[str, Any],
    config: dict[str, Any],
    runtime: dict[str, Any],
    columns: ColumnConfig | None = None,
) -> dict[str, Any]:
    cols = columns or ColumnConfig()
    actual_labels = [record.get("actual_gname") for record in predictions]
    predicted_labels = [record.get("predicted_gname") for record in predictions]
    pipeline_metrics = compute_prediction_metrics(actual_labels, predicted_labels)

    majority_gname = str(train_df[cols.gname].value_counts().idxmax())
    majority_rate = float((train_df[cols.gname] == majority_gname).mean())
    majority_predictions = [majority_gname] * len(validation_df)
    majority_metrics = compute_prediction_metrics(validation_df[cols.gname].tolist(), majority_predictions)

    example_fields = ("eventid", "actual_gname", "predicted_gname", "confidence_label", "is_correct")
    correct_examples = [
        {field: record.get(field) for field in example_fields}
        for record in predictions
        if record.get("is_correct")
    ][:3]
    incorrect_examples = [
        {field: record.get(field) for field in example_fields}
        for record in predictions
        if not record.get("is_correct")
    ][:3]

    return {
        "status": runtime.get("status", "completed"),
        "config": config,
        "split": split_summary,
        "label_distribution": {
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(validation_df)),
            "train_unique_groups": int(train_df[cols.gname].nunique()),
            "validation_unique_groups": int(validation_df[cols.gname].nunique()),
            "majority_gname": majority_gname,
            "majority_rate": majority_rate,
        },
        "metrics": {
            "pipeline": pipeline_metrics,
            "majority_baseline": {
                **majority_metrics,
                "predicted_gname": majority_gname,
            },
            "accuracy_lift": float(pipeline_metrics["accuracy"] - majority_metrics["accuracy"]),
        },
        "runtime": runtime,
        "examples": {
            "correct": correct_examples,
            "incorrect": incorrect_examples,
        },
    }


def persist_evaluation_artifacts(
    predictions: list[dict[str, Any]],
    summary: dict[str, Any],
    predictions_path: Path = EVALUATION_PREDICTIONS_FILE,
    summary_path: Path = EVALUATION_SUMMARY_FILE,
) -> tuple[Path, Path]:
    ensure_project_dirs()
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text(json.dumps(to_json_ready(predictions), indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(to_json_ready(summary), indent=2), encoding="utf-8")
    return predictions_path, summary_path


def evaluate_pipeline(
    known_df: pd.DataFrame,
    predictions_path: Path = EVALUATION_PREDICTIONS_FILE,
    summary_path: Path = EVALUATION_SUMMARY_FILE,
    validation_fraction: float = 0.2,
    min_group_count: int = 2,
    sample_size: int | None = 100,
    random_seed: int = 42,
    retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE,
    rerank_top_k: int = DEFAULT_RERANK_TOP_K,
    candidate_limit: int = DEFAULT_REASONER_CANDIDATE_LIMIT,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    reranker_backend: str = DEFAULT_RERANKER_BACKEND,
    reranker_model_name: str = DEFAULT_RERANKER_MODEL,
    matcher_backend: str = DEFAULT_REASONER_MATCHER_BACKEND,
    matcher_model_name: str = DEFAULT_REASONER_MATCHER_MODEL,
    extractor_backend: str = DEFAULT_REASONER_EXTRACTOR_BACKEND,
    extractor_model_name: str = DEFAULT_REASONER_EXTRACTOR_MODEL,
    max_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    save_every: int = 1,
    resume: bool = False,
    runtime_budget_minutes: float | None = 30.0,
    runtime_check_after_rows: int = 3,
    reranker: BaseReranker | None = None,
    matcher: BaseMatcher | None = None,
    extractor: BaseExtractor | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Evaluate the Faiss-only pipeline on held-out known incidents."""
    cols = ColumnConfig()
    capped_sample_size = None if sample_size is not None and sample_size <= 0 else sample_size
    train_df, validation_df, split_summary = build_validation_split(
        known_df=known_df,
        validation_fraction=validation_fraction,
        min_group_count=min_group_count,
        max_validation_rows=capped_sample_size,
        random_seed=random_seed,
        columns=cols,
    )

    if validation_df.empty:
        raise ValueError("Validation split is empty. Lower min_group_count or increase the known data size.")

    majority_gname = str(train_df[cols.gname].value_counts().idxmax())
    _progress(
        progress_callback,
        (
            f"Evaluation split ready: {len(train_df)} training rows, "
            f"{len(validation_df)} validation rows, majority baseline '{majority_gname}'."
        ),
    )

    _progress(progress_callback, "Building a temporary Faiss index from the training split to avoid validation leakage.")
    embeddings = build_embeddings(train_df[cols.summary].tolist(), model_name=embedding_model_name)
    index = build_faiss_index(embeddings)
    _progress(progress_callback, f"Temporary Faiss index built with {index.ntotal} training rows.")

    reranker = reranker or create_reranker(
        backend=reranker_backend,
        model_name=reranker_model_name,
        max_length=max_length,
    )
    matcher = matcher or create_matcher(
        backend=matcher_backend,
        model_name=matcher_model_name,
        max_length=max_length,
    )
    extractor = extractor or create_extractor(
        backend=extractor_backend,
        model_name=extractor_model_name,
    )

    selected_event_ids = validation_df[cols.event_id].tolist()
    existing_predictions = _load_existing_predictions(predictions_path, resume)
    existing_by_eventid: dict[Any, dict[str, Any]] = {}
    for prediction in existing_predictions:
        event_id = prediction.get(cols.event_id)
        if event_id in selected_event_ids and event_id not in existing_by_eventid:
            existing_by_eventid[event_id] = prediction

    ordered_predictions = [existing_by_eventid[event_id] for event_id in selected_event_ids if event_id in existing_by_eventid]
    already_completed = len(ordered_predictions)
    if already_completed:
        _progress(progress_callback, f"Resuming evaluation from {already_completed} saved prediction(s).")

    target_total = len(selected_event_ids)
    original_target_total = target_total
    budget_trimmed = False
    run_start = time.perf_counter()
    processed_this_run = 0

    for _, row in validation_df.iterrows():
        event_id = row[cols.event_id]
        if event_id in existing_by_eventid:
            if len(ordered_predictions) >= target_total:
                break
            continue
        if len(ordered_predictions) >= target_total:
            break

        record_start = time.perf_counter()
        query_text = str(row.get(cols.summary, ""))
        actual_gname = str(row.get(cols.gname, "")).strip()
        query_record = row.to_dict()
        query_record[cols.gname] = "Unknown"

        faiss_candidates = search_faiss(
            query_text=query_text,
            index=index,
            metadata=train_df,
            model_name=embedding_model_name,
            top_k=retrieval_top_k,
        )
        candidate_pool = build_candidate_pool(
            es_candidates=[],
            faiss_candidates=faiss_candidates,
            candidate_pool_size=candidate_pool_size,
        )
        reranked_candidates = reranker.rerank(
            query_text=query_text,
            candidates=candidate_pool,
            top_k=rerank_top_k,
        )
        reasoned = reason_single_result(
            result={
                cols.event_id: event_id,
                "query_summary": query_text,
                "query_record": query_record,
                "es_candidates": [],
                "faiss_candidates": faiss_candidates,
                "candidate_pool": candidate_pool,
                "reranked_candidates": reranked_candidates,
                "mode": "faiss_only",
            },
            matcher=matcher,
            extractor=extractor,
            candidate_limit=candidate_limit,
        )

        predicted_gname = reasoned.get("predicted_gname")
        prediction_seconds = time.perf_counter() - record_start
        reasoned["actual_gname"] = actual_gname
        reasoned["majority_baseline_prediction"] = majority_gname
        reasoned["prediction_seconds"] = float(prediction_seconds)
        reasoned["is_correct"] = _normalize_label(predicted_gname) == _normalize_label(actual_gname)

        existing_by_eventid[event_id] = reasoned
        ordered_predictions = [existing_by_eventid[selected_id] for selected_id in selected_event_ids[:target_total] if selected_id in existing_by_eventid]

        processed_this_run += 1
        completed_total = len(ordered_predictions)
        elapsed_seconds = time.perf_counter() - run_start
        average_seconds = elapsed_seconds / processed_this_run
        remaining_rows = max(target_total - completed_total, 0)
        estimated_remaining_minutes = (average_seconds * remaining_rows) / 60.0

        _progress(
            progress_callback,
            (
                f"Processed {completed_total}/{target_total} validation rows "
                f"| last row {prediction_seconds:.1f}s "
                f"| elapsed {elapsed_seconds / 60.0:.1f}m "
                f"| est remaining {estimated_remaining_minutes:.1f}m"
            ),
        )

        if save_every and processed_this_run % save_every == 0:
            _persist_predictions(ordered_predictions, predictions_path)
            _progress(progress_callback, f"Checkpoint saved after {completed_total} row(s): {predictions_path}")

        if (
            runtime_budget_minutes is not None
            and runtime_budget_minutes > 0
            and not budget_trimmed
            and processed_this_run >= runtime_check_after_rows
            and completed_total < target_total
        ):
            projected_total_minutes = (elapsed_seconds + (average_seconds * remaining_rows)) / 60.0
            if projected_total_minutes > runtime_budget_minutes:
                budget_seconds = runtime_budget_minutes * 60.0
                affordable_total = int(budget_seconds // max(average_seconds, 1e-9))
                affordable_total = max(completed_total, affordable_total)
                affordable_total = min(target_total, affordable_total)
                if affordable_total < target_total:
                    budget_trimmed = True
                    target_total = affordable_total
                    ordered_predictions = [
                        existing_by_eventid[selected_id]
                        for selected_id in selected_event_ids[:target_total]
                        if selected_id in existing_by_eventid
                    ]
                    _persist_predictions(ordered_predictions, predictions_path)
                    _progress(
                        progress_callback,
                        (
                            f"Projected runtime was about {projected_total_minutes:.1f} minutes, "
                            f"so the run was trimmed to {target_total} validation rows "
                            f"to stay near the {runtime_budget_minutes:.1f}-minute budget."
                        ),
                    )

    final_event_ids = selected_event_ids[:target_total]
    final_predictions = [existing_by_eventid[event_id] for event_id in final_event_ids if event_id in existing_by_eventid]
    final_validation_df = validation_df[validation_df[cols.event_id].isin(final_event_ids)].copy()
    final_validation_df["_eval_order"] = pd.Categorical(final_validation_df[cols.event_id], final_event_ids, ordered=True)
    final_validation_df = final_validation_df.sort_values("_eval_order").drop(columns="_eval_order").reset_index(drop=True)

    total_elapsed_seconds = time.perf_counter() - run_start
    status = "completed" if len(final_predictions) == len(final_event_ids) else "partial"
    if budget_trimmed and status == "completed":
        status = "completed_sampled"

    config = {
        "mode": "faiss_only",
        "random_seed": int(random_seed),
        "validation_fraction": float(validation_fraction),
        "min_group_count": int(min_group_count),
        "sample_size": None if capped_sample_size is None else int(capped_sample_size),
        "retrieval_top_k": int(retrieval_top_k),
        "candidate_pool_size": int(candidate_pool_size),
        "rerank_top_k": int(rerank_top_k),
        "candidate_limit": int(candidate_limit),
        "embedding_model_name": embedding_model_name,
        "reranker_backend": reranker.method_name,
        "matcher_backend": matcher.method_name,
        "extractor_backend": extractor.method_name,
        "runtime_budget_minutes": runtime_budget_minutes,
    }
    runtime = {
        "status": status,
        "elapsed_seconds": float(total_elapsed_seconds),
        "elapsed_minutes": float(total_elapsed_seconds / 60.0),
        "requested_validation_rows": int(original_target_total),
        "evaluated_rows": int(len(final_predictions)),
        "already_completed_rows": int(already_completed),
        "processed_this_run": int(processed_this_run),
        "budget_trimmed": budget_trimmed,
    }
    split_summary = dict(split_summary)
    split_summary["validation_rows"] = int(len(final_validation_df))

    summary = summarize_evaluation_results(
        predictions=final_predictions,
        train_df=train_df,
        validation_df=final_validation_df,
        split_summary=split_summary,
        config=config,
        runtime=runtime,
        columns=cols,
    )
    persist_evaluation_artifacts(
        predictions=final_predictions,
        summary=summary,
        predictions_path=predictions_path,
        summary_path=summary_path,
    )
    return summary
