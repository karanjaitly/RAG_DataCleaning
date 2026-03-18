from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import pandas as pd

from .config import (
    EDA_SUMMARY_FILE,
    FAISS_INDEX_FILE,
    FAISS_METADATA_FILE,
    KNOWN_ATTACKS_FILE,
    MILESTONE_REPORT_FILE,
    UNKNOWN_ATTACKS_FILE,
)
from .data_prep import load_gtd_data, split_known_unknown
from .eda import build_eda_summary, build_missing_value_profile, persist_eda_artifacts
from .es_indexer import create_client


def _make_check(week: str, name: str, status: str, details: dict[str, Any]) -> dict[str, Any]:
    return {
        "week": week,
        "name": name,
        "status": status,
        "details": details,
    }


def _week_status(checks: list[dict[str, Any]], week: str) -> str:
    week_checks = [check for check in checks if check["week"] == week]
    statuses = {check["status"] for check in week_checks}
    if "failed" in statuses:
        return "failed"
    if "partial" in statuses or "skipped" in statuses:
        return "partial"
    return "passed"


def verify_previous_work(
    data_path: str | Path | None = None,
    known_path: Path = KNOWN_ATTACKS_FILE,
    unknown_path: Path = UNKNOWN_ATTACKS_FILE,
    faiss_index_path: Path = FAISS_INDEX_FILE,
    faiss_metadata_path: Path = FAISS_METADATA_FILE,
    retrieval_preview_path: Path | None = None,
    check_elasticsearch: bool = False,
    es_host: str = "http://localhost:9200",
) -> dict[str, Any]:
    """Verify weeks 1-4 using reusable checks and existing artifacts."""
    checks: list[dict[str, Any]] = []
    df = load_gtd_data(data_path)

    missing_profile = build_missing_value_profile(df)
    eda_summary = build_eda_summary(df)
    eda_path, missing_profile_path = persist_eda_artifacts(
        summary=eda_summary,
        missing_profile=missing_profile,
        summary_path=EDA_SUMMARY_FILE,
    )
    checks.append(
        _make_check(
            week="1-2",
            name="dataset_eda_profile",
            status="passed",
            details={
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "eda_summary_path": str(eda_path),
                "missing_profile_path": str(missing_profile_path),
            },
        )
    )

    expected_known, expected_unknown = split_known_unknown(df)
    if known_path.exists() and unknown_path.exists():
        known_df = pd.read_csv(known_path)
        unknown_df = pd.read_csv(unknown_path)
        splits_match = len(known_df) == len(expected_known) and len(unknown_df) == len(expected_unknown)
        checks.append(
            _make_check(
                week="3-4",
                name="split_outputs",
                status="passed" if splits_match else "failed",
                details={
                    "known_rows_expected": int(len(expected_known)),
                    "known_rows_found": int(len(known_df)),
                    "unknown_rows_expected": int(len(expected_unknown)),
                    "unknown_rows_found": int(len(unknown_df)),
                },
            )
        )
    else:
        checks.append(
            _make_check(
                week="3-4",
                name="split_outputs",
                status="failed",
                details={
                    "known_exists": known_path.exists(),
                    "unknown_exists": unknown_path.exists(),
                },
            )
        )

    if faiss_index_path.exists() and faiss_metadata_path.exists():
        index = faiss.read_index(str(faiss_index_path))
        metadata = pd.read_csv(faiss_metadata_path)
        faiss_matches = index.ntotal == len(metadata) == len(expected_known)
        checks.append(
            _make_check(
                week="3-4",
                name="faiss_artifacts",
                status="passed" if faiss_matches else "failed",
                details={
                    "faiss_ntotal": int(index.ntotal),
                    "metadata_rows": int(len(metadata)),
                    "expected_known_rows": int(len(expected_known)),
                },
            )
        )
    else:
        checks.append(
            _make_check(
                week="3-4",
                name="faiss_artifacts",
                status="failed",
                details={
                    "index_exists": faiss_index_path.exists(),
                    "metadata_exists": faiss_metadata_path.exists(),
                },
            )
        )

    retrieval_path = retrieval_preview_path
    if retrieval_path is not None and retrieval_path.exists():
        payload = json.loads(retrieval_path.read_text(encoding="utf-8"))
        has_candidates = bool(payload) and any(
            item.get("candidate_pool") or item.get("faiss_candidates") or item.get("es_candidates")
            for item in payload
        )
        checks.append(
            _make_check(
                week="3-4",
                name="retrieval_preview",
                status="passed" if has_candidates else "failed",
                details={
                    "preview_path": str(retrieval_path),
                    "records": int(len(payload)),
                },
            )
        )
    else:
        checks.append(
            _make_check(
                week="3-4",
                name="retrieval_preview",
                status="partial",
                details={
                    "preview_path": str(retrieval_path) if retrieval_path is not None else "",
                    "message": "No retrieval preview was supplied for verification.",
                },
            )
        )

    if check_elasticsearch:
        client = create_client(es_host)
        es_available = bool(client.ping())
        checks.append(
            _make_check(
                week="3-4",
                name="elasticsearch_health",
                status="passed" if es_available else "partial",
                details={"es_host": es_host, "reachable": es_available},
            )
        )
    else:
        checks.append(
            _make_check(
                week="3-4",
                name="elasticsearch_health",
                status="skipped",
                details={"message": "Skipped live Elasticsearch verification."},
            )
        )

    week_statuses = {
        "week_1_2": _week_status(checks, "1-2"),
        "week_3_4": _week_status(checks, "3-4"),
    }
    overall_status = "passed" if all(status == "passed" for status in week_statuses.values()) else "partial"
    return {
        "overall_status": overall_status,
        "weeks": week_statuses,
        "checks": checks,
    }


def persist_verification_report(
    report: dict[str, Any],
    output_path: Path = MILESTONE_REPORT_FILE,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path
