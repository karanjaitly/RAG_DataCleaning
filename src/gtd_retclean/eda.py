from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ColumnConfig, EDA_SUMMARY_FILE, MISSING_VALUE_PROFILE_FILE, ensure_project_dirs
from .data_prep import UNKNOWN_GNAME_MARKERS


def _series_missing_mask(series: pd.Series) -> pd.Series:
    base_mask = series.isna()
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        blank_mask = series.fillna("").astype(str).str.strip().eq("")
        return base_mask | blank_mask
    return base_mask


def build_missing_value_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize missingness per column for the initial GTD EDA milestone."""
    row_count = len(df)
    rows: list[dict[str, Any]] = []

    for column_name in df.columns:
        missing_mask = _series_missing_mask(df[column_name])
        missing_count = int(missing_mask.sum())
        rows.append(
            {
                "column": column_name,
                "missing_count": missing_count,
                "missing_rate": (missing_count / row_count) if row_count else 0.0,
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=["missing_count", "column"],
        ascending=[False, True],
    ).reset_index(drop=True)


def build_eda_summary(
    df: pd.DataFrame,
    columns: ColumnConfig | None = None,
) -> dict[str, Any]:
    """Create a compact EDA summary focused on missing group attribution."""
    cols = columns or ColumnConfig()
    missing_profile = build_missing_value_profile(df)

    summary_series = df[cols.summary].fillna("").astype(str).str.strip()
    gname_series = df[cols.gname].fillna("").astype(str).str.strip().str.lower()

    non_empty_summary_mask = summary_series.ne("")
    unknown_mask = gname_series.isin(UNKNOWN_GNAME_MARKERS)
    known_mask = non_empty_summary_mask & ~unknown_mask
    unknown_summary_mask = non_empty_summary_mask & unknown_mask

    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "non_empty_summary_rows": int(non_empty_summary_mask.sum()),
        "known_attack_rows": int(known_mask.sum()),
        "unknown_attack_rows": int(unknown_summary_mask.sum()),
        "gname_unknown_rate": (float(unknown_summary_mask.sum()) / float(len(df))) if len(df) else 0.0,
        "columns_with_missing_values": int((missing_profile["missing_count"] > 0).sum()),
        "top_missing_columns": missing_profile.head(10).to_dict(orient="records"),
    }


def persist_eda_artifacts(
    summary: dict[str, Any],
    missing_profile: pd.DataFrame,
    summary_path: Path = EDA_SUMMARY_FILE,
    missing_profile_path: Path = MISSING_VALUE_PROFILE_FILE,
) -> tuple[Path, Path]:
    """Write EDA outputs so milestone verification can reuse them."""
    ensure_project_dirs()
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    missing_profile.to_csv(missing_profile_path, index=False)
    return summary_path, missing_profile_path
