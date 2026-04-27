from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from .config import ColumnConfig, DEFAULT_DATA_FILE, OUTPUTS_DIR, ensure_project_dirs


UNKNOWN_GNAME_MARKERS = {
    "unknown",
    "nan",
    "",
    "na",
    "n/a",
    "none",
    "unidentified",
}


def resolve_data_path(data_path: str | Path | None = None) -> Path:
    """Resolve GTD input location, preferring data/ then project root."""
    if data_path:
        candidate = Path(data_path)
        if candidate.exists():
            return candidate

    data_dir_candidate = Path("data") / DEFAULT_DATA_FILE
    root_candidate = Path(DEFAULT_DATA_FILE)
    if data_dir_candidate.exists():
        return data_dir_candidate
    if root_candidate.exists():
        return root_candidate

    raise FileNotFoundError(
        f"Could not find GTD file. Expected '{data_dir_candidate}' or '{root_candidate}'."
    )


def load_gtd_data(data_path: str | Path | None = None) -> pd.DataFrame:
    """Load the GTD spreadsheet and return a DataFrame."""
    source = resolve_data_path(data_path)
    return pd.read_excel(source)


def split_known_unknown(
    df: pd.DataFrame,
    columns: ColumnConfig | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split attacks into known (data lake) and unknown (dirty) groups.

    RAG step: known records become retrieval corpus; unknown records are query set.
    """
    cols = columns or ColumnConfig()

    required = [cols.event_id, cols.summary, cols.gname]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    cleaned = df.copy()
    cleaned[cols.summary] = cleaned[cols.summary].fillna("").astype(str).str.strip()
    cleaned[cols.gname] = cleaned[cols.gname].fillna("").astype(str).str.strip()

    has_summary = cleaned[cols.summary] != ""
    gname_lower = cleaned[cols.gname].str.lower()

    unknown_mask = has_summary & gname_lower.isin(UNKNOWN_GNAME_MARKERS)
    known_mask = has_summary & ~unknown_mask

    known_df = cleaned.loc[known_mask].reset_index(drop=True)
    unknown_df = cleaned.loc[unknown_mask].reset_index(drop=True)
    return known_df, unknown_df


def persist_splits(known_df: pd.DataFrame, unknown_df: pd.DataFrame) -> tuple[Path, Path]:
    """Save split datasets to outputs/ for reuse by indexers."""
    ensure_project_dirs()
    known_path = OUTPUTS_DIR / "known_attacks.csv"
    unknown_path = OUTPUTS_DIR / "unknown_attacks.csv"
    known_df.to_csv(known_path, index=False)
    unknown_df.to_csv(unknown_path, index=False)
    return known_path, unknown_path
