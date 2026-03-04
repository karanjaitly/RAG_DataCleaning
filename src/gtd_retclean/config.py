from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
INDICES_DIR = PROJECT_ROOT / "indices"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DEFAULT_DATA_FILE = "gtd_6_month.xlsx"
KNOWN_ATTACKS_FILE = OUTPUTS_DIR / "known_attacks.csv"
UNKNOWN_ATTACKS_FILE = OUTPUTS_DIR / "unknown_attacks.csv"

FAISS_INDEX_FILE = INDICES_DIR / "gtd_summary_faiss.index"
FAISS_METADATA_FILE = INDICES_DIR / "gtd_summary_faiss_metadata.csv"

DEFAULT_ES_HOST = "http://localhost:9200"
DEFAULT_ES_INDEX = "gtd_known_attacks"

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class ColumnConfig:
    """Centralized column names used by the pipeline."""

    event_id: str = "eventid"
    summary: str = "summary"
    gname: str = "gname"


def ensure_project_dirs() -> None:
    """Create output/index folders if they do not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDICES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
