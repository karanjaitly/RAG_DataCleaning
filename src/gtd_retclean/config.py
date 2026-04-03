from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
INDICES_DIR = PROJECT_ROOT / "indices"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODEL_CACHE_DIR = PROJECT_ROOT / ".cache" / "hf_models"

DEFAULT_DATA_FILE = "gtd_6_month.xlsx"
KNOWN_ATTACKS_FILE = OUTPUTS_DIR / "known_attacks.csv"
UNKNOWN_ATTACKS_FILE = OUTPUTS_DIR / "unknown_attacks.csv"
EDA_SUMMARY_FILE = OUTPUTS_DIR / "eda_summary.json"
MISSING_VALUE_PROFILE_FILE = OUTPUTS_DIR / "missing_value_profile.csv"
RETRIEVAL_PREVIEW_FILE = OUTPUTS_DIR / "retrieval_preview.json"
RERANKED_PREVIEW_FILE = OUTPUTS_DIR / "reranked_preview.json"
REASONED_PREVIEW_FILE = OUTPUTS_DIR / "reasoned_preview.json"
MILESTONE_REPORT_FILE = OUTPUTS_DIR / "milestone_verification.json"

FAISS_INDEX_FILE = INDICES_DIR / "gtd_summary_faiss.index"
FAISS_METADATA_FILE = INDICES_DIR / "gtd_summary_faiss_metadata.csv"

DEFAULT_ES_HOST = "http://localhost:9200"
DEFAULT_ES_INDEX = "gtd_known_attacks"

DEFAULT_EMBEDDING_MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_MODEL_DIR = MODEL_CACHE_DIR / "sentence-transformers-all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_MODEL = str(DEFAULT_EMBEDDING_MODEL_DIR)
DEFAULT_RERANKER_MODEL = DEFAULT_EMBEDDING_MODEL
DEFAULT_RERANKER_BACKEND = "late_interaction"
DEFAULT_RETRIEVAL_TOP_K = 5
DEFAULT_CANDIDATE_POOL_SIZE = 8
DEFAULT_RERANK_TOP_K = 3
DEFAULT_MAX_SEQUENCE_LENGTH = 256
DEFAULT_REASONER_MATCHER_REPO = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_REASONER_MATCHER_MODEL_DIR = MODEL_CACHE_DIR / "cross-encoder-ms-marco-MiniLM-L-6-v2"
DEFAULT_REASONER_MATCHER_BACKEND = "cross_encoder_matcher"
DEFAULT_REASONER_MATCHER_MODEL = str(DEFAULT_REASONER_MATCHER_MODEL_DIR)
DEFAULT_REASONER_EXTRACTOR_REPO = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_REASONER_EXTRACTOR_MODEL_DIR = MODEL_CACHE_DIR / "TinyLlama-1.1B-Chat-v1.0"
DEFAULT_REASONER_EXTRACTOR_BACKEND = "llama_extractor"
DEFAULT_REASONER_EXTRACTOR_MODEL = str(DEFAULT_REASONER_EXTRACTOR_MODEL_DIR)
DEFAULT_REASONER_CANDIDATE_LIMIT = 5


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
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
