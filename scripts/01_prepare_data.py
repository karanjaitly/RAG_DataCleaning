from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gtd_retclean.config import ColumnConfig
from gtd_retclean.data_prep import load_gtd_data, persist_splits, split_known_unknown


def main() -> None:
    parser = argparse.ArgumentParser(description="Split GTD data into known and unknown records.")
    parser.add_argument("--data-path", default=None, help="Path to GTD XLSX file.")
    args = parser.parse_args()

    df = load_gtd_data(args.data_path)
    known_df, unknown_df = split_known_unknown(df, ColumnConfig())
    known_path, unknown_path = persist_splits(known_df, unknown_df)

    print(f"Loaded rows: {len(df):,}")
    print(f"Known attacks: {len(known_df):,} -> {known_path}")
    print(f"Unknown attacks: {len(unknown_df):,} -> {unknown_path}")


if __name__ == "__main__":
    main()
