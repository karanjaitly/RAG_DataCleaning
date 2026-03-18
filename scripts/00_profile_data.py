from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gtd_retclean.eda import build_eda_summary, build_missing_value_profile, persist_eda_artifacts
from gtd_retclean.data_prep import load_gtd_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GTD EDA for the week 1-2 milestone.")
    parser.add_argument("--data-path", default=None, help="Path to GTD XLSX file.")
    args = parser.parse_args()

    df = load_gtd_data(args.data_path)
    summary = build_eda_summary(df)
    missing_profile = build_missing_value_profile(df)
    summary_path, missing_profile_path = persist_eda_artifacts(summary, missing_profile)

    print(f"Rows profiled: {summary['row_count']:,}")
    print(f"Columns profiled: {summary['column_count']:,}")
    print(f"Unknown attack rows: {summary['unknown_attack_rows']:,}")
    print(f"EDA summary: {summary_path}")
    print(f"Missing value profile: {missing_profile_path}")


if __name__ == "__main__":
    main()
