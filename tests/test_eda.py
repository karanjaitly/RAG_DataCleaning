from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gtd_retclean.eda import build_eda_summary, build_missing_value_profile


class EdaTests(unittest.TestCase):
    def test_build_eda_summary_counts_unknown_rows(self) -> None:
        df = pd.DataFrame(
            [
                {"eventid": 1, "summary": "A", "gname": "Group A", "city": "X"},
                {"eventid": 2, "summary": "B", "gname": "Unknown", "city": ""},
                {"eventid": 3, "summary": "C", "gname": "", "city": None},
            ]
        )

        summary = build_eda_summary(df)

        self.assertEqual(summary["row_count"], 3)
        self.assertEqual(summary["unknown_attack_rows"], 2)
        self.assertEqual(summary["known_attack_rows"], 1)

    def test_missing_value_profile_treats_blank_strings_as_missing(self) -> None:
        df = pd.DataFrame(
            [
                {"summary": "A", "city": "X"},
                {"summary": "B", "city": ""},
                {"summary": "C", "city": None},
            ]
        )

        profile = build_missing_value_profile(df)
        city_row = profile.loc[profile["column"] == "city"].iloc[0]

        self.assertEqual(int(city_row["missing_count"]), 2)


if __name__ == "__main__":
    unittest.main()
