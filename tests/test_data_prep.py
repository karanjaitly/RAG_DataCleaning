from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gtd_retclean.data_prep import split_known_unknown


class SplitKnownUnknownTests(unittest.TestCase):
    def test_split_known_unknown_respects_unknown_markers(self) -> None:
        df = pd.DataFrame(
            [
                {"eventid": 1, "summary": "Known case", "gname": "ETA"},
                {"eventid": 2, "summary": "Unknown case", "gname": "Unknown"},
                {"eventid": 3, "summary": "Blank group", "gname": ""},
                {"eventid": 4, "summary": "", "gname": "Some Group"},
            ]
        )

        known_df, unknown_df = split_known_unknown(df)

        self.assertEqual(known_df["eventid"].tolist(), [1])
        self.assertEqual(unknown_df["eventid"].tolist(), [2, 3])


if __name__ == "__main__":
    unittest.main()
