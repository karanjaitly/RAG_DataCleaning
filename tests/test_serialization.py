from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gtd_retclean.serialization import to_json_ready


class SerializationTests(unittest.TestCase):
    def test_to_json_ready_replaces_nan_with_none(self) -> None:
        payload = {"score": float("nan"), "nested": [1.0, float("inf"), {"value": -math.inf}]}

        converted = to_json_ready(payload)

        self.assertIsNone(converted["score"])
        self.assertIsNone(converted["nested"][1])
        self.assertIsNone(converted["nested"][2]["value"])


if __name__ == "__main__":
    unittest.main()
