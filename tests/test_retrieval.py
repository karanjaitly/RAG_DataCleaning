from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gtd_retclean.candidate_pool import build_candidate_pool


class RetrievalCandidatePoolTests(unittest.TestCase):
    def test_build_candidate_pool_merges_duplicate_event_ids(self) -> None:
        es_candidates = [
            {"eventid": 101, "summary": "Truck bombing downtown", "score": 5.0},
            {"eventid": 202, "summary": "Airport attack", "score": 3.0},
        ]
        faiss_candidates = [
            {"eventid": 101, "summary": "Truck bombing downtown", "score": 0.9},
            {"eventid": 303, "summary": "Embassy attack", "score": 0.8},
        ]

        pool = build_candidate_pool(es_candidates, faiss_candidates)

        self.assertEqual(len(pool), 3)
        self.assertEqual(pool[0]["eventid"], 101)
        self.assertEqual(pool[0]["source_count"], 2)
        self.assertEqual(set(pool[0]["source_engines"]), {"elasticsearch", "faiss"})


if __name__ == "__main__":
    unittest.main()
