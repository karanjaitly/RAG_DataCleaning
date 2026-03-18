from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gtd_retclean.reranker import (
    LateInteractionReranker,
    TokenOverlapReranker,
    late_interaction_score,
    rerank_retrieval_results,
)


class FakeEncoder:
    def __init__(self, mapping: dict[str, np.ndarray]) -> None:
        self.mapping = mapping

    def encode_texts(self, texts: list[str]) -> list[np.ndarray]:
        return [self.mapping[text] for text in texts]


class RerankerTests(unittest.TestCase):
    def test_late_interaction_score_prefers_better_token_alignment(self) -> None:
        query = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
        strong_doc = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
        weak_doc = np.array([[1.0, 0.0], [1.0, 0.0]], dtype="float32")

        self.assertGreater(late_interaction_score(query, strong_doc), late_interaction_score(query, weak_doc))

    def test_late_interaction_reranker_uses_encoder_outputs(self) -> None:
        mapping = {
            "truck bombing downtown": np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32"),
            "candidate_a": np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32"),
            "candidate_b": np.array([[1.0, 0.0], [1.0, 0.0]], dtype="float32"),
        }
        reranker = LateInteractionReranker(encoder=FakeEncoder(mapping))
        candidates = [
            {"eventid": 1, "summary": "candidate_b", "reciprocal_rank_score": 1.0},
            {"eventid": 2, "summary": "candidate_a", "reciprocal_rank_score": 0.5},
        ]

        reranked = reranker.rerank("truck bombing downtown", candidates)

        self.assertEqual(reranked[0]["eventid"], 2)
        self.assertEqual(reranked[0]["rerank_rank"], 1)

    def test_token_overlap_reranker_prefers_shared_terms(self) -> None:
        reranker = TokenOverlapReranker()
        candidates = [
            {"eventid": 1, "summary": "truck bombing downtown market"},
            {"eventid": 2, "summary": "airport security checkpoint"},
        ]

        reranked = reranker.rerank("truck bombing downtown", candidates)

        self.assertEqual(reranked[0]["eventid"], 1)

    def test_rerank_retrieval_results_uses_candidate_pool(self) -> None:
        reranker = TokenOverlapReranker()
        retrieval_results = [
            {
                "eventid": 9,
                "query_summary": "truck bombing downtown",
                "candidate_pool": [
                    {"eventid": 1, "summary": "truck bombing downtown market"},
                    {"eventid": 2, "summary": "airport security checkpoint"},
                ],
            }
        ]

        reranked = rerank_retrieval_results(retrieval_results, reranker, top_k=1)

        self.assertEqual(len(reranked[0]["reranked_candidates"]), 1)
        self.assertEqual(reranked[0]["reranked_candidates"][0]["eventid"], 1)


if __name__ == "__main__":
    unittest.main()
