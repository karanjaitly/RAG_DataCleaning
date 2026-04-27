from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gtd_retclean.reasoner import (
    EvidenceVoteExtractor,
    FieldAwareMatcher,
    LocalCrossEncoderMatcher,
    LocalLlamaExtractor,
    LocalRobertaMatcher,
    reason_single_result,
    reason_over_reranked_results,
)


class ReasonerTests(unittest.TestCase):
    def test_field_aware_matcher_prefers_structured_alignment(self) -> None:
        query_record = {
            "eventid": 900,
            "summary": "Assailants threw a grenade at a CRPF camp in Jammu and Kashmir.",
            "country_txt": "India",
            "region_txt": "South Asia",
            "provstate": "Jammu and Kashmir",
            "attacktype1_txt": "Bombing/Explosion",
            "targtype1_txt": "Police",
            "weaptype1_txt": "Explosives",
        }
        candidates = [
            {
                "eventid": 101,
                "gname": "Jaish-e-Mohammed (JeM)",
                "summary": "Militants threw a grenade at a CRPF camp in Pulwama, Jammu and Kashmir.",
                "country_txt": "India",
                "region_txt": "South Asia",
                "provstate": "Jammu and Kashmir",
                "attacktype1_txt": "Bombing/Explosion",
                "targtype1_txt": "Police",
                "weaptype1_txt": "Explosives",
                "rerank_rank": 2,
            },
            {
                "eventid": 202,
                "gname": "Sinai Province of the Islamic State",
                "summary": "Explosive device hit a military vehicle in North Sinai.",
                "country_txt": "Egypt",
                "region_txt": "Middle East & North Africa",
                "provstate": "North Sinai",
                "attacktype1_txt": "Bombing/Explosion",
                "targtype1_txt": "Military",
                "weaptype1_txt": "Explosives",
                "rerank_rank": 1,
            },
        ]

        matcher = FieldAwareMatcher()
        ranked = matcher.match(query_record=query_record, candidates=candidates)

        self.assertEqual(ranked[0]["eventid"], 101)
        self.assertGreater(ranked[0]["structured_match_score"], ranked[1]["structured_match_score"])
        self.assertIn("same country: India", ranked[0]["matcher_evidence"])

    def test_evidence_vote_extractor_prefers_group_consensus(self) -> None:
        extractor = EvidenceVoteExtractor()
        matched_candidates = [
            {"eventid": 11, "gname": "Group A", "matcher_score": 0.82, "match_features": {"country_txt": 1.0}},
            {"eventid": 12, "gname": "Group A", "matcher_score": 0.63, "match_features": {"attacktype1_txt": 1.0}},
            {"eventid": 13, "gname": "Group B", "matcher_score": 0.41, "match_features": {"country_txt": 1.0}},
        ]

        decision = extractor.extract(query_record={"summary": "query"}, matched_candidates=matched_candidates)

        self.assertEqual(decision["predicted_gname"], "Group A")
        self.assertEqual(decision["decision_type"], "group_consensus")
        self.assertEqual(decision["supporting_event_ids"], [11, 12])
        self.assertGreater(decision["confidence"], 0.5)

    def test_reason_over_reranked_results_uses_query_record(self) -> None:
        reranked_results = [
            {
                "eventid": 900,
                "query_summary": "Assailants threw a grenade at a CRPF camp in Jammu and Kashmir.",
                "query_record": {
                    "eventid": 900,
                    "summary": "Assailants threw a grenade at a CRPF camp in Jammu and Kashmir.",
                    "country_txt": "India",
                    "region_txt": "South Asia",
                    "provstate": "Jammu and Kashmir",
                    "attacktype1_txt": "Bombing/Explosion",
                    "targtype1_txt": "Police",
                },
                "reranked_candidates": [
                    {
                        "eventid": 101,
                        "gname": "Jaish-e-Mohammed (JeM)",
                        "summary": "Militants threw a grenade at a CRPF camp in Pulwama, Jammu and Kashmir.",
                        "country_txt": "India",
                        "region_txt": "South Asia",
                        "provstate": "Jammu and Kashmir",
                        "attacktype1_txt": "Bombing/Explosion",
                        "targtype1_txt": "Police",
                        "rerank_rank": 1,
                        "reranker_score": 0.7,
                    },
                    {
                        "eventid": 202,
                        "gname": "Lashkar-e-Taiba (LeT)",
                        "summary": "A bombing targeted police in Srinagar.",
                        "country_txt": "India",
                        "region_txt": "South Asia",
                        "provstate": "Jammu and Kashmir",
                        "attacktype1_txt": "Bombing/Explosion",
                        "targtype1_txt": "Police",
                        "rerank_rank": 2,
                        "reranker_score": 0.4,
                    },
                ],
            }
        ]

        reasoned = reason_over_reranked_results(
            reranked_results=reranked_results,
            matcher=FieldAwareMatcher(),
            extractor=EvidenceVoteExtractor(),
            candidate_limit=2,
        )

        self.assertEqual(reasoned[0]["status"], "predicted")
        self.assertEqual(reasoned[0]["predicted_gname"], "Jaish-e-Mohammed (JeM)")
        self.assertEqual(reasoned[0]["matched_candidates"][0]["matcher_rank"], 1)

    def test_reason_single_result_returns_enriched_decision(self) -> None:
        result = {
            "eventid": 900,
            "query_summary": "Assailants threw a grenade at a CRPF camp in Jammu and Kashmir.",
            "query_record": {
                "eventid": 900,
                "summary": "Assailants threw a grenade at a CRPF camp in Jammu and Kashmir.",
                "country_txt": "India",
                "region_txt": "South Asia",
                "provstate": "Jammu and Kashmir",
            },
            "reranked_candidates": [
                {
                    "eventid": 101,
                    "gname": "Jaish-e-Mohammed (JeM)",
                    "summary": "Militants threw a grenade at a CRPF camp in Pulwama, Jammu and Kashmir.",
                    "country_txt": "India",
                    "region_txt": "South Asia",
                    "provstate": "Jammu and Kashmir",
                }
            ],
        }

        reasoned = reason_single_result(
            result=result,
            matcher=FieldAwareMatcher(),
            extractor=EvidenceVoteExtractor(),
            candidate_limit=1,
        )

        self.assertEqual(reasoned["predicted_gname"], "Jaish-e-Mohammed (JeM)")
        self.assertEqual(reasoned["matched_candidates"][0]["matcher_rank"], 1)
        self.assertEqual(reasoned["matcher_method"], "field_weighted")

    def test_local_llama_extractor_accepts_generated_json(self) -> None:
        extractor = LocalLlamaExtractor(
            model_name="",
            generator=lambda prompt: (
                '{"predicted_gname": "Group A", "confidence_label": "high", '
                '"rationale": "The top candidates consistently point to Group A.", '
                '"support_event_ids": [11, 12]}'
            ),
        )
        matched_candidates = [
            {"eventid": 11, "gname": "Group A", "matcher_score": 0.82, "match_features": {"country_txt": 1.0}},
            {"eventid": 12, "gname": "Group A", "matcher_score": 0.63, "match_features": {"attacktype1_txt": 1.0}},
        ]

        decision = extractor.extract(query_record={"summary": "query"}, matched_candidates=matched_candidates)

        self.assertEqual(decision["predicted_gname"], "Group A")
        self.assertEqual(decision["extractor_method"], "llama_extractor")
        self.assertEqual(decision["supporting_event_ids"], [11, 12])
        self.assertEqual(decision["confidence_label"], "high")

    def test_local_roberta_matcher_supports_custom_scorer(self) -> None:
        matcher = LocalRobertaMatcher(
            model_name="",
            scorer=lambda query_record, candidates: [0.1, 0.9],
        )
        ranked = matcher.match(
            query_record={"summary": "query summary"},
            candidates=[
                {"eventid": 1, "summary": "weak match", "gname": "Group A"},
                {"eventid": 2, "summary": "strong match", "gname": "Group B"},
            ],
        )

        self.assertEqual(ranked[0]["eventid"], 2)
        self.assertEqual(ranked[0]["matcher_method"], "roberta_matcher")
        self.assertIn("transformer_match_score", ranked[0])

    def test_local_cross_encoder_matcher_supports_custom_scorer(self) -> None:
        matcher = LocalCrossEncoderMatcher(
            model_name="",
            scorer=lambda query_record, candidates: [0.2, 0.8],
        )
        ranked = matcher.match(
            query_record={"summary": "query summary"},
            candidates=[
                {"eventid": 1, "summary": "weak match", "gname": "Group A"},
                {"eventid": 2, "summary": "strong match", "gname": "Group B"},
            ],
        )

        self.assertEqual(ranked[0]["eventid"], 2)
        self.assertEqual(ranked[0]["matcher_method"], "cross_encoder_matcher")
        self.assertIn("transformer_match_score", ranked[0])


if __name__ == "__main__":
    unittest.main()
