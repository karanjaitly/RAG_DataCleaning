from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gtd_retclean.evaluation import (
    build_validation_split,
    compute_prediction_metrics,
    summarize_evaluation_results,
)


class EvaluationTests(unittest.TestCase):
    def test_build_validation_split_keeps_all_rows_and_preserves_training_examples(self) -> None:
        known_df = pd.DataFrame(
            {
                "eventid": [1, 2, 3, 4, 5, 6],
                "summary": ["a", "b", "c", "d", "e", "f"],
                "gname": ["Group A", "Group A", "Group A", "Group B", "Group B", "Group C"],
            }
        )

        train_df, validation_df, split_summary = build_validation_split(
            known_df=known_df,
            validation_fraction=0.34,
            min_group_count=2,
            max_validation_rows=None,
            random_seed=7,
        )

        self.assertEqual(len(train_df) + len(validation_df), len(known_df))
        self.assertEqual(split_summary["excluded_groups_from_validation"], 1)
        self.assertEqual(split_summary["excluded_validation_rows"], 1)
        self.assertTrue(set(validation_df["gname"]).issubset(set(train_df["gname"])))
        self.assertGreaterEqual(len(train_df[train_df["gname"] == "Group A"]), 1)
        self.assertGreaterEqual(len(train_df[train_df["gname"] == "Group B"]), 1)
        self.assertIn("Group C", set(train_df["gname"]))

    def test_build_validation_split_moves_unused_validation_rows_back_to_training(self) -> None:
        known_df = pd.DataFrame(
            {
                "eventid": [1, 2, 3, 4, 5, 6],
                "summary": ["a", "b", "c", "d", "e", "f"],
                "gname": ["Group A", "Group A", "Group A", "Group B", "Group B", "Group B"],
            }
        )

        train_df, validation_df, split_summary = build_validation_split(
            known_df=known_df,
            validation_fraction=0.5,
            min_group_count=2,
            max_validation_rows=2,
            random_seed=13,
        )

        self.assertEqual(len(validation_df), 2)
        self.assertEqual(len(train_df), 4)
        self.assertTrue(split_summary["sample_capped"])

    def test_compute_prediction_metrics_handles_abstentions(self) -> None:
        metrics = compute_prediction_metrics(
            actual_labels=["Group A", "Group B", "Group C"],
            predicted_labels=["Group A", None, "Group A"],
        )

        self.assertAlmostEqual(metrics["accuracy"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["coverage"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["covered_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["macro_f1"], 2.0 / 9.0)

    def test_summarize_evaluation_results_reports_baseline_lift(self) -> None:
        train_df = pd.DataFrame(
            {
                "eventid": [1, 2, 3, 4],
                "summary": ["a", "b", "c", "d"],
                "gname": ["Group A", "Group A", "Group B", "Group C"],
            }
        )
        validation_df = pd.DataFrame(
            {
                "eventid": [11, 12, 13],
                "summary": ["x", "y", "z"],
                "gname": ["Group A", "Group B", "Group B"],
            }
        )
        predictions = [
            {
                "eventid": 11,
                "actual_gname": "Group A",
                "predicted_gname": "Group A",
                "is_correct": True,
                "confidence_label": "high",
            },
            {
                "eventid": 12,
                "actual_gname": "Group B",
                "predicted_gname": "Group B",
                "is_correct": True,
                "confidence_label": "medium",
            },
            {
                "eventid": 13,
                "actual_gname": "Group B",
                "predicted_gname": None,
                "is_correct": False,
                "confidence_label": "low",
            },
        ]

        summary = summarize_evaluation_results(
            predictions=predictions,
            train_df=train_df,
            validation_df=validation_df,
            split_summary={"training_rows": 4, "validation_rows": 3},
            config={"mode": "faiss_only"},
            runtime={"status": "completed", "elapsed_seconds": 12.0},
        )

        self.assertAlmostEqual(summary["metrics"]["pipeline"]["accuracy"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["metrics"]["majority_baseline"]["accuracy"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["metrics"]["accuracy_lift"], 1.0 / 3.0)
        self.assertEqual(summary["label_distribution"]["majority_gname"], "Group A")


if __name__ == "__main__":
    unittest.main()
