from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gtd_retclean.config import MILESTONE_REPORT_FILE, RETRIEVAL_PREVIEW_FILE
from gtd_retclean.milestones import persist_verification_report, verify_previous_work


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify project milestones completed through week 4.")
    parser.add_argument("--data-path", default=None, help="Path to GTD XLSX file.")
    parser.add_argument(
        "--retrieval-preview-path",
        default=str(RETRIEVAL_PREVIEW_FILE),
        help="Path to saved retrieval preview JSON.",
    )
    parser.add_argument(
        "--check-es",
        action="store_true",
        help="Attempt a live Elasticsearch health check during verification.",
    )
    parser.add_argument("--es-host", default="http://localhost:9200", help="Elasticsearch host URL")
    parser.add_argument(
        "--output-path",
        default=str(MILESTONE_REPORT_FILE),
        help="Where to save milestone verification JSON.",
    )
    args = parser.parse_args()

    report = verify_previous_work(
        data_path=args.data_path,
        retrieval_preview_path=Path(args.retrieval_preview_path),
        check_elasticsearch=args.check_es,
        es_host=args.es_host,
    )
    output_path = persist_verification_report(report, Path(args.output_path))

    print(f"Overall status: {report['overall_status']}")
    print(f"Week 1-2: {report['weeks']['week_1_2']}")
    print(f"Week 3-4: {report['weeks']['week_3_4']}")
    print(f"Saved verification report: {output_path}")


if __name__ == "__main__":
    main()
