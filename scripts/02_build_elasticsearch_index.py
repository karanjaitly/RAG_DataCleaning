from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gtd_retclean.config import DEFAULT_ES_INDEX, KNOWN_ATTACKS_FILE
from gtd_retclean.es_indexer import bulk_index_known_records, create_client, ensure_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Index known GTD summaries into Elasticsearch.")
    parser.add_argument("--es-host", default="http://localhost:9200", help="Elasticsearch host URL")
    parser.add_argument("--index", default=DEFAULT_ES_INDEX, help="Elasticsearch index name")
    parser.add_argument("--known-path", default=str(KNOWN_ATTACKS_FILE), help="Path to known attacks CSV")
    args = parser.parse_args()

    known_df = pd.read_csv(args.known_path)
    records = known_df.fillna("").to_dict(orient="records")

    client = create_client(args.es_host)
    ensure_index(client, args.index)
    indexed, errors = bulk_index_known_records(client, args.index, records)

    print(f"Indexed records: {indexed}")
    print(f"Bulk errors: {errors}")


if __name__ == "__main__":
    main()
