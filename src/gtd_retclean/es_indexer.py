from __future__ import annotations

from typing import Any

from elasticsearch import Elasticsearch, helpers

from .config import DEFAULT_ES_HOST


def create_client(host: str = DEFAULT_ES_HOST) -> Elasticsearch:
    """Create Elasticsearch client for local development."""
    return Elasticsearch(hosts=[host])


def ensure_index(client: Elasticsearch, index_name: str) -> None:
    """Create index with summary-focused mapping if absent."""
    if client.indices.exists(index=index_name):
        return

    mappings = {
        "properties": {
            "eventid": {"type": "keyword"},
            "gname": {"type": "keyword"},
            "summary": {"type": "text"},
            "country_txt": {"type": "keyword"},
            "region_txt": {"type": "keyword"},
            "attacktype1_txt": {"type": "keyword"},
            "iyear": {"type": "integer"},
        }
    }
    client.indices.create(index=index_name, mappings=mappings)


def iter_bulk_actions(records: list[dict[str, Any]], index_name: str):
    """Yield documents for Elasticsearch bulk indexing."""
    for record in records:
        event_id = str(record.get("eventid", ""))
        yield {
            "_index": index_name,
            "_id": event_id or None,
            "_source": record,
        }


def bulk_index_known_records(
    client: Elasticsearch,
    index_name: str,
    records: list[dict[str, Any]],
) -> tuple[int, int]:
    """Index known attack summaries into Elasticsearch for lexical retrieval."""
    return helpers.bulk(client, iter_bulk_actions(records, index_name))


def search_by_summary(
    client: Elasticsearch,
    index_name: str,
    query_text: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Run BM25 text search against summary field."""
    response = client.search(
        index=index_name,
        size=top_k,
        query={"match": {"summary": {"query": query_text}}},
    )
    return response["hits"]["hits"]
