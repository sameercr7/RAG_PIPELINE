"""
Run this once (or periodically) to index analyzed_data into ChromaDB.
Usage:
    python ingest.py                  # index all rows
    python ingest.py --limit 5000     # index only first 5000 rows (for testing)
    python ingest.py --reset          # wipe collection and re-index
"""

import argparse
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import (
    CHROMA_PERSIST_DIR, CHROMA_COLLECTION,
    EMBED_MODEL, INGEST_BATCH_SIZE,
)
from db import fetch_analyzed_data_batch, get_total_analyzed_rows


def row_to_text(row: dict) -> str:
    """Convert a DB row into a single text string for embedding."""
    parts = []

    if row.get("input_text"):
        parts.append(row["input_text"])
    if row.get("contextual_understanding"):
        parts.append(f"Context: {row['contextual_understanding']}")
    if row.get("topic_title"):
        parts.append(f"Topic: {row['topic_title']}")
    if row.get("incidents"):
        parts.append(f"Incidents: {row['incidents']}")
    if row.get("events"):
        parts.append(f"Events: {row['events']}")
    if row.get("broad_category"):
        parts.append(f"Category: {row['broad_category']}")
    if row.get("sub_category"):
        parts.append(f"Sub-category: {row['sub_category']}")
    if row.get("district_names"):
        parts.append(f"Districts: {row['district_names']}")
    if row.get("thana_names"):
        parts.append(f"Thana: {row['thana_names']}")
    if row.get("person_names"):
        parts.append(f"Persons: {row['person_names']}")
    if row.get("organisation_names"):
        parts.append(f"Organisations: {row['organisation_names']}")
    if row.get("sentiment_label"):
        parts.append(f"Sentiment: {row['sentiment_label']}")

    return " | ".join(parts)


def row_to_metadata(row: dict) -> dict:
    """Extract lightweight metadata (ChromaDB only accepts str/int/float/bool)."""
    return {
        "db_id": row["id"],
        "source": str(row.get("post_bank_source") or ""),
        "author": str(row.get("post_bank_author_name") or ""),
        "timestamp": str(row.get("post_bank_post_timestamp") or ""),
        "url": str(row.get("post_bank_post_url") or ""),
        "district": str(row.get("district_names") or ""),
        "category": str(row.get("broad_category") or ""),
        "sentiment": str(row.get("sentiment_label") or ""),
    }


def main(limit: int | None = None, reset: bool = False):
    print(f"Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    if reset:
        try:
            client.delete_collection(CHROMA_COLLECTION)
            print("Existing collection deleted.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    total = get_total_analyzed_rows()
    if limit:
        total = min(total, limit)

    print(f"Total rows to index: {total}")

    offset = 0
    indexed = 0

    with tqdm(total=total, unit="rows") as pbar:
        while offset < total:
            rows = fetch_analyzed_data_batch(offset, INGEST_BATCH_SIZE)
            if not rows:
                break

            texts, ids, metadatas = [], [], []
            for row in rows:
                text = row_to_text(row)
                if not text.strip():
                    continue
                texts.append(text)
                ids.append(f"ad_{row['id']}")
                metadatas.append(row_to_metadata(row))

            if texts:
                embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=False).tolist()
                collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                )
                indexed += len(texts)

            offset += INGEST_BATCH_SIZE
            pbar.update(len(rows))

    print(f"\nDone. Indexed {indexed} documents into ChromaDB at '{CHROMA_PERSIST_DIR}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Max rows to index (for testing)")
    parser.add_argument("--reset", action="store_true", help="Wipe and re-index")
    args = parser.parse_args()
    main(limit=args.limit, reset=args.reset)