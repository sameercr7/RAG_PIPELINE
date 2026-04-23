import time
import json
import os
import concurrent.futures
from threading import Lock
import chromadb
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

from config import (
    CHROMA_PERSIST_DIR, CHROMA_COLLECTION,
    EMBED_MODEL, INGEST_BATCH_SIZE, STATE_FILE
)
from db import fetch_pending_by_source, update_ingestion_status
from ingest import row_to_text, row_to_metadata

# Global thread safety lock for ChromaDB write operations
db_lock = Lock()

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"ingestion_enabled": False, "source_config": []}
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except:
        return {"ingestion_enabled": False, "source_config": []}

def process_source_batch(source_name, embedder, collection):
    """Processes a single batch for a specific source."""
    rows = fetch_pending_by_source(source_name, limit=100)
    if not rows:
        return 0

    print(f"--- [Worker] Ingesting {len(rows)} posts from {source_name} ---")
    
    texts, ids, metadatas = [], [], []
    for row in rows:
        text = row_to_text(row)
        if not text.strip():
            continue
        texts.append(text)
        ids.append(f"ad_{row['id']}")
        metadatas.append(row_to_metadata(row))

    if texts:
        # Generate embeddings
        embeddings = embedder.encode(texts, batch_size=32, show_progress_bar=False).tolist()
        
        # Thread-safe upsert to ChromaDB
        with db_lock:
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
        
        # Mark as ingested in MySQL
        db_ids = [row['id'] for row in rows]
        update_ingestion_status(db_ids, "INGESTED")
        
    return len(rows)

def run_worker():
    print("--- Ingestion Worker (v2) Started ---")
    embedder = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    while True:
        state = load_state()
        
        if not state.get("ingestion_enabled", False):
            time.sleep(5)
            continue

        source_configs = state.get("source_config", [])
        total_processed_in_loop = 0
        
        # Follow priorities sequentially by iterating through the ordered list
        for cfg in source_configs:
            source_name = cfg.get("name")
            is_enabled = cfg.get("enabled", True)
            
            if not is_enabled:
                continue
                
            count = process_source_batch(source_name, embedder, collection)
            total_processed_in_loop += count
            
            if count >= 100:
                # If we processed a full batch, we stay at the top of priorities 
                # (restart loop to check if higher-priority sources have new data)
                break 

        if total_processed_in_loop == 0:
            time.sleep(5) # No data found in any enabled source, wait a bit
        else:
            time.sleep(1) # Small gap after processing a batch

if __name__ == "__main__":
    run_worker()
