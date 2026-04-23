import chromadb
from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION

def check_mathura():
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    col = client.get_collection(CHROMA_COLLECTION)
    
    # Check by document content (text search)
    res_text = col.get(where_document={"$contains": "Mathura"})
    print(f"Count of posts mentioning 'Mathura' in text: {len(res_text['ids'])}")
    
    # Check by metadata field 'district'
    # Note: ChromaDB $contains operator works for strings in some versions, 
    # but here we can just fetch and filter or check for exact match if it was stored as list.
    # Actually, in ingest.py, district was stored as str(row.get("district_names") or "").
    
    res_meta = col.get(where={"district": "Mathura"}) # This only works for exact matches
    print(f"Count of posts with exact district 'Mathura': {len(res_meta['ids'])}")
    
    # Partial match check
    all_data = col.get(include=["metadatas"])
    partial_meta = [m for m in all_data['metadatas'] if "Mathura" in (m.get("district") or "")]
    print(f"Count of posts with 'Mathura' somewhere in district metadata: {len(partial_meta)}")

if __name__ == "__main__":
    check_mathura()
