"""RAG retrieval + generation logic — using HuggingFace Inference API (free)."""

import chromadb
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_PERSIST_DIR, CHROMA_COLLECTION,
    EMBED_MODEL, TOP_K, HF_API_TOKEN, HF_MODEL,
    USE_LOCAL_LLM, LOCAL_LLM_URL, LOCAL_LLM_MODEL
)

_embedder = None
_collection = None
_hf_client = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        _collection = client.get_collection(CHROMA_COLLECTION)
    return _collection


def _get_hf_client():
    global _hf_client
    if _hf_client is None:
        if USE_LOCAL_LLM:
            print(f"--- Using Local LLM via Ollama: {LOCAL_LLM_URL} ---")
            _hf_client = InferenceClient(
                base_url=LOCAL_LLM_URL,
                api_key="docker-model-runner",  # dummy key for local endpoint
            )
        else:
            print("--- Using Cloud HuggingFace Inference ---")
            _hf_client = InferenceClient(
                provider="sambanova",  # free provider confirmed in available list
                api_key=HF_API_TOKEN,
            )
    return _hf_client


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Embed query and fetch top-k relevant chunks from ChromaDB."""
    embedder = _get_embedder()
    collection = _get_collection()

    query_vec = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_vec,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({"text": doc, "metadata": meta, "score": 1 - dist})
    return chunks


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        header = (
            f"[{i}] Source: {meta.get('source','?')} | "
            f"District: {meta.get('district','?')} | "
            f"Category: {meta.get('category','?')} | "
            f"Date: {meta.get('timestamp','?')}"
        )
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


SYSTEM_PROMPT = (
    "You are an intelligent assistant for UP Police Matrix, a social media intelligence platform. "
    "Answer questions based only on the retrieved intelligence data provided in the context. "
    "PAY SPECIAL ATTENTION to district names (like Mathura, Ghaziabad, etc.) mentioned in the context. "
    "If a user asks about a specific district, ensure you scan all provided source chunks for that name. "
    "Be factual and concise. Cite the source, district, and date when relevant. "
    "If the context does not contain information about the specific district or topic requested, say so clearly. "
    "Do not make up information or confuse different districts."
)


def answer(query: str) -> dict:
    """Full RAG pipeline: retrieve -> build context -> generate answer via HuggingFace."""
    chunks = retrieve(query)
    context = build_context(chunks)

    client = _get_hf_client()

    # Use either local model name or cloud model name
    model_id = LOCAL_LLM_MODEL if USE_LOCAL_LLM else "Meta-Llama/Meta-Llama-3.1-8B-Instruct"

    completion = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Context from intelligence database:\n\n{context}"
                    f"\n\n---\n\nQuestion: {query}"
                ),
            },
        ],
    )

    return {
        "answer": completion.choices[0].message.content.strip(),
        "sources": [
            {
                "score": round(c["score"], 3),
                "source": c["metadata"].get("source"),
                "district": c["metadata"].get("district"),
                "category": c["metadata"].get("category"),
                "timestamp": c["metadata"].get("timestamp"),
                "url": c["metadata"].get("url"),
            }
            for c in chunks
        ],
    }