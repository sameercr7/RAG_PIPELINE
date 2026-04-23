from rag import answer

def test_pipeline():
    query = "What is the latest intelligence from UP Police Matrix?"
    print(f"\nAsking: {query}")
    
    try:
        result = answer(query)
        print("\n--- AI ANSWER ---")
        print(result["answer"])
        print("\n--- SOURCES ---")
        for s in result["sources"]:
            print(f"- [{s['score']}] {s['source']} | {s['district']} | {s['timestamp']}")
    except Exception as e:
        print(f"\nError occurred: {e}")

if __name__ == "__main__":
    test_pipeline()
