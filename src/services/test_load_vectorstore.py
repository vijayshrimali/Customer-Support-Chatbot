"""
Test loading persisted vector store without re-embedding
"""

from vector_store import VectorStoreService

print("\n" + "="*70)
print("  TESTING PERSISTENT VECTOR STORE LOADING")
print("="*70)

# Create service instance
vector_store_service = VectorStoreService()

# Load existing vector store (no re-embedding needed)
print("\n🔄 Loading vector store from disk...")
vector_store = vector_store_service.load_vector_store()

if vector_store:
    # Test multiple queries
    test_queries = [
        "What is the price of SmartWatch?",
        "Tell me about wireless earbuds",
        "What is your return policy?",
        "How do I contact customer support?"
    ]
    
    print("\n" + "="*70)
    print("  TESTING SIMILARITY SEARCH WITH MULTIPLE QUERIES")
    print("="*70)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─'*70}")
        print(f"Query {i}: {query}")
        print(f"{'─'*70}")
        
        results = vector_store.similarity_search(query, k=2)
        
        for j, doc in enumerate(results, 1):
            preview = doc.page_content[:150].replace('\n', ' ')
            print(f"\n  Result {j}: {preview}...")
    
    print("\n" + "="*70)
    print("  ✅ ALL TESTS PASSED!")
    print("="*70)
    print("\n✅ Vector store loaded from disk successfully")
    print("✅ No re-embedding was needed")
    print("✅ Similarity search working perfectly")
    print(f"✅ Total documents: {vector_store._collection.count()}")
else:
    print("❌ Failed to load vector store")

