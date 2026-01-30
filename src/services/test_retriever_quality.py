"""
Test retriever quality and relevance
"""

from retriever_service import RetrieverService

print("\n" + "="*70)
print("  RETRIEVER QUALITY TESTING")
print("="*70)

# Initialize retriever service
service = RetrieverService(k=3)
retriever = service.load_retriever()

# Test queries with expected content
test_cases = [
    {
        "query": "What is the price of SmartWatch?",
        "expected_keywords": ["SmartWatch", "₹15,999", "price"],
        "category": "Product Pricing"
    },
    {
        "query": "How long does the earbuds battery last?",
        "expected_keywords": ["24-hour", "battery", "earbuds"],
        "category": "Product Features"
    },
    {
        "query": "Can I return a product?",
        "expected_keywords": ["return", "7-day", "policy"],
        "category": "Return Policy"
    },
    {
        "query": "What is the phone number for support?",
        "expected_keywords": ["1800-123-4567", "phone", "support"],
        "category": "Customer Support"
    },
    {
        "query": "Does SmartWatch track heart rate?",
        "expected_keywords": ["heart rate", "monitoring", "SmartWatch"],
        "category": "Product Features"
    }
]

print(f"\n🧪 Testing {len(test_cases)} query scenarios")
print("="*70)

results = []
for i, test in enumerate(test_cases, 1):
    print(f"\n{'─'*70}")
    print(f"Test Case {i}: {test['category']}")
    print(f"{'─'*70}")
    print(f"Query: {test['query']}")
    
    # Retrieve documents
    docs = service.retrieve(test['query'])
    
    # Check relevance
    print(f"\n📊 Relevance Check:")
    found_keywords = []
    for keyword in test['expected_keywords']:
        found = any(keyword.lower() in doc.page_content.lower() for doc in docs)
        status = "✅" if found else "❌"
        print(f"   {status} Keyword '{keyword}': {'Found' if found else 'Not found'}")
        if found:
            found_keywords.append(keyword)
    
    relevance_score = len(found_keywords) / len(test['expected_keywords']) * 100
    print(f"\n   Relevance Score: {relevance_score:.1f}%")
    
    results.append({
        "query": test['query'],
        "category": test['category'],
        "relevance": relevance_score,
        "passed": relevance_score >= 66.7  # At least 2/3 keywords
    })

# Summary
print(f"\n{'='*70}")
print(f"  TEST SUMMARY")
print(f"{'='*70}")

passed = sum(1 for r in results if r['passed'])
total = len(results)

print(f"\n✅ Passed: {passed}/{total} ({passed/total*100:.1f}%)")
print(f"❌ Failed: {total-passed}/{total}")

print(f"\n📊 Results by Category:")
for result in results:
    status = "✅ PASS" if result['passed'] else "❌ FAIL"
    print(f"   {status} - {result['category']}: {result['relevance']:.1f}%")

avg_relevance = sum(r['relevance'] for r in results) / len(results)
print(f"\n�� Average Relevance Score: {avg_relevance:.1f}%")

if passed == total:
    print(f"\n🎉 ALL TESTS PASSED! Retriever quality is excellent.")
else:
    print(f"\n⚠️  Some tests failed. Review retriever configuration.")

print(f"\n{'='*70}")
print(f"  ✅ QUALITY TESTING COMPLETE")
print(f"{'='*70}")

