"""
Test RAG Chain's strict adherence to retrieved context
"""

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
services_dir = os.path.join(parent_dir, 'services')
sys.path.insert(0, services_dir)

from rag_chain import RAGChain

print("\n" + "="*70)
print("  RAG CHAIN CONTEXT ADHERENCE TESTING")
print("="*70)

# Initialize RAG chain
rag = RAGChain()
rag.build_chain()

# Test cases
test_cases = [
    {
        "query": "What is the price of SmartWatch Pro X?",
        "category": "In Knowledge Base",
        "expected": "Should return ₹15,999"
    },
    {
        "query": "What is the price of iPhone 15?",
        "category": "Out of Knowledge Base",
        "expected": "Should say 'I don't have that information'"
    },
    {
        "query": "Tell me about Samsung Galaxy earbuds",
        "category": "Out of Knowledge Base",
        "expected": "Should say 'I don't have that information'"
    },
    {
        "query": "What is the warranty period for SmartWatch?",
        "category": "In Knowledge Base",
        "expected": "Should return 1 year warranty information"
    },
    {
        "query": "Do you sell laptops?",
        "category": "Out of Knowledge Base",
        "expected": "Should say 'I don't have that information'"
    }
]

print(f"\n🧪 Testing {len(test_cases)} context adherence scenarios")
print("="*70)

results = []
for i, test in enumerate(test_cases, 1):
    print(f"\n{'─'*70}")
    print(f"Test Case {i}: {test['category']}")
    print(f"{'─'*70}")
    print(f"Query: {test['query']}")
    print(f"Expected: {test['expected']}")
    
    # Get response
    response = rag.query(test['query'], verbose=False)
    
    print(f"\n📝 Response:")
    print(f"   {response}")
    
    # Check if response adheres to context
    if test['category'] == "Out of Knowledge Base":
        adheres = "don't have" in response.lower() or "not found" in response.lower() or "no information" in response.lower()
    else:
        adheres = "don't have" not in response.lower()
    
    status = "✅ PASS" if adheres else "❌ FAIL"
    print(f"\n{status} - Context adherence: {'Yes' if adheres else 'No'}")
    
    results.append({
        "query": test['query'],
        "category": test['category'],
        "adheres": adheres
    })

# Summary
print(f"\n{'='*70}")
print(f"  TEST SUMMARY")
print(f"{'='*70}")

passed = sum(1 for r in results if r['adheres'])
total = len(results)

print(f"\n✅ Passed: {passed}/{total} ({passed/total*100:.1f}%)")
print(f"❌ Failed: {total-passed}/{total}")

print(f"\n📊 Results by Category:")
for result in results:
    status = "✅ PASS" if result['adheres'] else "❌ FAIL"
    print(f"   {status} - {result['category']}")

if passed == total:
    print(f"\n🎉 ALL TESTS PASSED! RAG chain strictly adheres to context.")
    print(f"✅ Only answers from knowledge base")
    print(f"✅ Correctly refuses out-of-scope questions")
else:
    print(f"\n⚠️  Some tests failed. Review context adherence.")

print(f"\n{'='*70}")
print(f"  ✅ CONTEXT ADHERENCE TESTING COMPLETE")
print(f"{'='*70}")

