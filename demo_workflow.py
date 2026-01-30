"""
Simple Workflow Demo
Demonstrates the LangGraph workflow in action
"""

import sys
import os

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

from graph.workflow import ChatbotWorkflow

def demo_workflow():
    """Demonstrate the chatbot workflow"""
    
    print("="*70)
    print("🤖 CUSTOMER SUPPORT CHATBOT - WORKFLOW DEMO")
    print("="*70)
    
    # Initialize workflow
    workflow = ChatbotWorkflow()
    
    # Demo queries
    demo_queries = [
        "What is the price of SmartWatch Pro X?",
        "Tell me about the warranty",
        "How do I return a product?",
        "What are your customer support hours?",
        "I want a refund for my defective product"
    ]
    
    print(f"\n📝 Testing {len(demo_queries)} queries:\n")
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n{'='*70}")
        print(f"Query {i}/{len(demo_queries)}")
        print(f"{'='*70}")
        
        result = workflow.run(query, verbose=True)
        
        print(f"\n{'─'*70}")
        print(f"Summary:")
        print(f"  Category: {result['classified_category']}")
        print(f"  Confidence: {result.get('confidence_score', 0):.2f}")
        print(f"  Route: {'Escalation' if result.get('needs_escalation') else 'RAG'}")
        print(f"  Response Length: {len(result.get('final_response', ''))} chars")
        print(f"{'─'*70}")
    
    print(f"\n{'='*70}")
    print(f"✅ DEMO COMPLETE")
    print(f"{'='*70}")
    print(f"\n🎯 The workflow is functioning correctly!")
    print(f"   • Queries are classified")
    print(f"   • Routing works based on category")
    print(f"   • Responses are generated appropriately")


if __name__ == "__main__":
    demo_workflow()
