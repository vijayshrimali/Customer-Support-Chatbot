"""
LangGraph Workflow
Complete chatbot workflow with conditional routing
"""

import os
import sys
from typing import Literal
from langgraph.graph import StateGraph, END

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from graph.state import ChatbotState
from graph.classifier_node import classifier_node
from graph.rag_node import rag_response_node
from graph.escalation_node import escalation_node


class ChatbotWorkflow:
    """
    Complete LangGraph workflow for the customer support chatbot
    
    Workflow:
        User Query → Classifier → [product/general → RAG] or [returns → Escalation] → Response
    """
    
    def __init__(self):
        """Initialize the chatbot workflow"""
        print(f"\n{'='*70}")
        print(f"🚀 Initializing Chatbot Workflow")
        print(f"{'='*70}")
        
        # Create the state graph
        self.workflow = StateGraph(ChatbotState)
        
        # Add nodes to the graph
        print(f"\n📊 Adding nodes to workflow:")
        self._add_nodes()
        
        # Set entry point
        print(f"\n🎯 Setting entry point: classifier")
        self.workflow.set_entry_point("classifier")
        
        # Add conditional routing
        print(f"\n🔀 Adding conditional routing:")
        self._add_conditional_edges()
        
        # Add edges to END
        print(f"\n🏁 Adding terminal edges:")
        self._add_terminal_edges()
        
        # Compile the graph
        print(f"\n⚙️  Compiling workflow graph...")
        self.graph = self.workflow.compile()
        
        print(f"\n{'='*70}")
        print(f"✅ Chatbot Workflow Initialized Successfully!")
        print(f"{'='*70}")
        
        # Display workflow structure
        self._display_workflow_structure()
    
    def _add_nodes(self):
        """Add all nodes to the workflow"""
        # Classifier node
        self.workflow.add_node("classifier", classifier_node)
        print(f"   ✅ Added: classifier (query classification)")
        
        # RAG response node
        self.workflow.add_node("rag", rag_response_node)
        print(f"   ✅ Added: rag (knowledge-based responses)")
        
        # Escalation node
        self.workflow.add_node("escalation", escalation_node)
        print(f"   ✅ Added: escalation (human support handoff)")
    
    def _route_query(self, state: ChatbotState) -> Literal["rag", "escalation"]:
        """
        Route queries based on classified category
        
        Args:
            state: Current chatbot state
            
        Returns:
            Next node name: "rag" or "escalation"
        """
        category = state.get("classified_category", "general")
        
        # Routing logic:
        # - product → RAG (product information from knowledge base)
        # - general → RAG (general info from knowledge base)
        # - returns → escalation (needs human support)
        
        if category in ["product", "general"]:
            print(f"   ➡️  Routing '{category}' → RAG node")
            return "rag"
        else:  # returns or other categories
            print(f"   ➡️  Routing '{category}' → Escalation node")
            return "escalation"
    
    def _add_conditional_edges(self):
        """Add conditional routing edges"""
        self.workflow.add_conditional_edges(
            "classifier",  # From classifier node
            self._route_query,  # Routing function
            {
                "rag": "rag",  # If route returns "rag", go to rag node
                "escalation": "escalation"  # If route returns "escalation", go to escalation node
            }
        )
        print(f"   ✅ Conditional routing from classifier:")
        print(f"      • product/general → rag")
        print(f"      • returns → escalation")
    
    def _add_terminal_edges(self):
        """Add edges to END node"""
        self.workflow.add_edge("rag", END)
        print(f"   ✅ rag → END")
        
        self.workflow.add_edge("escalation", END)
        print(f"   ✅ escalation → END")
    
    def _display_workflow_structure(self):
        """Display the workflow structure"""
        print(f"\n📊 Workflow Structure:")
        print(f"{'='*70}")
        print(f"""
        User Query
            ↓
        ┌─────────────┐
        │ Classifier  │ (Categorize query)
        └─────────────┘
            ↓
            ├─→ [product/general] ─→ ┌─────────┐
            │                         │   RAG   │ (Knowledge base)
            │                         └─────────┘
            │                              ↓
            │                            [END]
            │
            └─→ [returns] ──────────→ ┌─────────────┐
                                      │ Escalation  │ (Human support)
                                      └─────────────┘
                                           ↓
                                         [END]
        """)
        print(f"{'='*70}")
    
    def run(self, user_query: str, verbose: bool = True) -> ChatbotState:
        """
        Run the workflow with a user query
        
        Args:
            user_query: User's question
            verbose: Print execution details
            
        Returns:
            Final chatbot state with response
        """
        from graph.state import create_initial_state
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"🚀 RUNNING WORKFLOW")
            print(f"{'='*70}")
            print(f"Query: {user_query}")
        
        # Create initial state
        initial_state = create_initial_state(user_query)
        
        # Run the workflow
        final_state = self.graph.invoke(initial_state)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"✅ WORKFLOW COMPLETE")
            print(f"{'='*70}")
            print(f"Category: {final_state.get('classified_category')}")
            print(f"Confidence: {final_state.get('confidence_score', 0):.2f}")
            print(f"Needs Escalation: {final_state.get('needs_escalation', False)}")
            print(f"\nResponse:")
            print(f"{'-'*70}")
            print(final_state.get('final_response', 'No response generated'))
            print(f"{'-'*70}")
        
        return final_state
    
    def get_graph(self):
        """
        Get the compiled graph object
        
        Returns:
            Compiled LangGraph workflow
        """
        return self.graph


# Global workflow instance (singleton)
_workflow_instance = None


def get_workflow() -> ChatbotWorkflow:
    """
    Get or create workflow instance (singleton)
    
    Returns:
        ChatbotWorkflow instance
    """
    global _workflow_instance
    if _workflow_instance is None:
        _workflow_instance = ChatbotWorkflow()
    return _workflow_instance


def run_chatbot(user_query: str, verbose: bool = True) -> ChatbotState:
    """
    Convenience function to run the chatbot workflow
    
    Args:
        user_query: User's question
        verbose: Print execution details
        
    Returns:
        Final chatbot state with response
    """
    workflow = get_workflow()
    return workflow.run(user_query, verbose=verbose)


# =============================================================================
# TESTING FUNCTIONS
# =============================================================================

def test_workflow_product_query():
    """Test workflow with product queries"""
    print(f"\n{'='*70}")
    print(f"🧪 TEST 1: Product Queries → RAG Node")
    print(f"{'='*70}")
    
    workflow = ChatbotWorkflow()
    
    test_queries = [
        "What is the price of SmartWatch Pro X?",
        "Tell me about Wireless Earbuds features",
        "Does the power bank support fast charging?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'-'*70}")
        print(f"Product Query {i}/{len(test_queries)}")
        print(f"{'-'*70}")
        
        result = workflow.run(query, verbose=False)
        
        print(f"Query: {query}")
        print(f"Category: {result['classified_category']}")
        print(f"Route: product → RAG")
        print(f"Response: {result['final_response'][:100]}...")
        
        # Verify product queries go to RAG
        assert result['final_response'] is not None
        assert result['classified_category'] in ['product', 'general']
        assert result.get('needs_escalation', False) == False
        
        print(f"✅ Test passed!")
    
    print(f"\n{'='*70}")
    print(f"✅ All product query tests passed!")
    print(f"{'='*70}")


def test_workflow_returns_query():
    """Test workflow with returns queries"""
    print(f"\n{'='*70}")
    print(f"🧪 TEST 2: Returns Queries → Escalation Node")
    print(f"{'='*70}")
    
    workflow = ChatbotWorkflow()
    
    test_queries = [
        "I want to return my smartwatch",
        "How do I get a refund?",
        "Can I exchange my defective earbuds?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'-'*70}")
        print(f"Returns Query {i}/{len(test_queries)}")
        print(f"{'-'*70}")
        
        result = workflow.run(query, verbose=False)
        
        print(f"Query: {query}")
        print(f"Category: {result['classified_category']}")
        print(f"Route: returns → Escalation")
        print(f"Response: {result['final_response'][:100]}...")
        
        # Verify returns queries go to escalation
        assert result['final_response'] is not None
        assert result['classified_category'] == 'returns'
        assert result.get('needs_escalation', False) == True
        assert 'support@techgear.com' in result['final_response']
        
        print(f"✅ Test passed!")
    
    print(f"\n{'='*70}")
    print(f"✅ All returns query tests passed!")
    print(f"{'='*70}")


def test_workflow_general_query():
    """Test workflow with general queries"""
    print(f"\n{'='*70}")
    print(f"🧪 TEST 3: General Queries → RAG Node")
    print(f"{'='*70}")
    
    workflow = ChatbotWorkflow()
    
    test_queries = [
        "What are your customer support hours?",
        "Do you accept cash on delivery?",
        "What payment methods do you accept?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'-'*70}")
        print(f"General Query {i}/{len(test_queries)}")
        print(f"{'-'*70}")
        
        result = workflow.run(query, verbose=False)
        
        print(f"Query: {query}")
        print(f"Category: {result['classified_category']}")
        print(f"Route: general → RAG")
        print(f"Response: {result['final_response'][:100]}...")
        
        # Verify general queries go to RAG
        assert result['final_response'] is not None
        assert result['classified_category'] == 'general'
        
        print(f"✅ Test passed!")
    
    print(f"\n{'='*70}")
    print(f"✅ All general query tests passed!")
    print(f"{'='*70}")


def test_complete_workflow():
    """Test complete workflow with diverse queries"""
    print(f"\n{'='*70}")
    print(f"🧪 TEST 4: Complete Workflow - Mixed Queries")
    print(f"{'='*70}")
    
    workflow = ChatbotWorkflow()
    
    # Mixed test cases
    test_cases = [
        {
            "query": "What is the warranty on SmartWatch Pro X?",
            "expected_category": "product",
            "expected_route": "rag"
        },
        {
            "query": "I want to return my order",
            "expected_category": "returns",
            "expected_route": "escalation"
        },
        {
            "query": "What are your shipping charges?",
            "expected_category": "general",
            "expected_route": "rag"
        },
        {
            "query": "My product is defective, need replacement",
            "expected_category": "returns",
            "expected_route": "escalation"
        }
    ]
    
    success_count = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'-'*70}")
        print(f"Mixed Query {i}/{len(test_cases)}")
        print(f"{'-'*70}")
        
        result = workflow.run(test['query'], verbose=False)
        
        print(f"Query: {test['query']}")
        print(f"Expected Category: {test['expected_category']}")
        print(f"Actual Category: {result['classified_category']}")
        print(f"Expected Route: {test['expected_route']}")
        
        # Check if routing is correct
        if test['expected_route'] == 'rag':
            assert result.get('needs_escalation', False) == False
            print(f"Actual Route: rag ✅")
        else:
            assert result.get('needs_escalation', False) == True
            print(f"Actual Route: escalation ✅")
        
        assert result['final_response'] is not None
        print(f"Response: {result['final_response'][:80]}...")
        
        success_count += 1
        print(f"✅ Test {i} passed!")
    
    print(f"\n{'='*70}")
    print(f"📊 COMPLETE WORKFLOW TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Total Tests: {len(test_cases)}")
    print(f"Passed: {success_count}")
    print(f"Success Rate: {(success_count/len(test_cases))*100:.1f}%")
    print(f"\n✅ Complete workflow functioning correctly!")


def test_workflow_routing_accuracy():
    """Test routing accuracy with edge cases"""
    print(f"\n{'='*70}")
    print(f"🧪 TEST 5: Routing Accuracy - Edge Cases")
    print(f"{'='*70}")
    
    workflow = ChatbotWorkflow()
    
    # Edge case queries
    test_cases = [
        ("price of smartwatch?", "product", "rag"),
        ("return policy?", "product", "rag"),  # Policy inquiry → RAG
        ("I want refund", "returns", "escalation"),
        ("support hours", "general", "rag"),
        ("cancel my order", "returns", "escalation"),
        ("warranty information", "product", "rag"),
    ]
    
    correct_routes = 0
    
    for query, expected_category, expected_route in test_cases:
        result = workflow.run(query, verbose=False)
        
        actual_route = "escalation" if result.get('needs_escalation', False) else "rag"
        
        print(f"\nQuery: '{query}'")
        print(f"  Category: {result['classified_category']}")
        print(f"  Expected Route: {expected_route}")
        print(f"  Actual Route: {actual_route}")
        
        if actual_route == expected_route:
            print(f"  ✅ Correct routing!")
            correct_routes += 1
        else:
            print(f"  ⚠️  Routing mismatch")
    
    accuracy = (correct_routes / len(test_cases)) * 100
    
    print(f"\n{'='*70}")
    print(f"📊 ROUTING ACCURACY")
    print(f"{'='*70}")
    print(f"Total Queries: {len(test_cases)}")
    print(f"Correct Routes: {correct_routes}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    assert accuracy >= 80, f"Routing accuracy too low: {accuracy}%"
    print(f"\n✅ Routing accuracy test passed!")


if __name__ == "__main__":
    """Run all workflow tests"""
    print(f"\n{'#'*70}")
    print(f"#  LANGGRAPH WORKFLOW - COMPREHENSIVE TESTING")
    print(f"{'#'*70}")
    
    try:
        # Test 1: Product queries
        test_workflow_product_query()
        
        # Test 2: Returns queries
        test_workflow_returns_query()
        
        # Test 3: General queries
        test_workflow_general_query()
        
        # Test 4: Complete workflow
        test_complete_workflow()
        
        # Test 5: Routing accuracy
        test_workflow_routing_accuracy()
        
        print(f"\n{'='*70}")
        print(f"✅ ALL WORKFLOW TESTS PASSED!")
        print(f"{'='*70}")
        print(f"\n🎯 Workflow is production-ready!")
        print(f"   • Conditional routing working correctly")
        print(f"   • All nodes integrated successfully")
        print(f"   • Edge cases handled properly")
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ WORKFLOW TEST FAILED")
        print(f"{'='*70}")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
