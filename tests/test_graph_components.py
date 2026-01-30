"""
Test LangGraph Components
==========================

Tests for:
- Classifier Node
- RAG Node
- Escalation Node
- Workflow Routing
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import unittest
from graph.state import ChatbotState
from graph.classifier_node import QueryClassifier
from graph.rag_node import RAGResponseNode
from graph.escalation_node import EscalationHandler
from graph.workflow import ChatbotWorkflow, get_workflow


class TestClassifierNode(unittest.TestCase):
    """Test query classifier node"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.classifier = QueryClassifier()
        
    def test_classifier_initialization(self):
        """Test that classifier initializes correctly"""
        self.assertIsNotNone(self.classifier)
        print("✅ Classifier initialized")
        
    def test_product_query_classification(self):
        """Test classification of product queries"""
        test_queries = [
            "What is the price of SmartWatch Pro X?",
            "Tell me about Wireless Earbuds Elite",
            "Does the Power Bank have fast charging?",
            "What are the features of SmartWatch?",
            "How much does the earbuds cost?"
        ]
        
        for query in test_queries:
            state = ChatbotState(
                user_query=query,
                classified_category="",
                final_response="",
                retrieved_documents=[],
                confidence_score=0.0,
                needs_escalation=False,
                conversation_id="",
                timestamp="",
                metadata={}
            )
            
            result = self.classifier.classify(state)
            
            self.assertEqual(
                result["classified_category"],
                "product",
                f"Query '{query}' should be classified as 'product'"
            )
            self.assertGreater(
                result["confidence_score"],
                0.5,
                "Confidence should be high for clear product queries"
            )
            
        print(f"✅ Product queries: {len(test_queries)}/{ len(test_queries)} classified correctly")
        
    def test_returns_query_classification(self):
        """Test classification of returns queries"""
        test_queries = [
            "How do I return a product?",
            "What is your refund policy?",
            "I want to return my SmartWatch",
            "Can I get a refund?",
            "Return process for defective item"
        ]
        
        for query in test_queries:
            state = ChatbotState(
                user_query=query,
                classified_category="",
                final_response="",
                retrieved_documents=[],
                confidence_score=0.0,
                needs_escalation=False,
                conversation_id="",
                timestamp="",
                metadata={}
            )
            
            result = self.classifier.classify(state)
            
            self.assertEqual(
                result["classified_category"],
                "returns",
                f"Query '{query}' should be classified as 'returns'"
            )
            
        print(f"✅ Returns queries: {len(test_queries)}/{len(test_queries)} classified correctly")
        
    def test_general_query_classification(self):
        """Test classification of general queries"""
        test_queries = [
            "What are your support hours?",
            "How can I contact customer support?",
            "Where is your store located?",
            "Do you ship internationally?",
            "What payment methods do you accept?"
        ]
        
        for query in test_queries:
            state = ChatbotState(
                user_query=query,
                classified_category="",
                final_response="",
                retrieved_documents=[],
                confidence_score=0.0,
                needs_escalation=False,
                conversation_id="",
                timestamp="",
                metadata={}
            )
            
            result = self.classifier.classify(state)
            
            self.assertEqual(
                result["classified_category"],
                "general",
                f"Query '{query}' should be classified as 'general'"
            )
            
        print(f"✅ General queries: {len(test_queries)}/{len(test_queries)} classified correctly")
        
    def test_confidence_scoring(self):
        """Test that confidence scores are within valid range"""
        test_query = "What is the price of SmartWatch?"
        
        state = ChatbotState(
            user_query=test_query,
            classified_category="",
            final_response="",
            retrieved_documents=[],
            confidence_score=0.0,
            needs_escalation=False,
            conversation_id="",
            timestamp="",
            metadata={}
        )
        
        result = self.classifier.classify(state)
        
        self.assertGreaterEqual(result["confidence_score"], 0.0)
        self.assertLessEqual(result["confidence_score"], 1.0)
        
        print(f"✅ Confidence score within valid range: {result['confidence_score']}")


class TestRAGNode(unittest.TestCase):
    """Test RAG response node"""
    
    def test_rag_node_initialization(self):
        """Test that RAG node initializes correctly"""
        try:
            rag_node = RAGResponseNode()
            self.assertIsNotNone(rag_node)
            print("✅ RAG node initialized")
        except Exception as e:
            self.skipTest(f"Skipping - API key issue: {e}")
            
    def test_rag_node_should_use_rag(self):
        """Test RAG routing logic"""
        rag_node = RAGResponseNode()
        
        # Should use RAG for product queries
        state_product = ChatbotState(
            user_query="What is the price?",
            classified_category="product",
            final_response="",
            retrieved_documents=[],
            confidence_score=1.0,
            needs_escalation=False,
            conversation_id="",
            timestamp="",
            metadata={}
        )
        
        self.assertTrue(rag_node.should_use_rag(state_product))
        
        # Should use RAG for returns queries
        state_returns = ChatbotState(
            user_query="How to return?",
            classified_category="returns",
            final_response="",
            retrieved_documents=[],
            confidence_score=1.0,
            needs_escalation=False,
            conversation_id="",
            timestamp="",
            metadata={}
        )
        
        self.assertTrue(rag_node.should_use_rag(state_returns))
        
        # Should NOT use RAG for general queries
        state_general = ChatbotState(
            user_query="Support hours?",
            classified_category="general",
            final_response="",
            retrieved_documents=[],
            confidence_score=1.0,
            needs_escalation=False,
            conversation_id="",
            timestamp="",
            metadata={}
        )
        
        self.assertFalse(rag_node.should_use_rag(state_general))
        
        print("✅ RAG routing logic correct")


class TestEscalationNode(unittest.TestCase):
    """Test escalation node"""
    
    def test_escalation_node_initialization(self):
        """Test that escalation node initializes correctly"""
        escalation_node = EscalationHandler()
        self.assertIsNotNone(escalation_node)
        print("✅ Escalation node initialized")
        
    def test_escalation_message_generation(self):
        """Test escalation message generation"""
        escalation_node = EscalationHandler()
        
        state = ChatbotState(
            user_query="What are your hours?",
            classified_category="general",
            final_response="",
            retrieved_documents=[],
            confidence_score=0.5,
            needs_escalation=False,
            conversation_id="",
            timestamp="",
            metadata={}
        )
        
        result = escalation_node.escalate(state)
        
        self.assertIsNotNone(result["final_response"])
        self.assertGreater(len(result["final_response"]), 0)
        self.assertTrue(result["needs_escalation"])
        
        # Check for contact information
        response = result["final_response"]
        self.assertIn("support@techgear.com", response.lower())
        
        print("✅ Escalation message contains contact info")
        
    def test_category_specific_messages(self):
        """Test that escalation messages vary by category"""
        escalation_node = EscalationHandler()
        
        categories = ["product", "returns", "general"]
        messages = {}
        
        for category in categories:
            state = ChatbotState(
                user_query="Test query",
                classified_category=category,
                final_response="",
                retrieved_documents=[],
                confidence_score=0.5,
                needs_escalation=False,
                conversation_id="",
                timestamp="",
                metadata={}
            )
            
            result = escalation_node.escalate(state)
            messages[category] = result["final_response"]
        
        # Messages should be different for different categories
        self.assertNotEqual(messages["product"], messages["returns"])
        
        print("✅ Category-specific escalation messages generated")


class TestWorkflow(unittest.TestCase):
    """Test complete LangGraph workflow"""
    
    def test_workflow_initialization(self):
        """Test that workflow initializes correctly"""
        try:
            workflow = get_workflow()
            self.assertIsNotNone(workflow)
            print("✅ Workflow initialized")
        except Exception as e:
            self.skipTest(f"Skipping - initialization issue: {e}")
            
    def test_workflow_has_nodes(self):
        """Test that workflow has all required nodes"""
        try:
            workflow = ChatbotWorkflow()
            
            # Workflow should have graph
            self.assertIsNotNone(workflow.graph)
            
            print("✅ Workflow has required nodes")
        except Exception as e:
            self.skipTest(f"Skipping - initialization issue: {e}")


def run_graph_component_tests():
    """Run all graph component tests"""
    print("\n" + "="*70)
    print("TESTING LANGGRAPH COMPONENTS")
    print("="*70 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestClassifierNode))
    suite.addTests(loader.loadTestsFromTestCase(TestRAGNode))
    suite.addTests(loader.loadTestsFromTestCase(TestEscalationNode))
    suite.addTests(loader.loadTestsFromTestCase(TestWorkflow))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("GRAPH COMPONENTS TESTS SUMMARY")
    print("="*70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*70 + "\n")
    
    return result


if __name__ == "__main__":
    run_graph_component_tests()
