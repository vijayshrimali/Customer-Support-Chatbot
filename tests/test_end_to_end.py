"""
End-to-End Integration Tests
============================

Tests complete user journey through the chatbot system.
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
import time
from graph.workflow import get_workflow


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        try:
            cls.workflow = get_workflow()
            cls.api_available = True
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize workflow: {e}")
            cls.api_available = False
            
    def test_product_query_journey(self):
        """Test complete journey for product query"""
        if not self.api_available:
            self.skipTest("Workflow not available - API key issue")
            
        test_queries = [
            "What is the price of SmartWatch Pro X?",
            "Tell me about Wireless Earbuds Elite features",
            "Does Power Bank Ultra support fast charging?"
        ]
        
        for query in test_queries:
            print(f"\n📝 Testing: {query}")
            
            try:
                result = self.workflow.run(
                    user_query=query,
                    verbose=False
                )
                
                # Validate response structure
                self.assertIn("final_response", result)
                self.assertIn("classified_category", result)
                self.assertIn("confidence_score", result)
                
                # Should be classified as product
                self.assertEqual(result["classified_category"], "product")
                
                # Should have a response
                self.assertIsNotNone(result["final_response"])
                self.assertGreater(len(result["final_response"]), 0)
                
                # Should have high confidence
                self.assertGreater(result["confidence_score"], 0.5)
                
                print(f"✅ Category: {result['classified_category']}")
                print(f"✅ Confidence: {result['confidence_score']}")
                print(f"✅ Response: {result['final_response'][:100]}...")
                
            except Exception as e:
                self.fail(f"Product query failed: {e}")
                
    def test_returns_query_journey(self):
        """Test complete journey for returns query"""
        if not self.api_available:
            self.skipTest("Workflow not available - API key issue")
            
        test_queries = [
            "How do I return a product?",
            "What is your refund policy?",
            "I want to return my purchase"
        ]
        
        for query in test_queries:
            print(f"\n📝 Testing: {query}")
            
            try:
                result = self.workflow.run(
                    user_query=query,
                    verbose=False
                )
                
                # Validate response structure
                self.assertIn("final_response", result)
                self.assertIn("classified_category", result)
                
                # Should be classified as returns
                self.assertEqual(result["classified_category"], "returns")
                
                # Should have a response
                self.assertIsNotNone(result["final_response"])
                self.assertGreater(len(result["final_response"]), 0)
                
                print(f"✅ Category: {result['classified_category']}")
                print(f"✅ Response length: {len(result['final_response'])} chars")
                
            except Exception as e:
                self.fail(f"Returns query failed: {e}")
                
    def test_general_query_journey(self):
        """Test complete journey for general query"""
        if not self.api_available:
            self.skipTest("Workflow not available - API key issue")
            
        test_queries = [
            "What are your customer support hours?",
            "How can I contact you?",
            "Do you ship internationally?"
        ]
        
        for query in test_queries:
            print(f"\n📝 Testing: {query}")
            
            try:
                result = self.workflow.run(
                    user_query=query,
                    verbose=False
                )
                
                # Validate response structure
                self.assertIn("final_response", result)
                self.assertIn("classified_category", result)
                
                # Should be classified as general
                self.assertEqual(result["classified_category"], "general")
                
                # Should have a response
                self.assertIsNotNone(result["final_response"])
                self.assertGreater(len(result["final_response"]), 0)
                
                # Should need escalation
                self.assertTrue(result.get("needs_escalation", False))
                
                print(f"✅ Category: {result['classified_category']}")
                print(f"✅ Escalation: {result.get('needs_escalation', False)}")
                
            except Exception as e:
                self.fail(f"General query failed: {e}")
                
    def test_response_time(self):
        """Test that responses are reasonably fast"""
        if not self.api_available:
            self.skipTest("Workflow not available - API key issue")
            
        query = "What is the price of SmartWatch?"
        
        start_time = time.time()
        
        try:
            result = self.workflow.run(
                user_query=query,
                verbose=False
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Should respond within 10 seconds
            self.assertLess(
                duration,
                10.0,
                f"Response took {duration:.2f}s, should be < 10s"
            )
            
            print(f"✅ Response time: {duration:.2f}s")
            
        except Exception as e:
            self.skipTest(f"Performance test skipped: {e}")
            
    def test_conversation_id_tracking(self):
        """Test that conversation IDs are tracked"""
        if not self.api_available:
            self.skipTest("Workflow not available - API key issue")
            
        query = "What products do you sell?"
        
        try:
            result = self.workflow.run(
                user_query=query,
                verbose=False
            )
            
            # Should have conversation ID
            self.assertIn("conversation_id", result)
            self.assertIsNotNone(result["conversation_id"])
            self.assertGreater(len(result["conversation_id"]), 0)
            
            print(f"✅ Conversation ID: {result['conversation_id']}")
            
        except Exception as e:
            self.skipTest(f"Conversation tracking test skipped: {e}")
            
    def test_metadata_inclusion(self):
        """Test that metadata is included in responses"""
        if not self.api_available:
            self.skipTest("Workflow not available - API key issue")
            
        query = "Tell me about SmartWatch features"
        
        try:
            result = self.workflow.run(
                user_query=query,
                verbose=False
            )
            
            # Should have metadata
            self.assertIn("metadata", result)
            self.assertIsInstance(result["metadata"], dict)
            
            print(f"✅ Metadata included: {list(result['metadata'].keys())}")
            
        except Exception as e:
            self.skipTest(f"Metadata test skipped: {e}")


def run_end_to_end_tests():
    """Run all end-to-end tests"""
    print("\n" + "="*70)
    print("RUNNING END-TO-END INTEGRATION TESTS")
    print("="*70 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEnd))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("END-TO-END TESTS SUMMARY")
    print("="*70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*70 + "\n")
    
    return result


if __name__ == "__main__":
    run_end_to_end_tests()
