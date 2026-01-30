"""
RAG Response Node
LangGraph node that uses the RAG chain to generate answers for product and return-related queries
"""

import os
import sys
from typing import Dict, Any

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
bot_dir = os.path.join(parent_dir, 'bot')
sys.path.insert(0, bot_dir)
sys.path.insert(0, parent_dir)

from bot.rag_chain import RAGChain
from graph.state import ChatbotState, update_state


class RAGResponseNode:
    """
    RAG Response Generator
    Uses RAG chain to generate context-aware responses for user queries
    """
    
    def __init__(self):
        """Initialize RAG Response Node"""
        self.rag_chain = None
        self.initialize_rag_chain()
    
    def initialize_rag_chain(self):
        """
        Initialize and build the RAG chain
        """
        print(f"\n{'='*70}")
        print(f"🤖 Initializing RAG Response Node")
        print(f"{'='*70}")
        
        try:
            # Create RAG chain instance
            self.rag_chain = RAGChain(
                temperature=0.3,  # Low temperature for factual responses
                top_k=3           # Retrieve top 3 relevant documents
            )
            
            # Initialize LLM
            self.rag_chain.initialize_llm()
            
            # Initialize retriever
            self.rag_chain.initialize_retriever()
            
            # Build the RAG chain
            self.rag_chain.build_chain()
            
            print(f"\n✅ RAG Response Node initialized successfully!")
            print(f"   Ready to generate context-aware responses")
            
        except Exception as e:
            print(f"\n❌ Error initializing RAG Response Node: {str(e)}")
            raise
    
    def should_use_rag(self, category: str) -> bool:
        """
        Determine if RAG should be used for this category
        
        Args:
            category: Classified category from classifier node
            
        Returns:
            bool: True if RAG should be used, False otherwise
        """
        # Use RAG for product and general queries
        # Returns/issues will be handled by escalation node
        rag_categories = ['product', 'general', 'product_inquiry', 'general_inquiry']
        return category in rag_categories if category else False
    
    def generate_response(self, query: str) -> Dict[str, Any]:
        """
        Generate response using RAG chain
        
        Args:
            query: User query string
            
        Returns:
            Dict containing response and retrieved documents
        """
        try:
            print(f"\n{'='*70}")
            print(f"🔍 Generating RAG Response")
            print(f"{'='*70}")
            print(f"Query: {query}")
            
            # Retrieve documents first (before generating response)
            retrieved_docs = self.rag_chain.retriever.invoke(query)
            
            print(f"\n✅ Retrieved {len(retrieved_docs)} documents")
            
            # Query the RAG chain (returns string response)
            response = self.rag_chain.query(query, verbose=False)
            
            print(f"\n✅ Response generated successfully!")
            print(f"   Retrieved {len(retrieved_docs)} documents")
            print(f"   Response length: {len(response)} characters")
            
            # Format retrieved documents for state
            formatted_docs = []
            for i, doc in enumerate(retrieved_docs, 1):
                formatted_docs.append({
                    'rank': i,
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'source': doc.metadata.get('source', 'unknown')
                })
            
            return {
                'response': response,
                'retrieved_documents': formatted_docs,
                'document_count': len(formatted_docs)
            }
            
        except Exception as e:
            print(f"\n❌ Error generating response: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'response': f"I apologize, but I encountered an error processing your request. Please try again or contact our support team.",
                'retrieved_documents': [],
                'document_count': 0,
                'error': str(e)
            }
    
    def process(self, state: ChatbotState) -> ChatbotState:
        """
        Process the state and generate RAG response
        
        Args:
            state: Current chatbot state
            
        Returns:
            Updated chatbot state with response
        """
        print(f"\n{'='*70}")
        print(f"📊 RAG Node Processing")
        print(f"{'='*70}")
        
        # Extract query and category
        query = state.get('user_query', '')
        category = state.get('classified_category', '')
        confidence = state.get('confidence_score', 0.0)
        
        print(f"User Query: {query}")
        print(f"Category: {category}")
        print(f"Confidence: {confidence:.2f}")
        
        # Check if RAG should be used
        if not self.should_use_rag(category):
            print(f"\n⚠️  RAG not applicable for category: {category}")
            print(f"   This query should be handled by another node")
            
            # Return state unchanged - let other nodes handle it
            return state
        
        # Generate response using RAG
        print(f"\n🚀 Using RAG chain to generate response...")
        result = self.generate_response(query)
        
        # Update state with response and documents
        updated_state = update_state(
            state,
            final_response=result['response'],
            retrieved_documents=result['retrieved_documents'],
            metadata={
                **(state.get('metadata', {}) or {}),
                'rag_used': True,
                'document_count': result['document_count'],
                'response_source': 'rag_chain',
                'error': result.get('error', None)
            }
        )
        
        print(f"\n✅ State updated with RAG response")
        print(f"   Documents retrieved: {result['document_count']}")
        print(f"   Response preview: {result['response'][:100]}...")
        
        return updated_state


# Global RAG node instance (singleton pattern)
_rag_node_instance = None


def get_rag_node() -> RAGResponseNode:
    """
    Get or create RAG node instance (singleton)
    
    Returns:
        RAGResponseNode instance
    """
    global _rag_node_instance
    if _rag_node_instance is None:
        _rag_node_instance = RAGResponseNode()
    return _rag_node_instance


def rag_response_node(state: ChatbotState) -> ChatbotState:
    """
    LangGraph node function for RAG response generation
    
    This is the main entry point for the RAG node in the LangGraph workflow.
    
    Args:
        state: Current chatbot state from LangGraph
        
    Returns:
        Updated chatbot state with RAG response
        
    Example:
        >>> from graph.state import create_initial_state
        >>> from graph.classifier_node import classifier_node
        >>> 
        >>> # Create and classify state
        >>> state = create_initial_state("What is the price of SmartWatch Pro X?")
        >>> state = classifier_node(state)
        >>> 
        >>> # Generate RAG response
        >>> state = rag_response_node(state)
        >>> print(state['final_response'])
    """
    # Get singleton RAG node instance
    rag_node = get_rag_node()
    
    # Process state and generate response
    return rag_node.process(state)


# =============================================================================
# TESTING FUNCTIONS
# =============================================================================

def test_rag_node_basic():
    """Test RAG node with basic queries"""
    print(f"\n{'='*70}")
    print(f"🧪 TESTING RAG NODE - Basic Queries")
    print(f"{'='*70}")
    
    from graph.state import create_initial_state
    from graph.classifier_node import classifier_node
    
    # Test queries
    test_queries = [
        "What is the price of SmartWatch Pro X?",
        "Tell me about Wireless Earbuds features",
        "What are your customer support hours?",
        "Is the power bank waterproof?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'-'*70}")
        print(f"Test {i}/{len(test_queries)}: {query}")
        print(f"{'-'*70}")
        
        # Create initial state
        state = create_initial_state(query)
        
        # Classify query
        state = classifier_node(state)
        
        # Generate RAG response
        state = rag_response_node(state)
        
        # Display results
        print(f"\n📊 Results:")
        print(f"   Category: {state['classified_category']}")
        print(f"   Confidence: {state.get('confidence_score', 0):.2f}")
        print(f"   Documents: {len(state.get('retrieved_documents', []))}")
        print(f"   Response: {state.get('final_response', 'No response')[:200]}...")


def test_rag_node_categories():
    """Test RAG node with different categories"""
    print(f"\n{'='*70}")
    print(f"🧪 TESTING RAG NODE - Category Handling")
    print(f"{'='*70}")
    
    from graph.state import create_initial_state
    
    # Test different categories
    test_cases = [
        ("What is the price of SmartWatch?", "product"),
        ("How do I return a product?", "returns"),
        ("What are your support hours?", "general"),
    ]
    
    for i, (query, expected_category) in enumerate(test_cases, 1):
        print(f"\n{'-'*70}")
        print(f"Test {i}/{len(test_cases)}")
        print(f"{'-'*70}")
        print(f"Query: {query}")
        print(f"Expected Category: {expected_category}")
        
        # Create state with pre-classified category
        state = create_initial_state(query)
        state['classified_category'] = expected_category
        state['confidence_score'] = 1.0
        
        # Generate RAG response
        state = rag_response_node(state)
        
        # Check if RAG was used
        rag_used = state.get('metadata', {}).get('rag_used', False)
        
        print(f"\n📊 Results:")
        print(f"   RAG Used: {'✅ Yes' if rag_used else '❌ No'}")
        print(f"   Category: {state['classified_category']}")
        
        if rag_used:
            print(f"   Documents: {len(state.get('retrieved_documents', []))}")
            print(f"   Response: {state.get('final_response', '')[:150]}...")
        else:
            print(f"   Note: This category should be handled by another node")


def test_full_pipeline():
    """Test complete pipeline: classification + RAG response"""
    print(f"\n{'='*70}")
    print(f"🧪 TESTING FULL PIPELINE - Classification → RAG")
    print(f"{'='*70}")
    
    from graph.state import create_initial_state
    from graph.classifier_node import classifier_node
    
    # Product queries that should use RAG
    product_queries = [
        "What is the warranty period for SmartWatch Pro X?",
        "Does the power bank support fast charging?",
        "How long does the battery last on Wireless Earbuds?",
    ]
    
    success_count = 0
    total_queries = len(product_queries)
    
    for i, query in enumerate(product_queries, 1):
        print(f"\n{'-'*70}")
        print(f"Pipeline Test {i}/{total_queries}")
        print(f"{'-'*70}")
        print(f"Query: {query}")
        
        try:
            # Step 1: Create initial state
            state = create_initial_state(query)
            print(f"\n✅ Step 1: Initial state created")
            
            # Step 2: Classify query
            state = classifier_node(state)
            print(f"✅ Step 2: Query classified as '{state['classified_category']}'")
            
            # Step 3: Generate RAG response
            state = rag_response_node(state)
            print(f"✅ Step 3: RAG response generated")
            
            # Verify response
            has_response = bool(state.get('final_response'))
            has_docs = len(state.get('retrieved_documents', [])) > 0
            
            if has_response and has_docs:
                success_count += 1
                print(f"\n✅ PIPELINE SUCCESS")
                print(f"   Category: {state['classified_category']}")
                print(f"   Documents: {len(state['retrieved_documents'])}")
                print(f"   Response: {state['final_response'][:200]}...")
            else:
                print(f"\n⚠️  INCOMPLETE RESPONSE")
                print(f"   Has Response: {has_response}")
                print(f"   Has Documents: {has_docs}")
                
        except Exception as e:
            print(f"\n❌ PIPELINE ERROR: {str(e)}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"📊 PIPELINE TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Total Tests: {total_queries}")
    print(f"Successful: {success_count}")
    print(f"Success Rate: {(success_count/total_queries)*100:.1f}%")


if __name__ == "__main__":
    """Run tests when script is executed directly"""
    print(f"\n{'#'*70}")
    print(f"#  RAG RESPONSE NODE - COMPREHENSIVE TESTING")
    print(f"{'#'*70}")
    
    try:
        # Test 1: Basic RAG node functionality
        test_rag_node_basic()
        
        # Test 2: Category handling
        test_rag_node_categories()
        
        # Test 3: Full pipeline
        test_full_pipeline()
        
        print(f"\n{'='*70}")
        print(f"✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print(f"{'='*70}")
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ TEST EXECUTION FAILED")
        print(f"{'='*70}")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
