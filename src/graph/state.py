"""
LangGraph State Schema
Defines the state structure for the chatbot workflow
"""

from typing import TypedDict, Optional, List, Dict, Any
from datetime import datetime


class ChatbotState(TypedDict, total=False):
    """
    State schema for the chatbot workflow using LangGraph
    
    This state is passed between nodes in the graph and maintains
    all necessary information for processing user queries.
    """
    
    # User Input
    user_query: str
    """The original query from the user"""
    
    # Classification
    classified_category: Optional[str]
    """
    Category assigned to the query by the classifier node.
    Possible values:
    - 'product_inquiry': Questions about products, prices, features
    - 'policy_inquiry': Questions about return, exchange, warranty policies
    - 'support_inquiry': Questions about customer support, contact info
    - 'general_inquiry': General questions that can be answered from knowledge base
    - 'escalate': Queries that need human intervention
    - 'out_of_scope': Questions outside the knowledge base
    """
    
    # Response Generation
    final_response: Optional[str]
    """The final response generated for the user"""
    
    # Additional Context (optional fields)
    retrieved_documents: Optional[List[Dict[str, Any]]]
    """Documents retrieved from the vector store for context"""
    
    confidence_score: Optional[float]
    """Confidence score of the classification (0.0 to 1.0)"""
    
    needs_escalation: Optional[bool]
    """Flag indicating if the query needs human escalation"""
    
    conversation_id: Optional[str]
    """Unique identifier for the conversation session"""
    
    timestamp: Optional[str]
    """Timestamp when the query was received"""
    
    metadata: Optional[Dict[str, Any]]
    """Additional metadata for logging and analytics"""


# Category definitions
QUERY_CATEGORIES = {
    "product_inquiry": {
        "description": "Questions about products, prices, features, specifications",
        "examples": [
            "What is the price of SmartWatch Pro X?",
            "Tell me about Wireless Earbuds Elite features",
            "Does the Power Bank support fast charging?"
        ]
    },
    "policy_inquiry": {
        "description": "Questions about return, exchange, warranty, shipping policies",
        "examples": [
            "What is your return policy?",
            "How can I exchange a product?",
            "What is the warranty period?"
        ]
    },
    "support_inquiry": {
        "description": "Questions about customer support, contact information, support hours",
        "examples": [
            "How can I contact customer support?",
            "What are your support hours?",
            "Can I chat with someone?"
        ]
    },
    "general_inquiry": {
        "description": "General questions that can be answered from knowledge base",
        "examples": [
            "Do you have payment on delivery?",
            "What payment methods do you accept?",
            "How long does shipping take?"
        ]
    },
    "escalate": {
        "description": "Complex queries requiring human intervention",
        "examples": [
            "I want to file a complaint",
            "I have an issue with my order",
            "This is not working properly"
        ]
    },
    "out_of_scope": {
        "description": "Questions outside the knowledge base or domain",
        "examples": [
            "What is the weather today?",
            "Tell me about iPhone 15",
            "Do you sell laptops?"
        ]
    }
}


def create_initial_state(user_query: str, conversation_id: Optional[str] = None) -> ChatbotState:
    """
    Create an initial state for a new user query
    
    Args:
        user_query: The user's question
        conversation_id: Optional conversation identifier
        
    Returns:
        ChatbotState with initialized fields
    """
    import uuid
    
    state: ChatbotState = {
        "user_query": user_query,
        "classified_category": None,
        "final_response": None,
        "retrieved_documents": None,
        "confidence_score": None,
        "needs_escalation": False,
        "conversation_id": conversation_id or str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "metadata": {}
    }
    
    return state


def update_state(
    current_state: ChatbotState,
    **updates
) -> ChatbotState:
    """
    Update the state with new values
    
    Args:
        current_state: Current state dictionary
        **updates: Key-value pairs to update
        
    Returns:
        Updated ChatbotState
    """
    updated_state = current_state.copy()
    updated_state.update(updates)
    return updated_state


def get_category_description(category: str) -> str:
    """
    Get description for a category
    
    Args:
        category: Category name
        
    Returns:
        Category description
    """
    return QUERY_CATEGORIES.get(category, {}).get("description", "Unknown category")


def get_category_examples(category: str) -> List[str]:
    """
    Get example queries for a category
    
    Args:
        category: Category name
        
    Returns:
        List of example queries
    """
    return QUERY_CATEGORIES.get(category, {}).get("examples", [])


def validate_state(state: ChatbotState) -> bool:
    """
    Validate that the state has required fields
    
    Args:
        state: State to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Check required field
    if "user_query" not in state or not state["user_query"]:
        return False
    
    # Validate category if present
    if state.get("classified_category"):
        if state["classified_category"] not in QUERY_CATEGORIES:
            return False
    
    return True


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("  LANGGRAPH STATE SCHEMA DEMONSTRATION")
    print("="*70)
    
    # Create initial state
    print("\n1. Creating initial state...")
    state = create_initial_state("What is the price of SmartWatch Pro X?")
    
    print(f"\n✅ Initial State Created:")
    print(f"   User Query: {state['user_query']}")
    print(f"   Conversation ID: {state['conversation_id']}")
    print(f"   Timestamp: {state['timestamp']}")
    print(f"   Classified Category: {state['classified_category']}")
    print(f"   Final Response: {state['final_response']}")
    
    # Update state with classification
    print("\n2. Updating state with classification...")
    state = update_state(
        state,
        classified_category="product_inquiry",
        confidence_score=0.95
    )
    
    print(f"\n✅ State Updated:")
    print(f"   Classified Category: {state['classified_category']}")
    print(f"   Confidence Score: {state['confidence_score']}")
    
    # Update state with response
    print("\n3. Updating state with final response...")
    state = update_state(
        state,
        final_response="The SmartWatch Pro X is priced at ₹15,999.",
        retrieved_documents=[
            {"source": "product_info.txt", "content": "SmartWatch Pro X Price: ₹15,999"}
        ]
    )
    
    print(f"\n✅ State Updated:")
    print(f"   Final Response: {state['final_response']}")
    print(f"   Retrieved Documents: {len(state['retrieved_documents'])} document(s)")
    
    # Validate state
    print("\n4. Validating state...")
    is_valid = validate_state(state)
    print(f"   State Valid: {'✅ Yes' if is_valid else '❌ No'}")
    
    # Display all categories
    print(f"\n{'='*70}")
    print("  AVAILABLE QUERY CATEGORIES")
    print(f"{'='*70}")
    
    for category, info in QUERY_CATEGORIES.items():
        print(f"\n📁 {category.upper().replace('_', ' ')}")
        print(f"   Description: {info['description']}")
        print(f"   Examples:")
        for example in info['examples']:
            print(f"      • {example}")
    
    print(f"\n{'='*70}")
    print("  ✅ STATE SCHEMA DEMONSTRATION COMPLETE")
    print(f"{'='*70}")
    
    # Show complete state structure
    print(f"\n📊 Complete State Structure:")
    print(f"   Fields: {list(state.keys())}")
    print(f"   Required: user_query")
    print(f"   Optional: All other fields")
