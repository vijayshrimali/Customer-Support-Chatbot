"""
Query Classifier Node - LangGraph Node Implementation
Classifies user queries using rule-based keyword matching
"""

import re
from typing import Dict, List, Tuple
from state import ChatbotState, update_state


# Keyword patterns for each category
CATEGORY_KEYWORDS = {
    "product": [
        # Product names
        r"smartwatch|smart watch|watch",
        r"earbuds|earphones|headphones",
        r"power bank|powerbank|battery pack",
        
        # Product-related queries
        r"price|cost|how much",
        r"features|specifications|specs",
        r"available|in stock|buy|purchase",
        r"model|version|variant",
        r"color|colour|size",
        r"battery life|charging",
        r"warranty period|guarantee",
        r"water resistant|waterproof",
        r"compatible|compatibility",
        r"review|rating|feedback"
    ],
    
    "returns": [
        # Return/Exchange related
        r"return|refund",
        r"exchange|replace|replacement",
        r"cancel|cancellation",
        r"warranty|guarantee",
        r"defective|damaged|broken|faulty",
        r"not working|issue|problem",
        r"complaint|dispute",
        r"money back|get my money",
        r"send back|ship back"
    ],
    
    "general": [
        # Support and general queries
        r"support|help|assist",
        r"contact|reach|call|email|phone",
        r"hours|timing|when open",
        r"shipping|delivery|dispatch",
        r"payment|pay|cod|cash on delivery",
        r"track|tracking|order status",
        r"account|login|register|sign up",
        r"location|address|store",
        r"offer|discount|coupon|deal"
    ]
}


class QueryClassifier:
    """
    Rule-based query classifier using keyword matching
    """
    
    def __init__(self):
        """Initialize the classifier"""
        self.categories = list(CATEGORY_KEYWORDS.keys())
        print(f"✅ QueryClassifier initialized")
        print(f"   Categories: {', '.join(self.categories)}")
    
    def preprocess_query(self, query: str) -> str:
        """
        Preprocess query for matching
        
        Args:
            query: Raw user query
            
        Returns:
            Preprocessed query (lowercase, trimmed)
        """
        # Convert to lowercase and strip whitespace
        processed = query.lower().strip()
        return processed
    
    def match_keywords(self, query: str, keywords: List[str]) -> Tuple[bool, int]:
        """
        Match keywords in query using regex patterns
        
        Args:
            query: Preprocessed query
            keywords: List of regex patterns to match
            
        Returns:
            Tuple of (matched: bool, match_count: int)
        """
        match_count = 0
        
        for pattern in keywords:
            if re.search(pattern, query, re.IGNORECASE):
                match_count += 1
        
        return match_count > 0, match_count
    
    def classify(self, query: str) -> Dict[str, any]:
        """
        Classify query into categories
        
        Args:
            query: User query string
            
        Returns:
            Classification result dict with category and confidence
        """
        # Preprocess query
        processed_query = self.preprocess_query(query)
        
        # Score each category
        category_scores = {}
        
        for category, keywords in CATEGORY_KEYWORDS.items():
            matched, count = self.match_keywords(processed_query, keywords)
            category_scores[category] = count
        
        # Determine best category
        max_score = max(category_scores.values())
        
        if max_score == 0:
            # No keywords matched - classify as general
            classified_category = "general"
            confidence = 0.3  # Low confidence
        else:
            # Get category with highest score
            classified_category = max(category_scores, key=category_scores.get)
            
            # Calculate confidence score
            total_matches = sum(category_scores.values())
            confidence = category_scores[classified_category] / total_matches if total_matches > 0 else 0.5
            
            # Boost confidence if significantly higher than others
            if max_score > 1:
                confidence = min(confidence + 0.2, 1.0)
        
        return {
            "category": classified_category,
            "confidence": round(confidence, 2),
            "scores": category_scores
        }
    
    def explain_classification(self, query: str, result: Dict) -> str:
        """
        Explain why a query was classified a certain way
        
        Args:
            query: Original query
            result: Classification result
            
        Returns:
            Explanation string
        """
        category = result["category"]
        confidence = result["confidence"]
        scores = result["scores"]
        
        explanation = f"Query: '{query}'\n"
        explanation += f"Classified as: {category.upper()}\n"
        explanation += f"Confidence: {confidence:.2f}\n"
        explanation += f"Match scores: {scores}"
        
        return explanation


def classifier_node(state: ChatbotState) -> ChatbotState:
    """
    LangGraph node that classifies the user query
    
    This is the main node function that will be used in the LangGraph workflow
    
    Args:
        state: Current chatbot state
        
    Returns:
        Updated state with classification
    """
    # Get user query from state
    user_query = state.get("user_query", "")
    
    if not user_query:
        # No query to classify
        return update_state(
            state,
            classified_category="general",
            confidence_score=0.0,
            metadata={"error": "Empty query"}
        )
    
    # Initialize classifier
    classifier = QueryClassifier()
    
    # Classify the query
    result = classifier.classify(user_query)
    
    # Update state with classification results
    updated_state = update_state(
        state,
        classified_category=result["category"],
        confidence_score=result["confidence"],
        metadata={
            "classification_scores": result["scores"],
            "classifier_type": "rule-based"
        }
    )
    
    return updated_state


def test_classifier():
    """
    Test the classifier with sample queries
    """
    print(f"\n{'='*70}")
    print(f"  QUERY CLASSIFIER TESTING")
    print(f"{'='*70}")
    
    # Initialize classifier
    classifier = QueryClassifier()
    
    # Test queries
    test_queries = [
        "What is the price of SmartWatch Pro X?",
        "Tell me about Wireless Earbuds features",
        "How can I return a product?",
        "I want to exchange my defective earbuds",
        "What are your customer support hours?",
        "How can I contact you?",
        "Do you accept cash on delivery?",
        "Is the power bank waterproof?",
        "My order is not working properly",
        "What payment methods do you accept?",
        "Can I cancel my order?",
        "What is the warranty on SmartWatch?"
    ]
    
    print(f"\n🧪 Testing {len(test_queries)} queries")
    print(f"{'='*70}\n")
    
    # Track results
    results_by_category = {"product": 0, "returns": 0, "general": 0}
    
    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}: {query}")
        
        # Classify
        result = classifier.classify(query)
        
        category = result["category"]
        confidence = result["confidence"]
        scores = result["scores"]
        
        results_by_category[category] += 1
        
        print(f"  → Category: {category.upper()}")
        print(f"  → Confidence: {confidence:.2f}")
        print(f"  → Scores: {scores}")
        print()
    
    # Summary
    print(f"{'='*70}")
    print(f"  CLASSIFICATION SUMMARY")
    print(f"{'='*70}\n")
    
    for category, count in results_by_category.items():
        percentage = (count / len(test_queries)) * 100
        print(f"{category.upper()}: {count}/{len(test_queries)} ({percentage:.1f}%)")
    
    print(f"\n{'='*70}")
    print(f"  ✅ CLASSIFIER TESTING COMPLETE")
    print(f"{'='*70}")


def test_with_state():
    """
    Test classifier node with ChatbotState
    """
    from state import create_initial_state
    
    print(f"\n{'='*70}")
    print(f"  TESTING CLASSIFIER NODE WITH STATE")
    print(f"{'='*70}")
    
    # Test queries
    test_queries = [
        "What is the price of SmartWatch?",
        "How do I return a product?",
        "What are your support hours?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─'*70}")
        print(f"Test {i}/{len(test_queries)}")
        print(f"{'─'*70}")
        
        # Create initial state
        state = create_initial_state(query)
        
        print(f"\nInitial State:")
        print(f"  user_query: {state['user_query']}")
        print(f"  classified_category: {state['classified_category']}")
        print(f"  confidence_score: {state['confidence_score']}")
        
        # Run classifier node
        updated_state = classifier_node(state)
        
        print(f"\nUpdated State:")
        print(f"  user_query: {updated_state['user_query']}")
        print(f"  classified_category: {updated_state['classified_category']}")
        print(f"  confidence_score: {updated_state['confidence_score']}")
        print(f"  metadata: {updated_state.get('metadata', {})}")
    
    print(f"\n{'='*70}")
    print(f"  ✅ STATE-BASED TESTING COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    # Run tests
    test_classifier()
    test_with_state()
