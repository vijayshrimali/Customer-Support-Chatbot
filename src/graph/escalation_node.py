"""
Escalation Node
LangGraph node that handles queries requiring human support
"""

import os
import sys
from typing import Dict, Any
from datetime import datetime

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from graph.state import ChatbotState, update_state


class EscalationHandler:
    """
    Escalation Handler for Customer Support
    
    Handles queries that require human intervention:
    - Returns and refunds
    - Exchanges and complaints
    - Out-of-scope queries
    - Complex issues beyond knowledge base
    """
    
    # Support contact information
    SUPPORT_EMAIL = "support@techgear.com"
    SUPPORT_PHONE = "1800-123-4567"
    SUPPORT_HOURS = "Monday to Saturday, 9 AM to 6 PM IST"
    SUPPORT_WEBSITE = "www.techgear.com/support"
    
    def __init__(self):
        """Initialize the Escalation Handler"""
        print(f"\n{'='*70}")
        print(f"🆘 Initializing Escalation Handler")
        print(f"{'='*70}")
        print(f"   Support Email: {self.SUPPORT_EMAIL}")
        print(f"   Support Phone: {self.SUPPORT_PHONE}")
        print(f"   Support Hours: {self.SUPPORT_HOURS}")
        print(f"✅ Escalation Handler initialized!")
    
    def should_escalate(self, category: str) -> bool:
        """
        Determine if a query should be escalated
        
        Args:
            category: Classified category from classifier node
            
        Returns:
            bool: True if query needs escalation
        """
        # Categories that require escalation
        escalation_categories = [
            'returns',
            'policy_inquiry',
            'escalate',
            'out_of_scope',
            'complaint',
            'issue'
        ]
        
        return category in escalation_categories if category else False
    
    def generate_escalation_message(
        self,
        category: str,
        query: str,
        confidence: float = 1.0
    ) -> Dict[str, Any]:
        """
        Generate appropriate escalation message based on category
        
        Args:
            category: Query category
            query: Original user query
            confidence: Classification confidence
            
        Returns:
            Dict containing response and metadata
        """
        print(f"\n{'='*70}")
        print(f"🆘 Generating Escalation Message")
        print(f"{'='*70}")
        print(f"Category: {category}")
        print(f"Query: {query}")
        print(f"Confidence: {confidence:.2f}")
        
        # Select appropriate message based on category
        if category == 'returns':
            response = self._get_returns_message()
            reason = "return_refund_request"
        elif category == 'out_of_scope':
            response = self._get_out_of_scope_message()
            reason = "query_outside_knowledge_base"
        elif category == 'policy_inquiry':
            response = self._get_policy_inquiry_message()
            reason = "policy_clarification_needed"
        else:
            response = self._get_general_escalation_message()
            reason = "general_support_needed"
        
        print(f"\n✅ Escalation message generated")
        print(f"   Reason: {reason}")
        print(f"   Response length: {len(response)} characters")
        
        return {
            'response': response,
            'escalation_reason': reason,
            'requires_human': True,
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_returns_message(self) -> str:
        """Generate message for return/refund requests"""
        return f"""Thank you for contacting TechGear Electronics regarding your return or refund request.

📋 **Return Policy Summary:**
• 7-day no-questions-asked return policy
• Full refund processed within 5-7 business days
• Product must be in original condition with all accessories
• Free pickup available for defective products

🔄 **To Process Your Return:**
Please contact our support team with:
1. Order number
2. Product name and details
3. Reason for return (optional)
4. Photos of the product (if defective)

📞 **Contact Our Support Team:**
• **Email:** {self.SUPPORT_EMAIL}
• **Phone:** {self.SUPPORT_PHONE}
• **Hours:** {self.SUPPORT_HOURS}
• **Website:** {self.SUPPORT_WEBSITE}

Our team will assist you with the return process and arrange pickup if needed. We typically respond within 2-4 hours during business hours.

Is there anything else I can help you with regarding our products or policies?"""
    
    def _get_out_of_scope_message(self) -> str:
        """Generate message for out-of-scope queries"""
        return f"""Thank you for your inquiry!

I apologize, but I don't have information about that in my current knowledge base. 

✅ **I Can Help You With:**

**Products:**
• SmartWatch Pro X (₹15,999) - Features, specifications, warranty
• Wireless Earbuds Elite (₹4,999) - Features, battery life, compatibility
• Power Bank Ultra 20000mAh (₹2,499) - Capacity, charging speed, warranty

**Services & Policies:**
• Return and exchange procedures
• Warranty coverage and claims
• Payment methods and offers
• Shipping and delivery information
• Customer support and contact details

📞 **For Other Inquiries:**
Please contact our support team directly:
• **Email:** {self.SUPPORT_EMAIL}
• **Phone:** {self.SUPPORT_PHONE}
• **Hours:** {self.SUPPORT_HOURS}

Our team can assist you with product recommendations, custom orders, bulk purchases, and any other questions beyond my current scope.

How else can I assist you with our available products or services?"""
    
    def _get_policy_inquiry_message(self) -> str:
        """Generate message for policy-related queries needing clarification"""
        return f"""Thank you for your policy inquiry!

For detailed information about our policies or if you need specific clarification, I recommend contacting our support team who can provide comprehensive guidance tailored to your situation.

📋 **General Policy Information Available:**
• Returns: 7-day return policy
• Warranty: 1-year standard warranty on all products
• Shipping: Free shipping on orders above ₹500
• Payment: Multiple payment options including COD

📞 **For Detailed Policy Information:**
• **Email:** {self.SUPPORT_EMAIL}
• **Phone:** {self.SUPPORT_PHONE}
• **Hours:** {self.SUPPORT_HOURS}
• **Website:** {self.SUPPORT_WEBSITE}

Our support team can provide:
• Specific policy details for your situation
• Exception cases and special circumstances
• Documentation and written confirmations
• Step-by-step guidance

Is there anything specific about our products I can help you with right now?"""
    
    def _get_general_escalation_message(self) -> str:
        """Generate general escalation message"""
        return f"""Thank you for contacting TechGear Electronics!

For personalized assistance with your inquiry, I recommend reaching out to our support team who can provide detailed help tailored to your needs.

📞 **Contact Our Support Team:**
• **Email:** {self.SUPPORT_EMAIL}
• **Phone:** {self.SUPPORT_PHONE}
• **Hours:** {self.SUPPORT_HOURS}
• **Website:** {self.SUPPORT_WEBSITE}

⚡ **Quick Response Times:**
• Email: 2-4 hours during business hours
• Phone: Immediate assistance
• Chat: Available on our website

💬 **Meanwhile, I Can Help With:**
• Product features, specifications, and pricing
• Warranty information and coverage
• Return policy and procedures
• Payment methods and shipping details
• General product information

Would you like to know more about any of our products (SmartWatch Pro X, Wireless Earbuds Elite, or Power Bank Ultra)?"""
    
    def process(self, state: ChatbotState) -> ChatbotState:
        """
        Process the state and generate escalation response
        
        Args:
            state: Current chatbot state
            
        Returns:
            Updated chatbot state with escalation response
        """
        print(f"\n{'='*70}")
        print(f"📊 Escalation Node Processing")
        print(f"{'='*70}")
        
        # Extract query and category
        query = state.get('user_query', '')
        category = state.get('classified_category', '')
        confidence = state.get('confidence_score', 0.0)
        
        print(f"User Query: {query}")
        print(f"Category: {category}")
        print(f"Confidence: {confidence:.2f}")
        
        # Check if escalation is needed
        if not self.should_escalate(category):
            print(f"\n⚠️  Escalation not needed for category: {category}")
            print(f"   This query should be handled by another node")
            return state
        
        # Generate escalation message
        print(f"\n🚀 Generating escalation response...")
        result = self.generate_escalation_message(category, query, confidence)
        
        # Update state
        updated_state = update_state(
            state,
            final_response=result['response'],
            needs_escalation=True,
            metadata={
                **(state.get('metadata', {}) or {}),
                'escalation_reason': result['escalation_reason'],
                'requires_human': result['requires_human'],
                'escalation_timestamp': result['timestamp'],
                'support_email': self.SUPPORT_EMAIL,
                'support_phone': self.SUPPORT_PHONE
            }
        )
        
        print(f"\n✅ State updated with escalation response")
        print(f"   Escalation reason: {result['escalation_reason']}")
        print(f"   Response preview: {result['response'][:100]}...")
        
        return updated_state


# Global escalation handler instance (singleton pattern)
_escalation_handler_instance = None


def get_escalation_handler() -> EscalationHandler:
    """
    Get or create escalation handler instance (singleton)
    
    Returns:
        EscalationHandler instance
    """
    global _escalation_handler_instance
    if _escalation_handler_instance is None:
        _escalation_handler_instance = EscalationHandler()
    return _escalation_handler_instance


def escalation_node(state: ChatbotState) -> ChatbotState:
    """
    LangGraph node function for escalating queries
    
    This is the main entry point for the escalation node in the LangGraph workflow.
    
    Args:
        state: Current chatbot state from LangGraph
        
    Returns:
        Updated chatbot state with escalation response
        
    Example:
        >>> from graph.state import create_initial_state
        >>> from graph.classifier_node import classifier_node
        >>> 
        >>> # Create and classify state
        >>> state = create_initial_state("I want to return my product")
        >>> state = classifier_node(state)
        >>> 
        >>> # Generate escalation response
        >>> state = escalation_node(state)
        >>> print(state['final_response'])
    """
    # Get singleton escalation handler instance
    handler = get_escalation_handler()
    
    # Process state and generate response
    return handler.process(state)


# =============================================================================
# TESTING FUNCTIONS
# =============================================================================

def test_escalation_categories():
    """Test escalation node with different categories"""
    print(f"\n{'='*70}")
    print(f"🧪 TESTING ESCALATION NODE - Category Handling")
    print(f"{'='*70}")
    
    from graph.state import create_initial_state
    
    # Test different categories
    test_cases = [
        ("I want to return my SmartWatch", "returns"),
        ("Do you sell laptops?", "out_of_scope"),
        ("What is your exchange policy?", "policy_inquiry"),
        ("I have a complaint about my order", "general"),
    ]
    
    for i, (query, category) in enumerate(test_cases, 1):
        print(f"\n{'-'*70}")
        print(f"Test {i}/{len(test_cases)}: {category.upper()}")
        print(f"{'-'*70}")
        print(f"Query: {query}")
        print(f"Category: {category}")
        
        # Create state with pre-classified category
        state = create_initial_state(query)
        state = update_state(
            state,
            classified_category=category,
            confidence_score=1.0
        )
        
        # Generate escalation response
        state = escalation_node(state)
        
        # Verify
        final_response = state.get('final_response') or ''
        print(f"\n📊 Results:")
        print(f"   Has Response: {'✅' if final_response else '❌'}")
        print(f"   Needs Escalation: {'✅' if state.get('needs_escalation') else '❌'}")
        print(f"   Support Email Included: {'✅' if 'support@techgear.com' in final_response else '❌'}")
        print(f"   Response Length: {len(final_response)} chars")
        if final_response:
            print(f"\n   Response Preview:")
            print(f"   {final_response[:200]}...")


def test_full_pipeline_with_escalation():
    """Test complete pipeline: classification → escalation"""
    print(f"\n{'='*70}")
    print(f"🧪 TESTING FULL PIPELINE - Classification → Escalation")
    print(f"{'='*70}")
    
    from graph.state import create_initial_state
    from graph.classifier_node import classifier_node
    
    # Return queries that should escalate
    escalation_queries = [
        "How do I return my defective earbuds?",
        "I want a refund for my smartwatch",
        "Can I exchange my power bank?",
    ]
    
    success_count = 0
    total_queries = len(escalation_queries)
    
    for i, query in enumerate(escalation_queries, 1):
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
            
            # Step 3: Generate escalation response
            state = escalation_node(state)
            print(f"✅ Step 3: Escalation response generated")
            
            # Verify response
            has_response = bool(state.get('final_response'))
            needs_escalation = state.get('needs_escalation', False)
            final_response = state.get('final_response') or ''
            has_contact_info = 'support@techgear.com' in final_response
            
            if has_response and needs_escalation and has_contact_info:
                success_count += 1
                print(f"\n✅ PIPELINE SUCCESS")
                print(f"   Category: {state['classified_category']}")
                print(f"   Needs Escalation: {needs_escalation}")
                print(f"   Contact Info: Included")
                print(f"   Response: {state['final_response'][:150]}...")
            else:
                print(f"\n⚠️  INCOMPLETE RESPONSE")
                print(f"   Has Response: {has_response}")
                print(f"   Needs Escalation: {needs_escalation}")
                print(f"   Has Contact Info: {has_contact_info}")
                
        except Exception as e:
            print(f"\n❌ PIPELINE ERROR: {str(e)}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"📊 PIPELINE TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Total Tests: {total_queries}")
    print(f"Successful: {success_count}")
    print(f"Success Rate: {(success_count/total_queries)*100:.1f}%")


def test_message_content():
    """Test the content of different escalation messages"""
    print(f"\n{'='*70}")
    print(f"🧪 TESTING ESCALATION MESSAGES - Content Verification")
    print(f"{'='*70}")
    
    handler = EscalationHandler()
    
    # Test each message type
    categories = ['returns', 'out_of_scope', 'policy_inquiry', 'general']
    
    for category in categories:
        print(f"\n{'-'*70}")
        print(f"Testing: {category.upper()} Message")
        print(f"{'-'*70}")
        
        result = handler.generate_escalation_message(
            category=category,
            query=f"Test query for {category}",
            confidence=1.0
        )
        
        response = result['response']
        
        # Verify required elements
        checks = {
            'Email': handler.SUPPORT_EMAIL in response,
            'Phone': handler.SUPPORT_PHONE in response,
            'Hours': handler.SUPPORT_HOURS in response,
            'Polite Tone': any(word in response.lower() for word in ['thank you', 'please', 'apologize']),
            'Call to Action': '?' in response,
            'Length > 100 chars': len(response) > 100
        }
        
        print(f"\nContent Verification:")
        for check, passed in checks.items():
            print(f"   {check}: {'✅' if passed else '❌'}")
        
        print(f"\nMessage Stats:")
        print(f"   Length: {len(response)} characters")
        print(f"   Lines: {response.count(chr(10)) + 1}")
        print(f"   Escalation Reason: {result['escalation_reason']}")


if __name__ == "__main__":
    """Run tests when script is executed directly"""
    print(f"\n{'#'*70}")
    print(f"#  ESCALATION NODE - COMPREHENSIVE TESTING")
    print(f"{'#'*70}")
    
    try:
        # Test 1: Category handling
        test_escalation_categories()
        
        # Test 2: Full pipeline
        test_full_pipeline_with_escalation()
        
        # Test 3: Message content
        test_message_content()
        
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
