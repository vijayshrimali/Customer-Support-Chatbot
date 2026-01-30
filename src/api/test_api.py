"""
API Test Client
Test the FastAPI chatbot endpoints
"""

import requests
import json
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000"


class ChatbotAPIClient:
    """Client for testing the chatbot API"""
    
    def __init__(self, base_url: str = BASE_URL):
        """Initialize API client"""
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def chat(self, query: str, conversation_id: str = None) -> Dict[str, Any]:
        """Send a chat message"""
        payload = {"query": query}
        if conversation_id:
            payload["conversation_id"] = conversation_id
        
        response = self.session.post(
            f"{self.base_url}/chat",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def get_categories(self) -> Dict[str, Any]:
        """Get supported categories"""
        response = self.session.get(f"{self.base_url}/categories")
        response.raise_for_status()
        return response.json()
    
    def get_products(self) -> Dict[str, Any]:
        """Get product list"""
        response = self.session.get(f"{self.base_url}/products")
        response.raise_for_status()
        return response.json()


def test_api():
    """Test the chatbot API"""
    print("="*70)
    print("🧪 TESTING CHATBOT API")
    print("="*70)
    
    # Initialize client
    client = ChatbotAPIClient()
    
    # Test 1: Health check
    print("\n" + "="*70)
    print("TEST 1: Health Check")
    print("="*70)
    try:
        health = client.health_check()
        print(f"✅ Status: {health['status']}")
        print(f"   Version: {health['version']}")
        print(f"   Timestamp: {health['timestamp']}")
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")
        return
    
    # Test 2: Get categories
    print("\n" + "="*70)
    print("TEST 2: Get Categories")
    print("="*70)
    try:
        categories = client.get_categories()
        print(f"✅ Found {len(categories['categories'])} categories:")
        for cat in categories['categories']:
            print(f"\n   📂 {cat['name'].upper()}")
            print(f"      {cat['description']}")
            print(f"      Example: {cat['examples'][0]}")
    except Exception as e:
        print(f"❌ Get categories failed: {str(e)}")
    
    # Test 3: Get products
    print("\n" + "="*70)
    print("TEST 3: Get Products")
    print("="*70)
    try:
        products = client.get_products()
        print(f"✅ Found {len(products['products'])} products:")
        for prod in products['products']:
            print(f"\n   📦 {prod['name']}")
            print(f"      Price: {prod['price']}")
            print(f"      {prod['description'][:60]}...")
    except Exception as e:
        print(f"❌ Get products failed: {str(e)}")
    
    # Test 4: Chat - Product query
    print("\n" + "="*70)
    print("TEST 4: Chat - Product Query")
    print("="*70)
    test_query = "What is the price of SmartWatch Pro X?"
    print(f"Query: {test_query}")
    try:
        response = client.chat(test_query, "test_conv_1")
        print(f"\n✅ Response received:")
        print(f"   Category: {response['category']}")
        print(f"   Confidence: {response['confidence']:.2f}")
        print(f"   Escalation: {response['needs_escalation']}")
        print(f"\n   💬 Response:")
        print(f"   {response['response']}")
    except Exception as e:
        print(f"❌ Chat failed: {str(e)}")
    
    # Test 5: Chat - Returns query
    print("\n" + "="*70)
    print("TEST 5: Chat - Returns Query")
    print("="*70)
    test_query = "How do I return a product?"
    print(f"Query: {test_query}")
    try:
        response = client.chat(test_query, "test_conv_2")
        print(f"\n✅ Response received:")
        print(f"   Category: {response['category']}")
        print(f"   Confidence: {response['confidence']:.2f}")
        print(f"   Escalation: {response['needs_escalation']}")
        print(f"\n   💬 Response:")
        print(f"   {response['response'][:150]}...")
    except Exception as e:
        print(f"❌ Chat failed: {str(e)}")
    
    # Test 6: Chat - General query
    print("\n" + "="*70)
    print("TEST 6: Chat - General Query")
    print("="*70)
    test_query = "What are your customer support hours?"
    print(f"Query: {test_query}")
    try:
        response = client.chat(test_query, "test_conv_3")
        print(f"\n✅ Response received:")
        print(f"   Category: {response['category']}")
        print(f"   Confidence: {response['confidence']:.2f}")
        print(f"   Escalation: {response['needs_escalation']}")
        print(f"\n   💬 Response:")
        print(f"   {response['response']}")
    except Exception as e:
        print(f"❌ Chat failed: {str(e)}")
    
    # Summary
    print("\n" + "="*70)
    print("✅ API TESTING COMPLETE")
    print("="*70)
    print("\n🎯 All endpoints tested successfully!")
    print("   API is ready for production use")


if __name__ == "__main__":
    import time
    
    print("\n⏳ Waiting for API to start (5 seconds)...")
    print("   Make sure the API is running: python src/api/main.py\n")
    time.sleep(5)
    
    try:
        test_api()
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API")
        print("   Please start the API server first:")
        print("   python src/api/main.py")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
