#!/usr/bin/env python3
"""Verify that all dependencies are installed correctly."""

import sys

def verify_imports():
    """Test importing all required packages."""
    print("=" * 60)
    print("Verifying TechGear RAG Chatbot Setup")
    print("=" * 60)
    
    packages = [
        ("langchain", "LangChain"),
        ("langchain_core", "LangChain Core"),
        ("langchain_google_genai", "LangChain Google GenAI"),
        ("langchain_community", "LangChain Community"),
        ("langchain_text_splitters", "LangChain Text Splitters"),
        ("chromadb", "ChromaDB"),
        ("langgraph", "LangGraph"),
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("pydantic", "Pydantic"),
        ("pydantic_settings", "Pydantic Settings"),
        ("dotenv", "Python Dotenv"),
        ("google.generativeai", "Google Generative AI"),
        ("tiktoken", "Tiktoken"),
        ("requests", "Requests"),
        ("pytest", "Pytest"),
        ("httpx", "HTTPX"),
    ]
    
    failed = []
    
    for module, name in packages:
        try:
            __import__(module)
            print(f"✅ {name:30} - OK")
        except ImportError as e:
            print(f"❌ {name:30} - FAILED")
            failed.append((name, str(e)))
    
    print("=" * 60)
    
    if failed:
        print("\n❌ Setup verification FAILED\n")
        print("Failed imports:")
        for name, error in failed:
            print(f"  - {name}: {error}")
        return False
    else:
        print("\n✅ All dependencies installed successfully!")
        print("\nProject is ready for development!")
        return True

if __name__ == "__main__":
    success = verify_imports()
    sys.exit(0 if success else 1)
