"""
Knowledge Base Loader
Load product_info.txt using LangChain's TextLoader
"""

from langchain_community.document_loaders import TextLoader
from pathlib import Path
import sys

def load_knowledge_base(file_path: str):
    """Load knowledge base from text file using LangChain TextLoader"""
    
    print("=" * 70)
    print("📚 LOADING KNOWLEDGE BASE WITH LANGCHAIN")
    print("=" * 70)
    
    # Check if file exists
    if not Path(file_path).exists():
        print(f"❌ Error: File not found at {file_path}")
        return None
    
    try:
        # Initialize TextLoader
        print(f"\n📂 Loading file: {file_path}")
        loader = TextLoader(file_path, encoding='utf-8')
        
        # Load documents
        print("⏳ Loading documents...")
        documents = loader.load()
        
        print(f"✅ Successfully loaded {len(documents)} document(s)\n")
        
        # Display document information
        print("=" * 70)
        print("📄 DOCUMENT DETAILS")
        print("=" * 70)
        
        for i, doc in enumerate(documents, 1):
            print(f"\n📑 Document {i}:")
            print(f"   Type: {type(doc).__name__}")
            print(f"   Content Length: {len(doc.page_content)} characters")
            print(f"   Metadata: {doc.metadata}")
            
            # Show preview of content
            print(f"\n   📝 Content Preview (first 500 characters):")
            print("   " + "-" * 66)
            preview = doc.page_content[:500].replace('\n', '\n   ')
            print(f"   {preview}")
            if len(doc.page_content) > 500:
                print(f"   ... ({len(doc.page_content) - 500} more characters)")
            print("   " + "-" * 66)
        
        # Additional statistics
        print("\n" + "=" * 70)
        print("📊 STATISTICS")
        print("=" * 70)
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        total_words = sum(len(doc.page_content.split()) for doc in documents)
        total_lines = sum(doc.page_content.count('\n') for doc in documents)
        
        print(f"✅ Total Documents: {len(documents)}")
        print(f"✅ Total Characters: {total_chars:,}")
        print(f"✅ Total Words: {total_words:,}")
        print(f"✅ Total Lines: {total_lines:,}")
        
        # Check for products
        product_count = doc.page_content.count('Product:')
        print(f"✅ Products Found: {product_count}")
        
        print("\n" + "=" * 70)
        print("✅ KNOWLEDGE BASE LOADED SUCCESSFULLY!")
        print("=" * 70)
        
        return documents
        
    except Exception as e:
        print(f"\n❌ Error loading knowledge base: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Path to knowledge base file
    knowledge_base_path = "data/knowledge_base/product_info.txt"
    
    # Load the knowledge base
    documents = load_knowledge_base(knowledge_base_path)
    
    if documents:
        print("\n🎉 Ready for next step: Text Chunking")
        sys.exit(0)
    else:
        print("\n❌ Failed to load knowledge base")
        sys.exit(1)
