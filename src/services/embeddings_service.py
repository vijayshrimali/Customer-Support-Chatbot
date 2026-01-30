"""
Embeddings Service
Generate vector embeddings using Google Gemini Embedding Model
"""

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from typing import List
from dotenv import load_dotenv
import os


class EmbeddingsService:
    """Service for generating embeddings using Google Gemini"""
    
    def __init__(self, api_key: str = None, model_name: str = "models/embedding-001"):
        """
        Initialize the embeddings service
        
        Args:
            api_key: Google Gemini API key (if None, loads from environment)
            model_name: Name of the embedding model to use
        """
        # Load environment variables
        load_dotenv()
        
        # Get API key
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Please set it in .env file or pass it as parameter."
            )
        
        self.model_name = model_name
        
        # Initialize Google Generative AI Embeddings
        print(f"🔧 Initializing Google Gemini Embeddings...")
        print(f"   Model: {model_name}")
        print(f"   API Key: {self.api_key[:20]}...{self.api_key[-4:]}")
        
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=self.api_key
            )
            print(f"✅ Embeddings service initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error initializing embeddings service: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (each is a list of floats)
        """
        print(f"\n⏳ Generating embeddings for {len(texts)} documents...")
        
        try:
            embeddings = self.embeddings.embed_documents(texts)
            
            print(f"✅ Successfully generated {len(embeddings)} embeddings")
            if embeddings:
                print(f"   Embedding dimension: {len(embeddings[0])}")
            
            return embeddings
            
        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector (list of floats)
        """
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
            
        except Exception as e:
            print(f"❌ Error embedding query: {e}")
            raise
    
    def get_embeddings_object(self):
        """
        Get the embeddings object for use in vector stores
        
        Returns:
            GoogleGenerativeAIEmbeddings object
        """
        return self.embeddings


def generate_embeddings_for_chunks(
    file_path: str = "data/knowledge_base/product_info.txt",
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    api_key: str = None
):
    """
    Complete pipeline: Load, chunk, and generate embeddings
    
    Args:
        file_path: Path to knowledge base file
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        api_key: Google Gemini API key
        
    Returns:
        Tuple of (chunks, embeddings, embeddings_service)
    """
    print("=" * 70)
    print("🚀 STEP 5: CREATE EMBEDDINGS WITH GOOGLE GEMINI")
    print("=" * 70)
    
    # Step 1: Load documents
    print("\n📂 Step 1: Loading knowledge base...")
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} document(s)")
    
    # Step 2: Chunk documents
    print("\n✂️  Step 2: Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")
    
    # Step 3: Initialize embeddings service
    print("\n🔧 Step 3: Initializing embeddings service...")
    embeddings_service = EmbeddingsService(api_key=api_key)
    
    # Step 4: Generate embeddings
    print("\n🎯 Step 4: Generating embeddings for all chunks...")
    chunk_texts = [chunk.page_content for chunk in chunks]
    embeddings = embeddings_service.embed_documents(chunk_texts)
    
    # Display statistics
    print("\n" + "=" * 70)
    print("📊 EMBEDDINGS STATISTICS")
    print("=" * 70)
    
    print(f"\n✅ Total Chunks: {len(chunks)}")
    print(f"✅ Total Embeddings: {len(embeddings)}")
    print(f"✅ Embedding Dimension: {len(embeddings[0]) if embeddings else 0}")
    print(f"✅ Model Used: {embeddings_service.model_name}")
    
    # Show sample embedding info
    if embeddings:
        print(f"\n📊 Sample Embedding (First chunk):")
        print(f"   Chunk text: {chunk_texts[0][:100]}...")
        print(f"   Embedding vector length: {len(embeddings[0])}")
        print(f"   First 5 dimensions: {embeddings[0][:5]}")
        print(f"   Data type: {type(embeddings[0][0])}")
    
    # Verify all embeddings
    print(f"\n🔍 Verification:")
    all_same_dim = all(len(emb) == len(embeddings[0]) for emb in embeddings)
    print(f"   All embeddings same dimension: {'✅ Yes' if all_same_dim else '❌ No'}")
    print(f"   All embeddings are lists: {'✅ Yes' if all(isinstance(emb, list) for emb in embeddings) else '❌ No'}")
    print(f"   All values are floats: {'✅ Yes' if all(isinstance(val, float) for emb in embeddings for val in emb[:5]) else '❌ No'}")
    
    print("\n" + "=" * 70)
    print("✅ EMBEDDINGS GENERATION COMPLETE!")
    print("=" * 70)
    print(f"\n🎯 Ready for Step 6: Store Embeddings in ChromaDB")
    print(f"   {len(embeddings)} embeddings ready to be stored\n")
    
    return chunks, embeddings, embeddings_service


def test_embedding_service():
    """Test the embedding service with sample texts"""
    print("\n" + "=" * 70)
    print("🧪 TESTING EMBEDDINGS SERVICE")
    print("=" * 70)
    
    # Initialize service
    service = EmbeddingsService()
    
    # Test with sample texts
    test_texts = [
        "SmartWatch Pro X with heart rate monitoring",
        "Return policy for TechGear Electronics",
        "Customer support contact information"
    ]
    
    print(f"\n📝 Test texts: {len(test_texts)}")
    for i, text in enumerate(test_texts, 1):
        print(f"   {i}. {text}")
    
    # Generate embeddings
    embeddings = service.embed_documents(test_texts)
    
    print(f"\n✅ Test Results:")
    print(f"   Embeddings generated: {len(embeddings)}")
    print(f"   Embedding dimension: {len(embeddings[0])}")
    print(f"   Sample values: {embeddings[0][:3]}")
    
    # Test query embedding
    print(f"\n🔍 Testing query embedding...")
    query = "What is the price of smartwatch?"
    query_embedding = service.embed_query(query)
    print(f"   Query: {query}")
    print(f"   Embedding dimension: {len(query_embedding)}")
    print(f"   Sample values: {query_embedding[:3]}")
    
    print("\n" + "=" * 70)
    print("✅ EMBEDDING SERVICE TEST COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    
    # Check if running in test mode
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_embedding_service()
    else:
        # Generate embeddings for knowledge base
        chunks, embeddings, service = generate_embeddings_for_chunks()
        
        print(f"\n✅ Successfully generated embeddings!")
        print(f"   Chunks: {len(chunks)}")
        print(f"   Embeddings: {len(embeddings)}")
        print(f"   Dimension: {len(embeddings[0])}")
