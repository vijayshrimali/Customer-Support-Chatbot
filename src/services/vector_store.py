"""
Vector Store Service - ChromaDB Integration
Store and retrieve embeddings using ChromaDB persistent storage
"""

import os
from typing import List, Optional
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

from embeddings_service import EmbeddingsService

# Load environment variables
load_dotenv()


class VectorStoreService:
    """Service for managing ChromaDB vector store"""
    
    def __init__(self, persist_directory="./chroma_db", collection_name="techgear_knowledge"):
        """Initialize ChromaDB vector store"""
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize embeddings service
        self.embeddings_service = EmbeddingsService()
        self.embeddings_function = self.embeddings_service.get_embeddings_object()
        
        # Create persist directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        print(f"✅ VectorStoreService initialized")
        print(f"   Persist Directory: {self.persist_directory}")
        print(f"   Collection Name: {self.collection_name}")
    
    def create_vector_store(self, documents, reset=False):
        """Create or update ChromaDB vector store with documents"""
        print(f"\n{'='*60}")
        print(f"Creating ChromaDB Vector Store")
        print(f"{'='*60}")
        
        if reset and os.path.exists(self.persist_directory):
            print(f"⚠️  Reset mode: Clearing existing vector store...")
            import shutil
            shutil.rmtree(self.persist_directory)
            os.makedirs(self.persist_directory, exist_ok=True)
        
        print(f"\n📊 Documents to store: {len(documents)}")
        
        # Create vector store from documents
        print(f"🔄 Creating embeddings and storing in ChromaDB...")
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings_function,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )
        
        print(f"✅ Vector store created successfully!")
        print(f"   Total documents stored: {len(documents)}")
        print(f"   Storage location: {self.persist_directory}")
        
        # Get collection stats
        collection = vector_store._collection
        print(f"\n📈 Collection Statistics:")
        print(f"   Collection name: {collection.name}")
        print(f"   Document count: {collection.count()}")
        
        return vector_store
    
    def load_vector_store(self):
        """Load existing ChromaDB vector store from disk"""
        if not os.path.exists(self.persist_directory):
            print(f"❌ Vector store not found at: {self.persist_directory}")
            return None
        
        print(f"\n{'='*60}")
        print(f"Loading ChromaDB Vector Store")
        print(f"{'='*60}")
        
        try:
            vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings_function,
                collection_name=self.collection_name
            )
            
            collection = vector_store._collection
            doc_count = collection.count()
            
            print(f"✅ Vector store loaded successfully!")
            print(f"   Collection name: {collection.name}")
            print(f"   Document count: {doc_count}")
            print(f"   Storage location: {self.persist_directory}")
            
            return vector_store
            
        except Exception as e:
            print(f"❌ Error loading vector store: {e}")
            return None
    
    def similarity_search(self, vector_store, query, k=3):
        """Perform similarity search on vector store"""
        print(f"\n{'='*60}")
        print(f"Similarity Search")
        print(f"{'='*60}")
        print(f"Query: {query}")
        print(f"Top K results: {k}")
        
        results = vector_store.similarity_search(query, k=k)
        
        print(f"\n✅ Found {len(results)} results:")
        for i, doc in enumerate(results, 1):
            preview = doc.page_content[:100].replace('\n', ' ')
            print(f"\n{i}. {preview}...")
            if doc.metadata:
                print(f"   Metadata: {doc.metadata}")
        
        return results


def store_knowledge_base_embeddings():
    """Main function to store knowledge base embeddings in ChromaDB"""
    print(f"\n{'='*70}")
    print(f"  STEP 6: STORE EMBEDDINGS IN CHROMADB")
    print(f"{'='*70}")
    
    # Step 1: Load the knowledge base
    print(f"\n📚 Step 1: Loading knowledge base...")
    from knowledge_loader import load_knowledge_base
    
    # Get absolute path to knowledge base
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    kb_path = os.path.join(project_root, "data", "knowledge_base", "product_info.txt")
    
    documents = load_knowledge_base(kb_path)
    if documents is None:
        print(f"❌ Failed to load knowledge base")
        return None
    print(f"✅ Loaded {len(documents)} document(s)")
    
    # Step 2: Chunk the documents
    print(f"\n✂️  Step 2: Chunking documents...")
    from text_chunker import TextChunker
    
    chunker = TextChunker()
    chunks = chunker.chunk_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")
    
    # Step 3: Create vector store and store embeddings
    print(f"\n💾 Step 3: Creating vector store and storing embeddings...")
    vector_store_service = VectorStoreService()
    
    vector_store = vector_store_service.create_vector_store(
        documents=chunks,
        reset=True
    )
    
    # Step 4: Verify storage
    print(f"\n✅ Step 4: Verifying stored embeddings...")
    collection = vector_store._collection
    
    print(f"\n📊 ChromaDB Collection Details:")
    print(f"   Collection Name: {collection.name}")
    print(f"   Total Documents: {collection.count()}")
    print(f"   Persist Directory: {vector_store_service.persist_directory}")
    
    # Verify we can load it back
    print(f"\n🔄 Verifying persistence...")
    loaded_store = vector_store_service.load_vector_store()
    
    if loaded_store:
        loaded_count = loaded_store._collection.count()
        print(f"✅ Successfully loaded vector store from disk")
        print(f"   Documents in loaded store: {loaded_count}")
        
        # Test similarity search
        print(f"\n🔍 Testing similarity search...")
        test_query = "What is the price of SmartWatch?"
        results = vector_store_service.similarity_search(
            vector_store=loaded_store,
            query=test_query,
            k=3
        )
        
        print(f"\n✅ Similarity search working!")
        print(f"   Query: '{test_query}'")
        print(f"   Results found: {len(results)}")
    
    print(f"\n{'='*70}")
    print(f"  ✅ STEP 6 COMPLETE: EMBEDDINGS STORED IN CHROMADB")
    print(f"{'='*70}")
    print(f"\n📁 Vector store location: ./chroma_db")
    print(f"📊 Total embeddings stored: {collection.count()}")
    print(f"🔄 Persistent: Yes (reusable without re-embedding)")
    
    return vector_store


if __name__ == "__main__":
    store_knowledge_base_embeddings()
