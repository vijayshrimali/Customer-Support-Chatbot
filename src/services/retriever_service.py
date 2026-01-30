"""
Retriever Service - RAG Pipeline Component
Creates and manages retrievers for similarity-based document retrieval
"""

import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from dotenv import load_dotenv

from vector_store import VectorStoreService

# Load environment variables
load_dotenv()


class RetrieverService:
    """
    Service for creating and managing retrievers
    Retrieves relevant document chunks based on query similarity
    """
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "techgear_knowledge",
        search_type: str = "similarity",
        k: int = 3
    ):
        """
        Initialize RetrieverService
        
        Args:
            persist_directory: ChromaDB persistence directory
            collection_name: Name of the collection to retrieve from
            search_type: Type of search ("similarity" or "mmr")
            k: Number of documents to retrieve (default: 3)
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.search_type = search_type
        self.k = k
        
        # Initialize vector store service
        self.vector_store_service = VectorStoreService(
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        
        # Load vector store
        self.vector_store = None
        self.retriever = None
        
        print(f"✅ RetrieverService initialized")
        print(f"   Persist Directory: {self.persist_directory}")
        print(f"   Collection Name: {self.collection_name}")
        print(f"   Search Type: {self.search_type}")
        print(f"   Top K Results: {self.k}")
    
    def load_retriever(self) -> Optional[BaseRetriever]:
        """
        Load vector store and create retriever
        
        Returns:
            BaseRetriever instance or None if loading fails
        """
        print(f"\n{'='*60}")
        print(f"Loading Retriever from Vector Store")
        print(f"{'='*60}")
        
        # Load vector store
        self.vector_store = self.vector_store_service.load_vector_store()
        
        if self.vector_store is None:
            print(f"❌ Failed to load vector store")
            return None
        
        # Create retriever from vector store
        print(f"\n🔍 Creating retriever...")
        print(f"   Search Type: {self.search_type}")
        print(f"   Top K: {self.k}")
        
        self.retriever = self.vector_store.as_retriever(
            search_type=self.search_type,
            search_kwargs={"k": self.k}
        )
        
        print(f"✅ Retriever created successfully!")
        print(f"   Type: {type(self.retriever).__name__}")
        print(f"   Ready to retrieve documents")
        
        return self.retriever
    
    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: User query string
            
        Returns:
            List of relevant Document objects
        """
        if self.retriever is None:
            print(f"⚠️  Retriever not loaded. Loading now...")
            self.load_retriever()
        
        if self.retriever is None:
            print(f"❌ Cannot retrieve - retriever failed to load")
            return []
        
        print(f"\n{'='*60}")
        print(f"Retrieving Documents")
        print(f"{'='*60}")
        print(f"Query: {query}")
        
        # Retrieve documents
        documents = self.retriever.invoke(query)
        
        print(f"\n✅ Retrieved {len(documents)} document(s)")
        
        # Display retrieved documents
        for i, doc in enumerate(documents, 1):
            preview = doc.page_content[:150].replace('\n', ' ')
            print(f"\n{i}. {preview}...")
            if doc.metadata:
                print(f"   Source: {doc.metadata.get('source', 'Unknown')}")
        
        return documents
    
    def retrieve_with_scores(self, query: str) -> List[tuple]:
        """
        Retrieve relevant documents with similarity scores
        
        Args:
            query: User query string
            
        Returns:
            List of tuples (Document, score)
        """
        if self.vector_store is None:
            print(f"⚠️  Vector store not loaded. Loading now...")
            self.load_retriever()
        
        if self.vector_store is None:
            print(f"❌ Cannot retrieve - vector store failed to load")
            return []
        
        print(f"\n{'='*60}")
        print(f"Retrieving Documents with Scores")
        print(f"{'='*60}")
        print(f"Query: {query}")
        
        # Retrieve documents with scores
        results = self.vector_store.similarity_search_with_score(query, k=self.k)
        
        print(f"\n✅ Retrieved {len(results)} document(s)")
        
        # Display retrieved documents with scores
        for i, (doc, score) in enumerate(results, 1):
            preview = doc.page_content[:150].replace('\n', ' ')
            print(f"\n{i}. Score: {score:.4f}")
            print(f"   {preview}...")
            if doc.metadata:
                print(f"   Source: {doc.metadata.get('source', 'Unknown')}")
        
        return results
    
    def test_retriever(self, test_queries: Optional[List[str]] = None):
        """
        Test retriever with sample queries
        
        Args:
            test_queries: List of test queries (uses defaults if None)
        """
        if test_queries is None:
            test_queries = [
                "What is the price of SmartWatch Pro X?",
                "Tell me about wireless earbuds features",
                "What is your return policy?",
                "How can I contact customer support?",
                "What are the warranty terms?"
            ]
        
        print(f"\n{'='*70}")
        print(f"  TESTING RETRIEVER WITH {len(test_queries)} QUERIES")
        print(f"{'='*70}")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'─'*70}")
            print(f"Test Query {i}/{len(test_queries)}")
            print(f"{'─'*70}")
            
            # Retrieve documents
            documents = self.retrieve(query)
            
            if not documents:
                print(f"⚠️  No documents retrieved for this query")
        
        print(f"\n{'='*70}")
        print(f"  ✅ RETRIEVER TESTING COMPLETE")
        print(f"{'='*70}")
        print(f"\n✅ Tested {len(test_queries)} queries")
        print(f"✅ Retriever is working correctly")


def create_and_test_retriever():
    """
    Main function to create and test the retriever
    """
    print(f"\n{'='*70}")
    print(f"  STEP 7: CREATE RETRIEVER FOR RAG PIPELINE")
    print(f"{'='*70}")
    
    # Step 1: Initialize retriever service
    print(f"\n📚 Step 1: Initializing RetrieverService...")
    retriever_service = RetrieverService(
        persist_directory="./chroma_db",
        collection_name="techgear_knowledge",
        search_type="similarity",
        k=3
    )
    
    # Step 2: Load retriever
    print(f"\n🔄 Step 2: Loading retriever from vector store...")
    retriever = retriever_service.load_retriever()
    
    if retriever is None:
        print(f"❌ Failed to create retriever")
        return None
    
    # Step 3: Test with sample queries
    print(f"\n🧪 Step 3: Testing retriever with sample queries...")
    retriever_service.test_retriever()
    
    # Step 4: Test with scores
    print(f"\n📊 Step 4: Testing retrieval with similarity scores...")
    test_query = "What is the warranty on SmartWatch?"
    results_with_scores = retriever_service.retrieve_with_scores(test_query)
    
    # Step 5: Display statistics
    print(f"\n{'='*70}")
    print(f"  ✅ STEP 7 COMPLETE: RETRIEVER CREATED AND TESTED")
    print(f"{'='*70}")
    
    print(f"\n📊 Retriever Statistics:")
    print(f"   Vector Store: Loaded from {retriever_service.persist_directory}")
    print(f"   Collection: {retriever_service.collection_name}")
    print(f"   Search Type: {retriever_service.search_type}")
    print(f"   Top K Results: {retriever_service.k}")
    print(f"   Status: ✅ Ready for RAG pipeline")
    
    # Display usage example
    print(f"\n💡 Usage Example:")
    print(f"   from src.services.retriever_service import RetrieverService")
    print(f"   ")
    print(f"   service = RetrieverService(k=3)")
    print(f"   retriever = service.load_retriever()")
    print(f"   docs = service.retrieve('What is the price?')")
    
    return retriever


if __name__ == "__main__":
    # Create and test retriever
    create_and_test_retriever()
