"""
RAG Chain - Retrieval Augmented Generation Pipeline
Combines Google Gemini LLM with ChromaDB retriever for context-aware responses
"""

import os
from typing import Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

import sys
# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
services_dir = os.path.join(parent_dir, 'services')
sys.path.insert(0, services_dir)

from retriever_service import RetrieverService

# Load environment variables
load_dotenv()


class RAGChain:
    """
    Retrieval Augmented Generation Chain
    Uses ChromaDB retriever + Google Gemini LLM for context-aware question answering
    """
    
    def __init__(
        self,
        model_name: str = None,
        temperature: float = 0.3,
        top_k: int = 3
    ):
        """
        Initialize RAG Chain
        
        Args:
            model_name: Google Gemini model name
            temperature: LLM temperature (0.0-1.0)
            top_k: Number of documents to retrieve
        """
        # Get model name from env if not provided
        if model_name is None:
            model_name = os.getenv("MODEL_NAME", "gemini-2.0-flash")
        
        self.model_name = model_name
        self.temperature = temperature
        self.top_k = top_k
        
        # Get API key
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Initialize components
        self.llm = None
        self.retriever_service = None
        self.retriever = None
        self.rag_chain = None
        
        print(f"✅ RAGChain initialized")
        print(f"   Model: {self.model_name}")
        print(f"   Temperature: {self.temperature}")
        print(f"   Top K: {self.top_k}")
    
    def initialize_llm(self):
        """Initialize Google Gemini LLM"""
        print(f"\n{'='*60}")
        print(f"Initializing Google Gemini LLM")
        print(f"{'='*60}")
        
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=self.temperature,
            convert_system_message_to_human=True
        )
        
        print(f"✅ LLM initialized successfully!")
        print(f"   Model: {self.model_name}")
        print(f"   Temperature: {self.temperature}")
        print(f"   API Key: {self.api_key[:15]}...{self.api_key[-4:]}")
    
    def initialize_retriever(self):
        """Initialize ChromaDB retriever"""
        print(f"\n{'='*60}")
        print(f"Initializing ChromaDB Retriever")
        print(f"{'='*60}")
        
        self.retriever_service = RetrieverService(k=self.top_k)
        self.retriever = self.retriever_service.load_retriever()
        
        if self.retriever is None:
            raise ValueError("Failed to load retriever")
        
        print(f"✅ Retriever initialized successfully!")
        print(f"   Top K: {self.top_k}")
    
    def create_prompt_template(self) -> ChatPromptTemplate:
        """
        Create RAG prompt template
        
        Returns:
            ChatPromptTemplate for RAG
        """
        # System message
        system_message = """You are a helpful customer service assistant for TechGear Electronics.

Your role is to answer customer questions about our products, policies, and services.

IMPORTANT INSTRUCTIONS:
1. Answer ONLY based on the provided context
2. If the context doesn't contain the answer, say "I don't have that information in my knowledge base"
3. Be concise, friendly, and professional
4. Include specific details like prices, features, and policies when available
5. Do NOT make up or infer information not present in the context
6. If asked about products not in the context, politely inform the customer

Context:
{context}

Customer Question: {question}

Your Response:"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("human", system_message)
        ])
        
        return prompt
    
    def format_docs(self, docs) -> str:
        """
        Format retrieved documents into context string
        
        Args:
            docs: List of Document objects
            
        Returns:
            Formatted context string
        """
        if not docs:
            return "No relevant information found."
        
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"[Source {i}]\n{doc.page_content}")
        
        return "\n\n".join(context_parts)
    
    def build_chain(self):
        """Build the complete RAG chain"""
        print(f"\n{'='*60}")
        print(f"Building RAG Chain")
        print(f"{'='*60}")
        
        # Initialize components if not already done
        if self.llm is None:
            self.initialize_llm()
        
        if self.retriever is None:
            self.initialize_retriever()
        
        # Create prompt template
        prompt = self.create_prompt_template()
        
        # Build RAG chain
        self.rag_chain = (
            {
                "context": self.retriever | self.format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        print(f"✅ RAG Chain built successfully!")
        print(f"\n📊 Chain Components:")
        print(f"   1. Retriever → Fetch relevant documents")
        print(f"   2. Format Docs → Create context string")
        print(f"   3. Prompt Template → Format question + context")
        print(f"   4. LLM (Gemini) → Generate response")
        print(f"   5. Output Parser → Extract text response")
    
    def query(self, question: str, verbose: bool = True) -> str:
        """
        Query the RAG chain
        
        Args:
            question: User question
            verbose: Print detailed information
            
        Returns:
            Generated response
        """
        if self.rag_chain is None:
            self.build_chain()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"RAG Query")
            print(f"{'='*60}")
            print(f"Question: {question}")
            print(f"\n🔍 Retrieving relevant context...")
        
        # Retrieve documents (for display purposes)
        if verbose:
            docs = self.retriever.invoke(question)
            print(f"✅ Retrieved {len(docs)} document(s)")
            for i, doc in enumerate(docs, 1):
                preview = doc.page_content[:100].replace('\n', ' ')
                print(f"   {i}. {preview}...")
        
        # Generate response
        if verbose:
            print(f"\n🤖 Generating response with Gemini...")
        
        response = self.rag_chain.invoke(question)
        
        if verbose:
            print(f"\n✅ Response generated!")
            print(f"\n{'─'*60}")
            print(f"Response:")
            print(f"{'─'*60}")
            print(response)
            print(f"{'─'*60}")
        
        return response
    
    def test_rag_chain(self):
        """Test RAG chain with sample queries"""
        print(f"\n{'='*70}")
        print(f"  TESTING RAG CHAIN")
        print(f"{'='*70}")
        
        test_queries = [
            "What is the price of SmartWatch Pro X?",
            "What features does the Wireless Earbuds Elite have?",
            "What is your return policy?",
            "How can I contact customer support?",
            "Tell me about the warranty on Power Bank Ultra"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*70}")
            print(f"Test Query {i}/{len(test_queries)}")
            print(f"{'='*70}")
            
            response = self.query(query, verbose=True)
        
        print(f"\n{'='*70}")
        print(f"  ✅ RAG CHAIN TESTING COMPLETE")
        print(f"{'='*70}")


def create_and_test_rag_chain():
    """
    Main function to create and test RAG chain
    """
    print(f"\n{'='*70}")
    print(f"  STEP 8: BUILD RAG CHAIN WITH GOOGLE GEMINI")
    print(f"{'='*70}")
    
    # Step 1: Initialize RAG Chain
    print(f"\n📚 Step 1: Initializing RAG Chain...")
    rag_chain = RAGChain(
        model_name=None,  # Will use MODEL_NAME from .env
        temperature=0.3,
        top_k=3
    )
    
    # Step 2: Build the chain
    print(f"\n🔨 Step 2: Building RAG pipeline...")
    rag_chain.build_chain()
    
    # Step 3: Test with sample queries
    print(f"\n🧪 Step 3: Testing RAG chain...")
    rag_chain.test_rag_chain()
    
    # Step 4: Test strict context adherence
    print(f"\n{'='*70}")
    print(f"  TESTING STRICT CONTEXT ADHERENCE")
    print(f"{'='*70}")
    
    # Test with question outside knowledge base
    print(f"\n🧪 Test: Question OUTSIDE knowledge base")
    out_of_scope_query = "What is the price of iPhone 15?"
    response = rag_chain.query(out_of_scope_query, verbose=True)
    
    # Step 5: Display statistics
    print(f"\n{'='*70}")
    print(f"  ✅ STEP 8 COMPLETE: RAG CHAIN READY")
    print(f"{'='*70}")
    
    print(f"\n📊 RAG Chain Statistics:")
    print(f"   Model: {rag_chain.model_name}")
    print(f"   Temperature: {rag_chain.temperature}")
    print(f"   Retriever Top K: {rag_chain.top_k}")
    print(f"   Status: ✅ Ready for production use")
    
    # Display usage example
    print(f"\n💡 Usage Example:")
    print(f"   from src.bot.rag_chain import RAGChain")
    print(f"   ")
    print(f"   rag = RAGChain()")
    print(f"   rag.build_chain()")
    print(f"   response = rag.query('What is the price of SmartWatch?')")
    print(f"   print(response)")
    
    return rag_chain


if __name__ == "__main__":
    # Create and test RAG chain
    create_and_test_rag_chain()
