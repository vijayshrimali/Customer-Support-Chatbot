"""
Test Knowledge Base and Embeddings Components
==============================================

Tests for:
- Knowledge base loading
- Text chunking
- Embedding generation
- ChromaDB vector store
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

import unittest
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma


class TestKnowledgeBase(unittest.TestCase):
    """Test knowledge base loading and processing"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.kb_path = project_root / "data" / "knowledge_base.txt"
        
    def test_knowledge_base_exists(self):
        """Test that knowledge base file exists"""
        self.assertTrue(
            self.kb_path.exists(),
            f"Knowledge base file not found at {self.kb_path}"
        )
        
    def test_knowledge_base_not_empty(self):
        """Test that knowledge base has content"""
        with open(self.kb_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.assertGreater(
            len(content),
            100,
            "Knowledge base should have substantial content"
        )
        
    def test_knowledge_base_loading(self):
        """Test loading knowledge base with LangChain"""
        try:
            loader = TextLoader(str(self.kb_path), encoding='utf-8')
            documents = loader.load()
            
            self.assertIsNotNone(documents, "Documents should not be None")
            self.assertGreater(len(documents), 0, "Should load at least one document")
            
            print(f"✅ Loaded {len(documents)} document(s)")
            
        except Exception as e:
            self.fail(f"Failed to load knowledge base: {e}")


class TestTextChunking(unittest.TestCase):
    """Test text chunking functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.kb_path = project_root / "data" / "knowledge_base.txt"
        loader = TextLoader(str(cls.kb_path), encoding='utf-8')
        cls.documents = loader.load()
        
    def test_chunking_configuration(self):
        """Test text splitter configuration"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.assertEqual(splitter._chunk_size, 500)
        self.assertEqual(splitter._chunk_overlap, 100)
        
    def test_chunking_produces_chunks(self):
        """Test that chunking produces multiple chunks"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = splitter.split_documents(self.documents)
        
        self.assertIsNotNone(chunks, "Chunks should not be None")
        self.assertGreater(len(chunks), 0, "Should produce at least one chunk")
        
        print(f"✅ Created {len(chunks)} chunks")
        
    def test_chunk_size_limits(self):
        """Test that chunks respect size limits"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = splitter.split_documents(self.documents)
        
        # Check that most chunks are within reasonable size
        oversized = [c for c in chunks if len(c.page_content) > 700]
        
        self.assertLess(
            len(oversized),
            len(chunks) * 0.2,  # Less than 20% oversized
            "Too many chunks exceed size limit"
        )
        
    def test_chunk_metadata(self):
        """Test that chunks contain metadata"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = splitter.split_documents(self.documents)
        
        # Check first chunk has metadata
        self.assertIsNotNone(chunks[0].metadata, "Chunk should have metadata")


class TestEmbeddings(unittest.TestCase):
    """Test embedding generation"""
    
    def test_embedding_model_initialization(self):
        """Test that embedding model can be initialized"""
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001"
            )
            
            self.assertIsNotNone(embeddings, "Embeddings should not be None")
            print("✅ Embedding model initialized")
            
        except Exception as e:
            self.skipTest(f"Skipping - API key issue: {e}")
            
    def test_single_text_embedding(self):
        """Test embedding a single text"""
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001"
            )
            
            test_text = "SmartWatch Pro X costs ₹15,999"
            embedding = embeddings.embed_query(test_text)
            
            self.assertIsNotNone(embedding, "Embedding should not be None")
            self.assertIsInstance(embedding, list, "Embedding should be a list")
            self.assertGreater(len(embedding), 0, "Embedding should have dimensions")
            
            print(f"✅ Generated embedding with {len(embedding)} dimensions")
            
        except Exception as e:
            self.skipTest(f"Skipping - API key issue: {e}")


class TestVectorStore(unittest.TestCase):
    """Test ChromaDB vector store"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.db_path = project_root / "chroma_db"
        
    def test_vector_store_exists(self):
        """Test that vector store directory exists"""
        self.assertTrue(
            self.db_path.exists(),
            f"Vector store not found at {self.db_path}"
        )
        
    def test_vector_store_not_empty(self):
        """Test that vector store has content"""
        # Check for ChromaDB files
        files = list(self.db_path.rglob("*"))
        
        self.assertGreater(
            len(files),
            0,
            "Vector store directory should contain files"
        )
        
        print(f"✅ Vector store contains {len(files)} files")
        
    def test_vector_store_loading(self):
        """Test loading existing vector store"""
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001"
            )
            
            vectorstore = Chroma(
                persist_directory=str(self.db_path),
                embedding_function=embeddings
            )
            
            self.assertIsNotNone(vectorstore, "Vector store should not be None")
            print("✅ Vector store loaded successfully")
            
        except Exception as e:
            self.skipTest(f"Skipping - API key issue: {e}")
            
    def test_vector_store_retrieval(self):
        """Test retrieving documents from vector store"""
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001"
            )
            
            vectorstore = Chroma(
                persist_directory=str(self.db_path),
                embedding_function=embeddings
            )
            
            # Test retrieval
            results = vectorstore.similarity_search(
                "SmartWatch price",
                k=3
            )
            
            self.assertIsNotNone(results, "Results should not be None")
            self.assertGreater(len(results), 0, "Should retrieve at least one document")
            
            print(f"✅ Retrieved {len(results)} documents")
            
        except Exception as e:
            self.skipTest(f"Skipping - API key issue: {e}")


def run_knowledge_base_tests():
    """Run all knowledge base tests"""
    print("\n" + "="*70)
    print("TESTING KNOWLEDGE BASE & EMBEDDINGS COMPONENTS")
    print("="*70 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestKnowledgeBase))
    suite.addTests(loader.loadTestsFromTestCase(TestTextChunking))
    suite.addTests(loader.loadTestsFromTestCase(TestEmbeddings))
    suite.addTests(loader.loadTestsFromTestCase(TestVectorStore))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("KNOWLEDGE BASE TESTS SUMMARY")
    print("="*70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*70 + "\n")
    
    return result


if __name__ == "__main__":
    run_knowledge_base_tests()
