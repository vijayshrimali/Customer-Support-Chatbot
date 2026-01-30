"""
Text Chunking Service
Split documents into overlapping chunks using RecursiveCharacterTextSplitter
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from typing import List
from langchain_core.documents import Document


class TextChunker:
    """Split documents into smaller chunks for better retrieval"""
    
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        """
        Initialize the text chunker
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize RecursiveCharacterTextSplitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        print(f"✅ TextChunker initialized:")
        print(f"   - Chunk Size: {chunk_size} characters")
        print(f"   - Chunk Overlap: {chunk_overlap} characters")
        print(f"   - Separators: ['\\n\\n', '\\n', ' ', '']")
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of Document objects to split
            
        Returns:
            List of chunked Document objects
        """
        print(f"\n⏳ Chunking {len(documents)} document(s)...")
        
        chunks = self.text_splitter.split_documents(documents)
        
        print(f"✅ Created {len(chunks)} chunks")
        
        return chunks
    
    def display_chunk_info(self, chunks: List[Document], num_samples: int = 5):
        """
        Display information about the chunks
        
        Args:
            chunks: List of chunked documents
            num_samples: Number of sample chunks to display
        """
        print("\n" + "=" * 70)
        print("📊 CHUNK STATISTICS")
        print("=" * 70)
        
        # Calculate statistics
        total_chunks = len(chunks)
        chunk_lengths = [len(chunk.page_content) for chunk in chunks]
        avg_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
        min_length = min(chunk_lengths) if chunk_lengths else 0
        max_length = max(chunk_lengths) if chunk_lengths else 0
        
        print(f"\n✅ Total Chunks Created: {total_chunks}")
        print(f"✅ Average Chunk Length: {avg_length:.1f} characters")
        print(f"✅ Minimum Chunk Length: {min_length} characters")
        print(f"✅ Maximum Chunk Length: {max_length} characters")
        print(f"✅ Total Characters: {sum(chunk_lengths):,}")
        
        # Display sample chunks
        print(f"\n" + "=" * 70)
        print(f"📄 SAMPLE CHUNKS (First {min(num_samples, total_chunks)} of {total_chunks})")
        print("=" * 70)
        
        for i, chunk in enumerate(chunks[:num_samples], 1):
            print(f"\n┌─ Chunk {i} " + "─" * 60)
            print(f"│ Length: {len(chunk.page_content)} characters")
            print(f"│ Metadata: {chunk.metadata}")
            print(f"│")
            print(f"│ Content:")
            print("│ " + "─" * 66)
            # Display content with proper indentation
            content_lines = chunk.page_content.split('\n')
            for line in content_lines[:10]:  # Show first 10 lines
                print(f"│ {line}")
            if len(content_lines) > 10:
                print(f"│ ... ({len(content_lines) - 10} more lines)")
            print("└" + "─" * 68)
        
        if total_chunks > num_samples:
            print(f"\n... and {total_chunks - num_samples} more chunks")
        
        # Check for overlap
        if len(chunks) > 1:
            print(f"\n" + "=" * 70)
            print("🔍 OVERLAP VERIFICATION")
            print("=" * 70)
            
            # Check overlap between first two chunks
            chunk1_end = chunks[0].page_content[-self.chunk_overlap:]
            chunk2_start = chunks[1].page_content[:self.chunk_overlap]
            
            if chunk1_end in chunks[1].page_content:
                print(f"\n✅ Overlap detected between Chunk 1 and Chunk 2")
                print(f"   Overlapping text ({self.chunk_overlap} chars):")
                print(f"   '{chunk1_end}'")
            else:
                print(f"\n⚠️  No exact overlap found (may be split at word boundary)")


def load_and_chunk_knowledge_base(
    file_path: str = "data/knowledge_base/product_info.txt",
    chunk_size: int = 300,
    chunk_overlap: int = 50
) -> List[Document]:
    """
    Load knowledge base and split into chunks
    
    Args:
        file_path: Path to the knowledge base file
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunked documents
    """
    print("=" * 70)
    print("🔧 STEP 4: TEXT CHUNKING")
    print("=" * 70)
    
    # Step 1: Load documents
    print("\n📂 Loading knowledge base...")
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} document(s)")
    print(f"   Total content: {len(documents[0].page_content):,} characters")
    
    # Step 2: Initialize chunker
    print("\n🔨 Initializing text chunker...")
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Step 3: Chunk documents
    chunks = chunker.chunk_documents(documents)
    
    # Step 4: Display information
    chunker.display_chunk_info(chunks, num_samples=5)
    
    print("\n" + "=" * 70)
    print("✅ TEXT CHUNKING COMPLETE!")
    print("=" * 70)
    print(f"\n🎯 Ready for Step 5: Create Embeddings")
    print(f"   {len(chunks)} chunks ready to be embedded\n")
    
    return chunks


if __name__ == "__main__":
    # Load and chunk the knowledge base
    chunks = load_and_chunk_knowledge_base(
        file_path="data/knowledge_base/product_info.txt",
        chunk_size=300,
        chunk_overlap=50
    )
    
    print(f"✅ Successfully created {len(chunks)} chunks!")
