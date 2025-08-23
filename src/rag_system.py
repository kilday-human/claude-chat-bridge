#!/usr/bin/env python3
"""
RAG System for Claude-GPT Bridge

Provides document ingestion, embedding storage, and semantic search
for retrieval-augmented generation. Uses ChromaDB for vector storage
and sentence-transformers for embeddings.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import hashlib

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

class RAGSystem:
    """
    Core RAG system for document retrieval and context injection.
    
    Features:
    - Document ingestion with chunking
    - Semantic embeddings with sentence-transformers  
    - ChromaDB vector storage
    - Efficient similarity search
    - Citation tracking
    """
    
    def __init__(self, 
                 db_path: str = "./data/chroma_db",
                 model_name: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):
        """
        Initialize RAG system.
        
        Args:
            db_path: Path to ChromaDB storage
            model_name: Sentence transformer model
            chunk_size: Document chunk size in characters
            chunk_overlap: Overlap between chunks
        """
        self.db_path = Path(db_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Create data directory
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="bridge_knowledge",
            metadata={"description": "Knowledge base for Claude-GPT Bridge"}
        )
        
        logger.info(f"RAG system initialized. Collection has {self.collection.count()} documents.")
    
    def chunk_text(self, text: str, source: str) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks for embedding.
        
        Args:
            text: Full document text
            source: Document source identifier
            
        Returns:
            List of chunk dictionaries with metadata
        """
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                # Create chunk metadata
                chunk_metadata = {
                    "source": source,
                    "chunk_id": chunk_id,
                    "start_pos": start,
                    "end_pos": end,
                    "chunk_size": len(chunk_text)
                }
                
                chunks.append({
                    "id": f"{source}_{chunk_id}",
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })
                chunk_id += 1
            
            start += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def ingest_document(self, content: str, source: str, metadata: Optional[Dict] = None) -> int:
        """
        Ingest a document into the RAG system.
        
        Args:
            content: Document text content
            source: Source identifier (file path, URL, etc.)
            metadata: Additional metadata
            
        Returns:
            Number of chunks created
        """
        logger.info(f"Ingesting document: {source}")
        
        # Create document hash for deduplication
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        
        # Chunk the document
        chunks = self.chunk_text(content, source)
        
        if not chunks:
            logger.warning(f"No chunks created for {source}")
            return 0
        
        # Prepare data for ChromaDB
        ids = [chunk["id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = []
        
        for chunk in chunks:
            chunk_metadata = chunk["metadata"].copy()
            chunk_metadata["content_hash"] = content_hash
            if metadata:
                chunk_metadata.update(metadata)
            metadatas.append(chunk_metadata)
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(documents)} chunks")
        embeddings = self.embedding_model.encode(documents)
        
        # Store in ChromaDB
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )
        
        logger.info(f"Ingested {len(chunks)} chunks from {source}")
        return len(chunks)
    
    def ingest_file(self, file_path: str, metadata: Optional[Dict] = None) -> int:
        """
        Ingest a text file into the RAG system.
        
        Args:
            file_path: Path to text file
            metadata: Additional metadata
            
        Returns:
            Number of chunks created
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read file content
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create metadata
        file_metadata = {
            "file_path": str(path),
            "file_name": path.name,
            "file_size": len(content)
        }
        
        if metadata:
            file_metadata.update(metadata)
        
        return self.ingest_document(content, str(path), file_metadata)
    
    def search(self, query: str, n_results: int = 5, min_similarity: float = 0.3) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using semantic similarity.
        
        Args:
            query: Search query
            n_results: Maximum number of results
            min_similarity: Minimum similarity threshold (0-1)
            
        Returns:
            List of relevant chunks with metadata and similarity scores
        """
        if self.collection.count() == 0:
            logger.warning("No documents in RAG system")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=min(n_results, self.collection.count())
        )
        
        # Format results
        formatted_results = []
        
        if results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                distance = results['distances'][0][i] if results['distances'] else 1.0
                similarity = 1.0 - distance  # Convert distance to similarity
                
                if similarity >= min_similarity:
                    result = {
                        "id": doc_id,
                        "text": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "similarity": similarity,
                        "distance": distance
                    }
                    formatted_results.append(result)
        
        logger.info(f"Found {len(formatted_results)} relevant chunks for query")
        return formatted_results
    
    def get_context(self, query: str, max_tokens: int = 2000) -> Tuple[str, List[str]]:
        """
        Get relevant context and citations for a query.
        
        Args:
            query: Search query
            max_tokens: Maximum tokens for context (rough estimate)
            
        Returns:
            Tuple of (context_text, citation_list)
        """
        # Search for relevant chunks
        results = self.search(query, n_results=10)
        
        if not results:
            return "", []
        
        # Build context and citations
        context_parts = []
        citations = []
        current_tokens = 0
        max_chars = max_tokens * 4  # Rough token-to-char conversion
        
        for result in results:
            chunk_text = result["text"]
            source = result["metadata"].get("source", "unknown")
            
            # Add chunk if we have space
            if current_tokens + len(chunk_text) < max_chars:
                context_parts.append(chunk_text)
                current_tokens += len(chunk_text)
                
                # Add unique citations
                if source not in citations:
                    citations.append(source)
            else:
                break
        
        # Format context
        context = "\n\n".join(context_parts)
        
        return context, citations
    
    def reset(self) -> None:
        """Reset the RAG system (delete all documents)."""
        logger.warning("Resetting RAG system - all documents will be deleted")
        self.client.reset()
        self.collection = self.client.get_or_create_collection(
            name="bridge_knowledge",
            metadata={"description": "Knowledge base for Claude-GPT Bridge"}
        )
    
    def stats(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        count = self.collection.count()
        
        # Get unique sources
        if count > 0:
            all_metadata = self.collection.get()['metadatas']
            sources = set()
            total_chunks = 0
            
            for metadata in all_metadata:
                if metadata:
                    sources.add(metadata.get('source', 'unknown'))
                    total_chunks += 1
        else:
            sources = set()
            total_chunks = 0
        
        return {
            "total_chunks": total_chunks,
            "unique_sources": len(sources),
            "sources": list(sources),
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_size": self.chunk_size,
            "db_path": str(self.db_path)
        }


def create_knowledge_base(docs_dir: str = "docs/knowledge") -> RAGSystem:
    """
    Create and populate RAG system from a documents directory.
    
    Args:
        docs_dir: Directory containing text files to ingest
        
    Returns:
        Initialized RAG system
    """
    rag = RAGSystem()
    docs_path = Path(docs_dir)
    
    if not docs_path.exists():
        logger.info(f"Creating knowledge directory: {docs_path}")
        docs_path.mkdir(parents=True, exist_ok=True)
        
        # Create sample documents
        sample_docs = {
            "ai_basics.txt": """
Artificial Intelligence (AI) refers to computer systems that can perform tasks typically requiring human intelligence.
Machine Learning is a subset of AI that enables systems to learn from data without explicit programming.
Large Language Models (LLMs) like GPT and Claude are AI systems trained on vast amounts of text data.
Natural Language Processing (NLP) is the branch of AI focused on understanding and generating human language.
""",
            "bridge_system.txt": """
The Claude-GPT Bridge is an AI orchestration system that routes queries between different language models.
It uses intelligent routing to select appropriate models based on query complexity and requirements.
The system includes cost tracking, performance monitoring, and citation management for transparency.
RAG (Retrieval-Augmented Generation) enhances responses by incorporating relevant external knowledge.
"""
        }
        
        for filename, content in sample_docs.items():
            sample_path = docs_path / filename
            with open(sample_path, 'w') as f:
                f.write(content.strip())
            logger.info(f"Created sample document: {sample_path}")
    
    # Ingest all text files
    ingested_count = 0
    for file_path in docs_path.glob("*.txt"):
        try:
            count = rag.ingest_file(str(file_path))
            ingested_count += count
            logger.info(f"Ingested {count} chunks from {file_path.name}")
        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {e}")
    
    logger.info(f"Knowledge base created with {ingested_count} total chunks")
    return rag


# CLI for testing RAG system
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test RAG System")
    parser.add_argument("--reset", action="store_true", help="Reset knowledge base")
    parser.add_argument("--ingest", type=str, help="Ingest a file")
    parser.add_argument("--search", type=str, help="Search query")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize RAG system
    rag = RAGSystem()
    
    if args.reset:
        rag.reset()
        print("RAG system reset")
    
    if args.ingest:
        count = rag.ingest_file(args.ingest)
        print(f"Ingested {count} chunks from {args.ingest}")
    
    if args.search:
        results = rag.search(args.search)
        print(f"\nSearch results for: {args.search}")
        print("-" * 50)
        for result in results:
            print(f"Similarity: {result['similarity']:.3f}")
            print(f"Source: {result['metadata']['source']}")
            print(f"Text: {result['text'][:200]}...")
            print()
    
    if args.stats:
        stats = rag.stats()
        print("RAG System Statistics:")
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Unique sources: {stats['unique_sources']}")
        print(f"Embedding model: {stats['embedding_model']}")
        print(f"Sources: {stats['sources']}")
