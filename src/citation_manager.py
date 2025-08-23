#!/usr/bin/env python3
"""
Citation Manager for Claude-GPT Bridge

Handles source tracking, citation formatting, and transparency
for RAG-enhanced responses. Supports multiple citation formats
and automatic source verification.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urlparse
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class Citation:
    """
    Individual citation with source metadata.
    """
    source: str
    title: Optional[str] = None
    url: Optional[str] = None
    file_path: Optional[str] = None
    chunk_id: Optional[str] = None
    similarity: float = 0.0
    accessed_at: datetime = field(default_factory=datetime.now)
    
    @property
    def is_url(self) -> bool:
        """Check if citation is a URL."""
        return bool(self.url or (self.source and self.source.startswith(('http://', 'https://'))))
    
    @property
    def is_file(self) -> bool:
        """Check if citation is a file."""
        return bool(self.file_path or (self.source and Path(self.source).exists()))
    
    @property
    def display_name(self) -> str:
        """Get display name for citation."""
        if self.title:
            return self.title
        elif self.is_url:
            parsed = urlparse(self.source)
            return f"{parsed.netloc}{parsed.path}"
        elif self.is_file:
            return Path(self.source).name
        else:
            return self.source
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert citation to dictionary."""
        return {
            "source": self.source,
            "title": self.title,
            "url": self.url,
            "file_path": self.file_path,
            "chunk_id": self.chunk_id,
            "similarity": self.similarity,
            "accessed_at": self.accessed_at.isoformat(),
            "is_url": self.is_url,
            "is_file": self.is_file,
            "display_name": self.display_name
        }


class CitationManager:
    """
    Manages citations for RAG-enhanced responses.
    
    Features:
    - Automatic citation extraction from RAG results
    - Multiple citation formats (APA, MLA, Chicago, etc.)
    - Deduplication and ranking
    - Source verification and metadata
    """
    
    def __init__(self, max_citations: int = 10, min_similarity: float = 0.3):
        """
        Initialize citation manager.
        
        Args:
            max_citations: Maximum number of citations to track
            min_similarity: Minimum similarity score to include citation
        """
        self.max_citations = max_citations
        self.min_similarity = min_similarity
        self.citations: List[Citation] = []
        self.cited_sources: Set[str] = set()
    
    def add_citation(self, 
                    source: str, 
                    similarity: float = 0.0,
                    chunk_id: Optional[str] = None,
                    title: Optional[str] = None,
                    metadata: Optional[Dict] = None) -> bool:
        """
        Add a citation from RAG result.
        
        Args:
            source: Source identifier (URL, file path, etc.)
            similarity: Similarity score from RAG search
            chunk_id: Chunk identifier
            title: Document title
            metadata: Additional metadata
            
        Returns:
            True if citation was added, False if skipped
        """
        # Check similarity threshold
        if similarity < self.min_similarity:
            logger.debug(f"Skipping citation {source}: similarity {similarity} below threshold")
            return False
        
        # Check for duplicates
        if source in self.cited_sources:
            logger.debug(f"Citation {source} already exists")
            return False
        
        # Check max citations
        if len(self.citations) >= self.max_citations:
            logger.debug(f"Max citations ({self.max_citations}) reached")
            return False
        
        # Create citation object
        citation = Citation(
            source=source,
            similarity=similarity,
            chunk_id=chunk_id,
            title=title
        )
        
        # Extract metadata
        if metadata:
            citation.url = metadata.get('url')
            citation.file_path = metadata.get('file_path')
            if not citation.title:
                citation.title = metadata.get('title') or metadata.get('file_name')
        
        # Add to collections
        self.citations.append(citation)
        self.cited_sources.add(source)
        
        logger.info(f"Added citation: {citation.display_name} (similarity: {similarity:.3f})")
        return True
    
    def add_from_rag_results(self, rag_results: List[Dict[str, Any]]) -> int:
        """
        Add citations from RAG search results.
        
        Args:
            rag_results: List of RAG search results with metadata
            
        Returns:
            Number of citations added
        """
        added_count = 0
        
        for result in rag_results:
            source = result.get('metadata', {}).get('source', result.get('id', 'unknown'))
            similarity = result.get('similarity', 0.0)
            chunk_id = result.get('id')
            metadata = result.get('metadata', {})
            
            if self.add_citation(source, similarity, chunk_id, metadata=metadata):
                added_count += 1
        
        # Sort citations by similarity (highest first)
        self.citations.sort(key=lambda c: c.similarity, reverse=True)
        
        logger.info(f"Added {added_count} citations from RAG results")
        return added_count
    
    def format_citations(self, style: str = "numbered") -> str:
        """
        Format citations in specified style.
        
        Args:
            style: Citation style ('numbered', 'apa', 'mla', 'inline')
            
        Returns:
            Formatted citation string
        """
        if not self.citations:
            return ""
        
        if style == "numbered":
            return self._format_numbered()
        elif style == "apa":
            return self._format_apa()
        elif style == "mla":
            return self._format_mla()
        elif style == "inline":
            return self._format_inline()
        else:
            logger.warning(f"Unknown citation style: {style}, using numbered")
            return self._format_numbered()
    
    def _format_numbered(self) -> str:
        """Format citations as numbered list."""
        lines = ["**Sources:**"]
        
        for i, citation in enumerate(self.citations, 1):
            if citation.is_url:
                line = f"{i}. {citation.display_name} ({citation.source})"
            elif citation.is_file:
                line = f"{i}. {citation.display_name}"
            else:
                line = f"{i}. {citation.source}"
            
            # Add similarity score for transparency
            if citation.similarity > 0:
                line += f" [relevance: {citation.similarity:.1%}]"
            
            lines.append(line)
        
        return "\n".join(lines)
    
    def _format_apa(self) -> str:
        """Format citations in APA style."""
        lines = ["**References:**"]
        
        for citation in self.citations:
            if citation.is_url:
                # Basic URL format (would need more metadata for full APA)
                line = f"Retrieved from {citation.source}"
            elif citation.is_file:
                # File format
                line = f"{citation.display_name}. (Local document)"
            else:
                line = citation.source
            
            lines.append(line)
        
        return "\n".join(lines)
    
    def _format_mla(self) -> str:
        """Format citations in MLA style."""
        lines = ["**Works Cited:**"]
        
        for citation in self.citations:
            if citation.is_url:
                line = f'"{citation.display_name}." Web. {citation.accessed_at.strftime("%d %b %Y")}.'
            elif citation.is_file:
                line = f'"{citation.display_name}." Print.'
            else:
                line = f'"{citation.source}."'
            
            lines.append(line)
        
        return "\n".join(lines)
    
    def _format_inline(self) -> str:
        """Format citations as inline references."""
        if not self.citations:
            return ""
        
        # Create short references
        refs = []
        for i, citation in enumerate(self.citations, 1):
            refs.append(f"[{i}]")
        
        return f"Sources: {', '.join(refs)}"
    
    def get_citation_context(self) -> str:
        """Get context string explaining citations."""
        if not self.citations:
            return ""
        
        total = len(self.citations)
        url_count = sum(1 for c in self.citations if c.is_url)
        file_count = sum(1 for c in self.citations if c.is_file)
        avg_similarity = sum(c.similarity for c in self.citations) / total
        
        context_parts = [
            f"Response based on {total} source{'s' if total != 1 else ''}",
        ]
        
        if url_count > 0:
            context_parts.append(f"{url_count} web source{'s' if url_count != 1 else ''}")
        
        if file_count > 0:
            context_parts.append(f"{file_count} document{'s' if file_count != 1 else ''}")
        
        context_parts.append(f"average relevance: {avg_similarity:.1%}")
        
        return "(" + ", ".join(context_parts) + ")"
    
    def to_json(self) -> List[Dict[str, Any]]:
        """Export citations as JSON-serializable data."""
        return [citation.to_dict() for citation in self.citations]
    
    def clear(self) -> None:
        """Clear all citations."""
        self.citations.clear()
        self.cited_sources.clear()
        logger.debug("Cleared all citations")
    
    def stats(self) -> Dict[str, Any]:
        """Get citation statistics."""
        if not self.citations:
            return {
                "total_citations": 0,
                "url_citations": 0,
                "file_citations": 0,
                "average_similarity": 0.0,
                "similarity_range": [0.0, 0.0]
            }
        
        similarities = [c.similarity for c in self.citations]
        
        return {
            "total_citations": len(self.citations),
            "url_citations": sum(1 for c in self.citations if c.is_url),
            "file_citations": sum(1 for c in self.citations if c.is_file),
            "average_similarity": sum(similarities) / len(similarities),
            "similarity_range": [min(similarities), max(similarities)],
            "sources": [c.source for c in self.citations]
        }


def extract_inline_citations(text: str) -> List[str]:
    """
    Extract inline citation markers from text (e.g., [1], [Smith 2023]).
    
    Args:
        text: Text containing citation markers
        
    Returns:
        List of citation markers found
    """
    # Match patterns like [1], [2-5], [Smith 2023], etc.
    patterns = [
        r'\[(\d+(?:-\d+)?)\]',  # [1], [2-5]
        r'\[([A-Za-z]+\s+\d{4})\]',  # [Smith 2023]
        r'\[([A-Za-z]+\s+et\s+al\.\s+\d{4})\]'  # [Smith et al. 2023]
    ]
    
    citations = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        citations.extend(matches)
    
    return list(set(citations))  # Remove duplicates


def validate_citation_format(citation_text: str, style: str = "numbered") -> bool:
    """
    Validate citation format.
    
    Args:
        citation_text: Citation text to validate
        style: Expected citation style
        
    Returns:
        True if format is valid
    """
    if style == "numbered":
        # Check for numbered list format
        lines = citation_text.split('\n')
        if not lines[0].startswith("**Sources:**"):
            return False
        
        for i, line in enumerate(lines[1:], 1):
            if line.strip() and not line.strip().startswith(f"{i}."):
                return False
        
        return True
    
    # Add validation for other styles as needed
    return True


# CLI for testing citation manager
if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Test Citation Manager")
    parser.add_argument("--test", action="store_true", help="Run test citations")
    parser.add_argument("--style", default="numbered", choices=["numbered", "apa", "mla", "inline"],
                       help="Citation style")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    if args.test:
        # Create test citation manager
        cm = CitationManager()
        
        # Add test citations
        test_citations = [
            {
                "source": "https://openai.com/research/gpt-4",
                "similarity": 0.85,
                "title": "GPT-4 Technical Report",
                "metadata": {"url": "https://openai.com/research/gpt-4"}
            },
            {
                "source": "docs/ai_basics.txt",
                "similarity": 0.72,
                "title": "AI Basics",
                "metadata": {"file_path": "docs/ai_basics.txt", "file_name": "ai_basics.txt"}
            },
            {
                "source": "https://arxiv.org/abs/2005.14165",
                "similarity": 0.68,
                "title": "Language Models are Few-Shot Learners",
                "metadata": {"url": "https://arxiv.org/abs/2005.14165"}
            }
        ]
        
        # Add citations
        for citation in test_citations:
            cm.add_citation(
                source=citation["source"],
                similarity=citation["similarity"],
                title=citation["title"],
                metadata=citation["metadata"]
            )
        
        # Display formatted citations
        print("Citation Manager Test")
        print("=" * 50)
        print(cm.format_citations(args.style))
        print("\n" + "=" * 50)
        print("Context:", cm.get_citation_context())
        print("\nStatistics:")
        stats = cm.stats()
