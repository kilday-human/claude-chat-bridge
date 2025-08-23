#!/usr/bin/env python3
"""
RAG Integration for Claude-GPT Bridge

Integrates RAG system with existing Bridge architecture.
Handles router decisions for RAG usage, context injection,
and citation management in Bridge responses.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

from src.rag_system import RAGSystem, create_knowledge_base
from src.citation_manager import CitationManager
from src.state_manager import StateManager

logger = logging.getLogger(__name__)

class RAGBridge:
    """
    Integrates RAG capabilities with Bridge system.
    
    Features:
    - Automatic RAG triggering based on query analysis
    - Context injection for both GPT and Claude
    - Citation management and formatting
    - State management for conversation context
    """
    
    def __init__(self, 
                 knowledge_dir: str = "docs/knowledge",
                 max_context_tokens: int = 2000,
                 citation_style: str = "numbered"):
        """
        Initialize RAG Bridge integration.
        
        Args:
            knowledge_dir: Directory containing knowledge documents
            max_context_tokens: Maximum tokens for RAG context
            citation_style: Citation formatting style
        """
        self.max_context_tokens = max_context_tokens
        self.citation_style = citation_style
        
        # Initialize components
        logger.info("Initializing RAG Bridge components...")
        
        self.rag_system = self._initialize_rag_system(knowledge_dir)
        self.citation_manager = CitationManager()
        self.state_manager = StateManager(max_tokens=max_context_tokens * 2)
        
        logger.info("RAG Bridge initialized successfully")
    
    def _initialize_rag_system(self, knowledge_dir: str) -> RAGSystem:
        """Initialize or load existing RAG system."""
        try:
            # Try to create/load knowledge base
            rag = create_knowledge_base(knowledge_dir)
            stats = rag.stats()
            
            if stats["total_chunks"] == 0:
                logger.warning(f"No documents found in {knowledge_dir}")
                logger.info("RAG system ready but empty - add documents to enable retrieval")
            else:
                logger.info(f"RAG system loaded: {stats['total_chunks']} chunks from {stats['unique_sources']} sources")
            
            return rag
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            # Return empty RAG system
            return RAGSystem()
    
    def should_use_rag(self, prompt: str, metadata: Optional[Dict] = None) -> bool:
        """
        Determine if RAG should be used for this prompt.
        
        Args:
            prompt: User prompt
            metadata: Additional metadata for decision
            
        Returns:
            True if RAG should be used
        """
        # RAG usage criteria
        prompt_lower = prompt.lower()
        
        # Skip RAG for very short prompts
        if len(prompt.split()) < 3:
            return False
        
        # Use RAG for informational queries
        info_keywords = [
            'what', 'how', 'why', 'when', 'where', 'explain', 'describe',
            'tell me about', 'information about', 'details on', 'overview of'
        ]
        
        # Use RAG for specific domains (if we have relevant docs)
        domain_keywords = [
            'ai', 'artificial intelligence', 'machine learning', 'ml',
            'neural network', 'deep learning', 'nlp', 'language model',
            'bridge', 'router', 'api', 'documentation'
        ]
        
        # Check for information-seeking patterns
        has_info_pattern = any(keyword in prompt_lower for keyword in info_keywords)
        has_domain_content = any(keyword in prompt_lower for keyword in domain_keywords)
        
        # Use RAG if we have informational query or domain-relevant content
        should_use = has_info_pattern or has_domain_content
        
        logger.debug(f"RAG decision for '{prompt[:50]}...': {should_use}")
        return should_use
    
    def enhance_prompt(self, prompt: str) -> Tuple[str, List[str]]:
        """
        Enhance prompt with RAG context and return citations.
        
        Args:
            prompt: Original user prompt
            
        Returns:
            Tuple of (enhanced_prompt, citations)
        """
        # Clear previous citations
        self.citation_manager.clear()
        
        # Check if we should use RAG
        if not self.should_use_rag(prompt):
            logger.debug("Skipping RAG enhancement")
            return prompt, []
        
        # Get RAG context
        try:
            context, citations = self.rag_system.get_context(
                prompt, 
                max_tokens=self.max_context_tokens
            )
            
            if not context:
                logger.info("No relevant context found in knowledge base")
                return prompt, []
            
            # Add citations to manager
            rag_results = self.rag_system.search(prompt, n_results=10)
            self.citation_manager.add_from_rag_results(rag_results)
            
            # Store context in state manager
            self.state_manager.set_rag_context(context, citations)
            
            # Enhance prompt with context
            enhanced_prompt = self._format_enhanced_prompt(prompt, context)
            
            logger.info(f"Enhanced prompt with {len(context)} chars of context, {len(citations)} citations")
            return enhanced_prompt, citations
            
        except Exception as e:
            logger.error(f"RAG enhancement failed: {e}")
            return prompt, []
    
    def _format_enhanced_prompt(self, original_prompt: str, context: str) -> str:
        """Format prompt with RAG context."""
        template = """Context information:
{context}

Based on the above context, please answer the following question:
{prompt}

Please cite relevant sources in your response when applicable."""
        
        return template.format(context=context, prompt=original_prompt)
    
    def format_response_with_citations(self, response: str) -> str:
        """
        Add citations to model response.
        
        Args:
            response: Raw model response
            
        Returns:
            Response with formatted citations
        """
        if not self.citation_manager.citations:
            return response
        
        # Add citation context and formatted citations
        citation_context = self.citation_manager.get_citation_context()
        formatted_citations = self.citation_manager.format_citations(self.citation_style)
        
        enhanced_response = f"{response}\n\n{citation_context}\n\n{formatted_citations}"
        
        logger.debug(f"Added {len(self.citation_manager.citations)} citations to response")
        return enhanced_response
    
    def process_bridge_request(self, 
                             prompt: str, 
                             model_type: str = "gpt",
                             max_tokens: int = 512) -> Dict[str, Any]:
        """
        Process a complete bridge request with RAG enhancement.
        
        Args:
            prompt: User prompt
            model_type: Model type (gpt, claude)
            max_tokens: Maximum response tokens
            
        Returns:
            Dictionary with enhanced prompt, context info, and metadata
        """
        # Add user message to state
        self.state_manager.add_message("user", prompt)
        
        # Enhance prompt with RAG
        enhanced_prompt, citations = self.enhance_prompt(prompt)
        
        # Get optimized context for the model
        context_info = self.state_manager.get_context_for_model(
            model_max_tokens=max_tokens * 8  # Assume context can be larger than response
        )
        
        # Prepare result
        result = {
            "original_prompt": prompt,
            "enhanced_prompt": enhanced_prompt,
            "rag_used": enhanced_prompt != prompt,
            "citations": citations,
            "context_info": context_info,
            "rag_stats": {
                "context_length": len(context_info["rag_context"]),
                "citation_count": len(citations),
                "estimated_tokens": context_info["estimated_tokens"]
            }
        }
        
        logger.info(f"Processed bridge request: RAG={'yes' if result['rag_used'] else 'no'}, "
                   f"citations={len(citations)}, tokens={context_info['estimated_tokens']}")
        
        return result
    
    def finalize_response(self, response: str, model_type: str) -> str:
        """
        Finalize response with citations and state management.
        
        Args:
            response: Raw model response
            model_type: Model that generated the response
            
        Returns:
            Final formatted response
        """
        # Add assistant response to state
        self.state_manager.add_message("assistant", response)
        
        # Add citations if available
        final_response = self.format_response_with_citations(response)
        
        return final_response
    
    def get_rag_stats(self) -> Dict[str, Any]:
        """Get comprehensive RAG system statistics."""
        rag_stats = self.rag_system.stats()
        citation_stats = self.citation_manager.stats()
        state_stats = self.state_manager.stats()
        
        return {
            "rag_system": rag_stats,
            "citations": citation_stats,
            "state": state_stats,
            "integration": {
                "max_context_tokens": self.max_context_tokens,
                "citation_style": self.citation_style
            }
        }
    
    def add_knowledge_source(self, content: str, source: str, metadata: Optional[Dict] = None) -> int:
        """
        Add a knowledge source to the RAG system.
        
        Args:
            content: Document content
            source: Source identifier
            metadata: Additional metadata
            
        Returns:
            Number of chunks created
        """
        return self.rag_system.ingest_document(content, source, metadata)
    
    def reset_conversation(self) -> None:
        """Reset conversation state but preserve knowledge base."""
        self.state_manager.clear()
        self.citation_manager.clear()
        logger.info("Conversation state reset")
    
    def reset_knowledge_base(self) -> None:
        """Reset the entire knowledge base."""
        self.rag_system.reset()
        self.citation_manager.clear()
        self.state_manager.clear()
        logger.warning("Knowledge base and conversation state reset")


# Router integration functions
def should_use_rag_for_query(prompt: str, complexity_score: int = 0) -> bool:
    """
    Router helper to determine if RAG should be used.
    
    Args:
        prompt: User prompt
        complexity_score: Complexity score from router
        
    Returns:
        True if RAG should be used
    """
    # Use existing RAG bridge logic
    rag_bridge = RAGBridge()
    return rag_bridge.should_use_rag(prompt)


def create_rag_enhanced_bridge() -> RAGBridge:
    """Factory function to create RAG-enhanced bridge."""
    return RAGBridge()


# CLI for testing RAG integration
if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Test RAG Integration")
    parser.add_argument("--test", action="store_true", help="Run test queries")
    parser.add_argument("--query", type=str, help="Test specific query")
    parser.add_argument("--stats", action="store_true", help="Show RAG stats")
    parser.add_argument("--reset", action="store_true", help="Reset knowledge base")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize RAG Bridge
    rag_bridge = RAGBridge()
    
    if args.reset:
        rag_bridge.reset_knowledge_base()
        print("Knowledge base reset")
    
    if args.stats:
        stats = rag_bridge.get_rag_stats()
        print("RAG Integration Statistics:")
        print(json.dumps(stats, indent=2))
    
    if args.query:
        print(f"Processing query: {args.query}")
        print("-" * 50)
        
        # Process the query
        result = rag_bridge.process_bridge_request(args.query)
        
        print(f"RAG used: {result['rag_used']}")
        print(f"Citations: {len(result['citations'])}")
        print(f"Context tokens: {result['rag_stats']['estimated_tokens']}")
        
        if result['rag_used']:
            print("\nEnhanced prompt:")
            print(result['enhanced_prompt'][:500] + "..." if len(result['enhanced_prompt']) > 500 else result['enhanced_prompt'])
    
    if args.test:
        # Run test queries
        test_queries = [
            "Hello world",  # Should not use RAG
            "What is artificial intelligence?",  # Should use RAG
            "Explain machine learning",  # Should use RAG
            "How does the bridge system work?",  # Should use RAG
            "2+2=?",  # Should not use RAG
        ]
        
        print("RAG Integration Test")
        print("=" * 50)
        
        for query in test_queries:
            result = rag_bridge.process_bridge_request(query)
            print(f"Query: {query}")
            print(f"  RAG used: {'✓' if result['rag_used'] else '✗'}")
            print(f"  Citations: {len(result['citations'])}")
            print(f"  Context tokens: {result['rag_stats']['estimated_tokens']}")
            print()
        
        # Show final stats
        stats = rag_bridge.get_rag_stats()
        print("Final Statistics:")
        print(f"Knowledge chunks: {stats['rag_system']['total_chunks']}")
        print(f"Conversation messages: {stats['state']['total_messages']}")
        print(f"Total citations: {stats['citations']['total_citations']}")
