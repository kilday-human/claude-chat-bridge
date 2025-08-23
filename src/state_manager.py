#!/usr/bin/env python3
"""
State Manager for Claude-GPT Bridge

Handles context compression, token counting, and state management
to keep conversations within model limits while preserving important
information and citations.
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    logging.warning("tiktoken not available, using approximate token counting")

logger = logging.getLogger(__name__)

@dataclass
class ConversationState:
    """
    Represents the current conversation state with compressed history.
    """
    messages: List[Dict[str, str]] = field(default_factory=list)
    rag_context: str = ""
    citations: List[str] = field(default_factory=list)
    summary: str = ""
    total_tokens: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "messages": self.messages,
            "rag_context": self.rag_context,
            "citations": self.citations,
            "summary": self.summary,
            "total_tokens": self.total_tokens,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat()
        }


class StateManager:
    """
    Manages conversation state and context compression.
    
    Features:
    - Token counting with tiktoken
    - Context compression strategies
    - Message history summarization
    - RAG context management
    - Citation preservation
    """
    
    def __init__(self, 
                 max_tokens: int = 4000,
                 compression_threshold: float = 0.8,
                 model_name: str = "gpt-4"):
        """
        Initialize state manager.
        
        Args:
            max_tokens: Maximum token limit for context
            compression_threshold: Trigger compression at this ratio of max_tokens
            model_name: Model name for token encoding
        """
        self.max_tokens = max_tokens
        self.compression_threshold = compression_threshold
        self.compression_trigger = int(max_tokens * compression_threshold)
        
        # Initialize tokenizer
        if HAS_TIKTOKEN:
            try:
                self.encoding = tiktoken.encoding_for_model(model_name)
            except KeyError:
                logger.warning(f"Model {model_name} not found, using cl100k_base encoding")
                self.encoding = tiktoken.get_encoding("cl100k_base")
        else:
            self.encoding = None
        
        self.state = ConversationState()
        logger.info(f"State manager initialized: max_tokens={max_tokens}, compression at {compression_threshold:.1%}")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Token count
        """
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Rough approximation: 4 characters per token
            return len(text) // 4
    
    def estimate_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Estimate token count for message list.
        
        Args:
            messages: List of messages with 'role' and 'content'
            
        Returns:
            Estimated token count
        """
        total = 0
        for message in messages:
            # Count content tokens
            content_tokens = self.count_tokens(message.get('content', ''))
            # Add overhead for role and formatting (roughly 4 tokens per message)
            total += content_tokens + 4
        
        return total
    
    def compress_message_history(self, messages: List[Dict[str, str]], target_tokens: int) -> Tuple[List[Dict[str, str]], str]:
        """
        Compress message history to fit within token limit.
        
        Args:
            messages: List of messages to compress
            target_tokens: Target token count after compression
            
        Returns:
            Tuple of (compressed_messages, summary_of_removed)
        """
        if not messages:
            return [], ""
        
        # Always keep the first message (system prompt) and last message (current)
        if len(messages) <= 2:
            return messages, ""
        
        # Estimate current token count
        current_tokens = self.estimate_message_tokens(messages)
        
        if current_tokens <= target_tokens:
            return messages, ""
        
        # Strategy: Keep first message, last message, and as many recent messages as possible
        first_message = messages[0]
        last_message = messages[-1]
        middle_messages = messages[1:-1]
        
        # Start with first and last
        kept_messages = [first_message]
        kept_tokens = self.estimate_message_tokens([first_message, last_message])
        
        # Add recent messages working backwards
        for message in reversed(middle_messages):
            message_tokens = self.estimate_message_tokens([message])
            if kept_tokens + message_tokens <= target_tokens - 50:  # Leave some buffer
                kept_messages.insert(-1 if len(kept_messages) > 1 else -1, message)
                kept_tokens += message_tokens
            else:
                break
        
        # Add last message
        kept_messages.append(last_message)
        
        # Create summary of removed messages
        removed_count = len(messages) - len(kept_messages)
        if removed_count > 0:
            summary = f"[{removed_count} earlier messages compressed for context length]"
        else:
            summary = ""
        
        logger.info(f"Compressed {len(messages)} messages to {len(kept_messages)} ({current_tokens} -> {self.estimate_message_tokens(kept_messages)} tokens)")
        
        return kept_messages, summary
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation state.
        
        Args:
            role: Message role (user, assistant, system)
            content: Message content
        """
        message = {"role": role, "content": content}
        self.state.messages.append(message)
        self.state.last_updated = datetime.now()
        
        # Update token count
        self._update_token_count()
        
        # Check if compression is needed
        if self.state.total_tokens > self.compression_trigger:
            self._compress_state()
        
        logger.debug(f"Added message: {role}, tokens: {self.count_tokens(content)}")
    
    def set_rag_context(self, context: str, citations: List[str]) -> None:
        """
        Set RAG context and citations.
        
        Args:
            context: Retrieved context text
            citations: List of citation sources
        """
        self.state.rag_context = context
        self.state.citations = citations
        self.state.last_updated = datetime.now()
        
        # Update token count
        self._update_token_count()
        
        logger.info(f"Set RAG context: {len(context)} chars, {len(citations)} citations")
    
    def get_context_for_model(self, model_max_tokens: int) -> Dict[str, Any]:
        """
        Get optimized context for a specific model.
        
        Args:
            model_max_tokens: Maximum tokens the model can handle
            
        Returns:
            Context dictionary with messages, rag_context, etc.
        """
        # Reserve tokens for response (roughly 25% of limit)
        available_tokens = int(model_max_tokens * 0.75)
        
        # Estimate current usage
        message_tokens = self.estimate_message_tokens(self.state.messages)
        rag_tokens = self.count_tokens(self.state.rag_context)
        citation_tokens = self.count_tokens(str(self.state.citations))
        
        total_estimated = message_tokens + rag_tokens + citation_tokens
        
        # Compress if needed
        if total_estimated > available_tokens:
            # Prioritize: keep RAG context, compress messages if needed
            target_message_tokens = available_tokens - rag_tokens - citation_tokens - 100  # buffer
            
            if target_message_tokens > 0:
                compressed_messages, summary = self.compress_message_history(
                    self.state.messages, 
                    target_message_tokens
                )
            else:
                # Extreme case: compress RAG context too
                compressed_messages = self.state.messages[-2:]  # Keep only recent messages
                rag_context = self._compress_rag_context(available_tokens // 2)
                summary = "[Context heavily compressed due to length limits]"
                
                return {
                    "messages": compressed_messages,
                    "rag_context": rag_context,
                    "citations": self.state.citations,
                    "summary": summary,
                    "estimated_tokens": self.estimate_message_tokens(compressed_messages) + 
                                      self.count_tokens(rag_context)
                }
        else:
            compressed_messages = self.state.messages
            summary = ""
        
        return {
            "messages": compressed_messages,
            "rag_context": self.state.rag_context,
            "citations": self.state.citations,
            "summary": summary,
            "estimated_tokens": self.estimate_message_tokens(compressed_messages) + rag_tokens
        }
    
    def _compress_rag_context(self, target_tokens: int) -> str:
        """
        Compress RAG context to fit token limit.
        
        Args:
            target_tokens: Target token count
            
        Returns:
            Compressed context
        """
        if not self.state.rag_context:
            return ""
        
        current_tokens = self.count_tokens(self.state.rag_context)
        
        if current_tokens <= target_tokens:
            return self.state.rag_context
        
        # Simple compression: take first part of context
        target_chars = int(len(self.state.rag_context) * (target_tokens / current_tokens))
        compressed = self.state.rag_context[:target_chars]
        
        # Try to end at a sentence boundary
        last_period = compressed.rfind('.')
        if last_period > target_chars * 0.8:  # If we can find a period in the last 20%
            compressed = compressed[:last_period + 1]
        
        compressed += "\n[... context truncated for length ...]"
        
        logger.info(f"Compressed RAG context: {current_tokens} -> {self.count_tokens(compressed)} tokens")
        return compressed
    
    def _update_token_count(self) -> None:
        """Update total token count for current state."""
        message_tokens = self.estimate_message_tokens(self.state.messages)
        rag_tokens = self.count_tokens(self.state.rag_context)
        citation_tokens = self.count_tokens(str(self.state.citations))
        
        self.state.total_tokens = message_tokens + rag_tokens + citation_tokens
    
    def _compress_state(self) -> None:
        """Compress current state to reduce token usage."""
        logger.info(f"Compressing state: current tokens {self.state.total_tokens}")
        
        # Compress message history
        target_tokens = int(self.max_tokens * 0.6)  # Use 60% for messages
        compressed_messages, summary = self.compress_message_history(
            self.state.messages, 
            target_tokens
        )
        
        self.state.messages = compressed_messages
        if summary:
            self.state.summary = summary
        
        # Update token count
        self._update_token_count()
        
        logger.info(f"State compressed to {self.state.total_tokens} tokens")
    
    def clear(self) -> None:
        """Clear conversation state."""
        self.state = ConversationState()
        logger.info("Conversation state cleared")
    
    def save_state(self, filepath: str) -> None:
        """
        Save conversation state to file.
        
        Args:
            filepath: Path to save state file
        """
        state_data = self.state.to_dict()
        
        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2)
        
        logger.info(f"State saved to {filepath}")
    
    def load_state(self, filepath: str) -> None:
        """
        Load conversation state from file.
        
        Args:
            filepath: Path to state file
        """
        with open(filepath, 'r') as f:
            state_data = json.load(f)
        
        # Reconstruct state object
        self.state = ConversationState(
            messages=state_data['messages'],
            rag_context=state_data['rag_context'],
            citations=state_data['citations'],
            summary=state_data.get('summary', ''),
            total_tokens=state_data.get('total_tokens', 0),
            created_at=datetime.fromisoformat(state_data['created_at']),
            last_updated=datetime.fromisoformat(state_data['last_updated'])
        )
        
        # Update token count in case of model changes
        self._update_token_count()
        
        logger.info(f"State loaded from {filepath}")
    
    def stats(self) -> Dict[str, Any]:
        """Get state statistics."""
        return {
            "total_messages": len(self.state.messages),
            "total_tokens": self.state.total_tokens,
            "max_tokens": self.max_tokens,
            "token_usage_percent": (self.state.total_tokens / self.max_tokens) * 100,
            "rag_context_length": len(self.state.rag_context),
            "citations_count": len(self.state.citations),
            "has_summary": bool(self.state.summary),
            "created_at": self.state.created_at.isoformat(),
            "last_updated": self.state.last_updated.isoformat()
        }


def optimize_prompt_for_tokens(prompt: str, max_tokens: int) -> str:
    """
    Optimize a prompt to fit within token limits.
    
    Args:
        prompt: Original prompt
        max_tokens: Maximum allowed tokens
        
    Returns:
        Optimized prompt
    """
    # Simple optimization strategies
    if HAS_TIKTOKEN:
        encoding = tiktoken.get_encoding("cl100k_base")
        current_tokens = len(encoding.encode(prompt))
    else:
        current_tokens = len(prompt) // 4
    
    if current_tokens <= max_tokens:
        return prompt
    
    # Strategy 1: Remove extra whitespace
    optimized = re.sub(r'\s+', ' ', prompt.strip())
    
    # Strategy 2: If still too long, truncate intelligently
    if HAS_TIKTOKEN:
        current_tokens = len(encoding.encode(optimized))
    else:
        current_tokens = len(optimized) // 4
    
    if current_tokens > max_tokens:
        # Truncate to roughly the right character count
        target_chars = int(len(optimized) * (max_tokens / current_tokens))
        optimized = optimized[:target_chars]
        
        # Try to end at a sentence
        last_period = optimized.rfind('.')
        if last_period > target_chars * 0.8:
            optimized = optimized[:last_period + 1]
        
        optimized += "..."
    
    return optimized


# CLI for testing state manager
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test State Manager")
    parser.add_argument("--test", action="store_true", help="Run test conversation")
    parser.add_argument("--max-tokens", type=int, default=1000, help="Max tokens for test")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    if args.test:
        # Create test state manager
        sm = StateManager(max_tokens=args.max_tokens)
        
        # Simulate conversation
        sm.add_message("system", "You are a helpful AI assistant.")
        sm.add_message("user", "What is artificial intelligence?")
        sm.add_message("assistant", "Artificial intelligence (AI) refers to computer systems that can perform tasks typically requiring human intelligence, such as learning, reasoning, and problem-solving.")
        
        # Add RAG context
        rag_context = "AI encompasses machine learning, natural language processing, computer vision, and robotics. Modern AI systems use neural networks and deep learning."
        citations = ["https://example.com/ai-overview", "docs/ai_basics.txt"]
        sm.set_rag_context(rag_context, citations)
        
        # Add more messages to trigger compression
        for i in range(10):
            sm.add_message("user", f"Tell me more about AI topic number {i+1}. Please provide detailed explanations with examples and use cases.")
            sm.add_message("assistant", f"Here's information about AI topic {i+1}: " + "This is a detailed response that would contain comprehensive information about the topic. " * 10)
        
        # Show final state
        print("State Manager Test")
        print("=" * 50)
        
        stats = sm.stats()
        print("Statistics:")
        print(json.dumps(stats, indent=2))
        
        print(f"\nFinal message count: {len(sm.state.messages)}")
        print(f"Total tokens: {sm.state.total_tokens}")
        print(f"Compression summary: {sm.state.summary}")
        
        # Test context generation for different model sizes
        for model_size in [2000, 4000, 8000]:
            context = sm.get_context_for_model(model_size)
            print(f"\nContext for {model_size} token model:")
            print(f"  Messages: {len(context['messages'])}")
            print(f"  Estimated tokens: {context['estimated_tokens']}")
            if context['summary']:
                print(f"  Summary: {context['summary']}")
