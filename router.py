#!/usr/bin/env python3
"""
router.py — Production-ready heuristic router for multi-model orchestration.

Smart routing strategy:
- Maintains your simple, reliable length-based foundation
- Adds targeted complexity detection for edge cases
- Includes logging and debugging capabilities
- Designed for production reliability and explainability
"""
import re
import logging
from typing import Tuple, Dict, Any

# Configure logging for router decisions
logger = logging.getLogger(__name__)

# Router configuration constants
CONFIG = {
    "length_threshold": 200,
    "word_count_threshold": 40,
    "complexity_boost_threshold": 100,  # Lower threshold if complexity detected
}

def detect_complexity_signals(prompt: str) -> Dict[str, Any]:
    """
    Detect signals that indicate prompt complexity beyond just length.
    Returns dict with detected signals and confidence scores.
    """
    signals = {
        "math_symbols": bool(re.search(r'[+\-*/=^%√∫∑]', prompt)),
        "code_indicators": any(word in prompt.lower() for word in 
                             ['function', 'def ', 'class ', 'import ', 'return ', 'if ', 'for ', 'while ']),
        "analysis_keywords": any(word in prompt.lower() for word in 
                               ['analyze', 'compare', 'evaluate', 'assess', 'explain', 'detailed']),
        "creative_keywords": any(word in prompt.lower() for word in 
                               ['story', 'poem', 'haiku', 'creative', 'write', 'compose']),
        "multiple_questions": prompt.count('?') > 1,
        "bullet_points": prompt.count('\n-') > 2 or prompt.count('•') > 2,
        "word_count": len(prompt.split()),
        "char_count": len(prompt),
    }
    
    # Calculate complexity score (0-10 scale)
    score = 0
    if signals["math_symbols"]: score += 2
    if signals["code_indicators"]: score += 2
    if signals["analysis_keywords"]: score += 2  # Analysis tasks deserve strong models
    if signals["multiple_questions"]: score += 1
    if signals["bullet_points"]: score += 1
    if signals["word_count"] > 50: score += 2
    if signals["char_count"] > 300: score += 1
    
    signals["complexity_score"] = min(score, 10)
    return signals


def choose_model(prompt: str,
                 cheap_gpt: str,
                 strong_gpt: str, 
                 cheap_claude: str,
                 strong_claude: str) -> Tuple[str, str]:
    """
    Production heuristic router combining length-based logic with complexity detection.
    
    Strategy:
    1. Start with your proven length threshold (200 chars)
    2. Add complexity boost for shorter prompts with clear complexity signals
    3. Maintain predictable, explainable decisions
    
    Args:
        prompt: Input text to analyze
        cheap_gpt, strong_gpt, cheap_claude, strong_claude: Model identifiers
        
    Returns:
        Tuple of (selected_gpt_model, selected_claude_model)
    """
    
    # Analyze prompt characteristics
    signals = detect_complexity_signals(prompt)
    char_count = signals["char_count"]
    complexity_score = signals["complexity_score"]
    
    # Decision logic: enhanced length-based with complexity boost
    use_strong = False
    reason = ""
    
    # Primary decision: length threshold
    if char_count >= CONFIG["length_threshold"]:
        use_strong = True
        reason = f"length ({char_count} chars >= {CONFIG['length_threshold']})"
    
    # Complexity boost: shorter prompts that show clear complexity
    elif complexity_score >= 2 and char_count >= 20:  # Math symbols alone = score 2
        use_strong = True
        complexity_reasons = []
        if signals["math_symbols"]: complexity_reasons.append("math")
        if signals["code_indicators"]: complexity_reasons.append("code")
        if signals["analysis_keywords"]: complexity_reasons.append("analysis")
        if signals["multiple_questions"]: complexity_reasons.append("multi-part")
        
        reason = f"complexity boost (score: {complexity_score}, signals: {', '.join(complexity_reasons)})"
    
    # Default: use cheap models
    else:
        reason = f"simple prompt (length: {char_count}, complexity: {complexity_score})"
    
    # Log decision for debugging/monitoring
    logger.debug(f"Router decision: {'strong' if use_strong else 'cheap'} models - {reason}")
    
    # Return model selection
    if use_strong:
        return strong_gpt, strong_claude
    else:
        return cheap_gpt, cheap_claude


def get_routing_explanation(prompt: str, 
                           chosen_gpt: str, 
                           chosen_claude: str,
                           cheap_gpt: str,
                           strong_gpt: str,
                           cheap_claude: str, 
                           strong_claude: str) -> str:
    """
    Generate human-readable explanation of routing decision for CLI output.
    """
    signals = detect_complexity_signals(prompt)
    char_count = signals["char_count"]
    complexity_score = signals["complexity_score"]
    
    # Determine if strong models were chosen
    using_strong = (chosen_gpt == strong_gpt and chosen_claude == strong_claude)
    
    if using_strong:
        if char_count >= CONFIG["length_threshold"]:
            return f"long prompt ({char_count} chars) → strong models"
        else:
            complexity_reasons = []
            if signals["math_symbols"]: complexity_reasons.append("math")
            if signals["code_indicators"]: complexity_reasons.append("code")  
            if signals["analysis_keywords"]: complexity_reasons.append("analysis")
            if signals["multiple_questions"]: complexity_reasons.append("multi-part")
            
            return f"complex prompt (score: {complexity_score}) → strong models [{', '.join(complexity_reasons)}]"
    else:
        return f"simple prompt ({char_count} chars, score: {complexity_score}) → cheap models"


def configure_router(length_threshold: int = None,
                    word_count_threshold: int = None,
                    complexity_boost_threshold: int = None) -> Dict[str, int]:
    """
    Update router configuration for tuning in production.
    Returns current configuration.
    """
    if length_threshold is not None:
        CONFIG["length_threshold"] = length_threshold
    if word_count_threshold is not None:
        CONFIG["word_count_threshold"] = word_count_threshold  
    if complexity_boost_threshold is not None:
        CONFIG["complexity_boost_threshold"] = complexity_boost_threshold
        
    logger.info(f"Router configured: {CONFIG}")
    return CONFIG.copy()


def router_stats(prompts: list) -> Dict[str, Any]:
    """
    Analyze a batch of prompts and return routing statistics.
    Useful for tuning and monitoring router performance.
    """
    if not prompts:
        return {}
        
    stats = {
        "total_prompts": len(prompts),
        "would_use_strong": 0,
        "would_use_cheap": 0,
        "avg_length": sum(len(p) for p in prompts) / len(prompts),
        "avg_complexity": 0,
        "length_triggers": 0,
        "complexity_triggers": 0,
    }
    
    complexity_scores = []
    
    for prompt in prompts:
        signals = detect_complexity_signals(prompt)
        complexity_scores.append(signals["complexity_score"])
        
        char_count = len(prompt)
        
        # Simulate routing decision
        if char_count >= CONFIG["length_threshold"]:
            stats["would_use_strong"] += 1
            stats["length_triggers"] += 1
        elif signals["complexity_score"] >= 2 and char_count >= 20:
            stats["would_use_strong"] += 1
            stats["complexity_triggers"] += 1
        else:
            stats["would_use_cheap"] += 1
    
    stats["avg_complexity"] = sum(complexity_scores) / len(complexity_scores)
    stats["strong_model_rate"] = stats["would_use_strong"] / stats["total_prompts"]
    
    return stats


# For debugging and CLI integration
if __name__ == "__main__":
    # Test examples
    test_prompts = [
        "Hello world",
        "What is 2+2?",
        "Write a detailed analysis of quantum computing's impact on cybersecurity, including technical implementation challenges and potential mitigation strategies for current encryption methods.",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\nExplain this code and optimize it.",
        "Compare Python vs JavaScript for web development. What are the pros and cons of each? Which would you recommend for a startup?"
    ]
    
    print("Router test results:")
    print("-" * 50)
    
    for prompt in test_prompts:
        gpt, claude = choose_model(prompt, "gpt-mini", "gpt-4o", "claude-haiku", "claude-sonnet")
        explanation = get_routing_explanation(prompt, gpt, claude, "gpt-mini", "gpt-4o", "claude-haiku", "claude-sonnet")
        
        print(f"Prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
        print(f"Models: {gpt}, {claude}")
        print(f"Reason: {explanation}")
        print()
    
    # Show batch statistics
    stats = router_stats(test_prompts)
    print("Batch statistics:")
    print(f"Strong model rate: {stats['strong_model_rate']:.2%}")
    print(f"Average length: {stats['avg_length']:.0f} chars")
    print(f"Average complexity: {stats['avg_complexity']:.1f}/10")
