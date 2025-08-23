"""
router.py — smarter heuristic router for multi-model orchestration.
Decides which model to use and returns both the choice and the reason.
"""

from typing import Dict

# Registry of available models (real + placeholders for demo extensibility)
MODELS = {
    "gpt-mini": "gpt-4o-mini",          # cheap, fast
    "gpt": "gpt-4o",                    # stronger reasoning
    "claude": "claude-3.5-sonnet",      # creative, long context
    "claude-lite": "claude-3-haiku",    # cheaper Claude option
}

def choose_model(prompt: str) -> Dict[str, str]:
    """
    Heuristic router:
      - Short factual → gpt-mini
      - Math/structured → gpt
      - Creative/writing → claude
      - Long prompts (>500 chars) → claude
      - Default → gpt-mini
    Returns dict: {"model": key, "reason": explanation}
    """

    p = prompt.lower()

    # Long prompts → Claude
    if len(prompt) > 500:
        return {"model": "claude", "reason": "prompt length > 500"}

    # Creative tasks → Claude
    creative_keywords = ["story", "poem", "haiku", "song", "lyrics", "creative", "imagine"]
    if any(word in p for word in creative_keywords):
        return {"model": "claude", "reason": "creative keyword detected"}

    # Math or structured Qs → GPT
    if any(sym in p for sym in ["+", "-", "*", "/", "=", "equation", "calculate"]):
        return {"model": "gpt", "reason": "math/structured query detected"}

    # Factual short Qs → GPT-mini
    if len(prompt.split()) < 20:
        return {"model": "gpt-mini", "reason": "short factual query"}

    # Fallback
    return {"model": "gpt-mini", "reason": "default fallback"}
