"""
router.py — simple heuristic router for Claude vs ChatGPT.
"""

def choose_model(prompt: str) -> str:
    """
    Very naive heuristic routing.
    Returns "chatgpt" or "claude".
    """
    # Example heuristic: long prompts → Claude, short → ChatGPT
    if len(prompt) > 500:
        return "claude"
    if "claude" in prompt.lower():
        return "claude"
    return "chatgpt"
