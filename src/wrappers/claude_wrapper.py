import os
import time

try:
    import anthropic
except ImportError:
    anthropic = None

class ClaudeWrapper:
    """
    Wrapper for Anthropic Claude API.
    Exposes a generate(prompt: str) -> dict interface.
    Supports mock mode if no API key is set or mock=True.
    """

    def __init__(self, model: str = "claude-3-5-sonnet-20240620", max_tokens: int = 1024, mock: bool = False):
        self.model = model
        self.max_tokens = max_tokens
        self.mock = mock

        api_key = os.getenv("CLAUDE_API_KEY")

        # Enable mock mode if explicitly requested OR key missing
        if self.mock or not api_key:
            self.mock = True
            self.client = None
        else:
            if anthropic is None:
                raise ImportError("anthropic package not installed. Run `pip install anthropic`.")
            self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str, mock: bool = False) -> dict:
        if self.mock:
            # Return fake reply
            return {
                "model": self.model,
                "response": f"[mock-claude-reply] {prompt[:40]}...",
                "input_tokens": 1,
                "output_tokens": 5,
                "total_tokens": 6,
                "latency_sec": 0.001,
            }

        start = time.time()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        end = time.time()

        text = ""
        if response and response.content and len(response.content) > 0:
            text = response.content[0].text

        input_tokens = getattr(response.usage, "input_tokens", 0)
        output_tokens = getattr(response.usage, "output_tokens", 0)
        total_tokens = input_tokens + output_tokens

        return {
            "model": self.model,
            "response": text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "latency_sec": round(end - start, 3),
        }
