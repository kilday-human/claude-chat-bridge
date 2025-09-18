"""
chatgpt_wrapper.py — wrapper to call OpenAI ChatGPT API.
Returns (text, metadata) for consistency with Claude wrapper.
"""

import os
import time
import requests
from dotenv import load_dotenv

# Ensure .env gets loaded
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

ENDPOINT = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def send_to_chatgpt(prompt: str, mock: bool = False):
    """
    Send a prompt to OpenAI ChatGPT.
    Returns: (text, metadata) just like Claude wrapper.
    """
    if mock:
        text = f"[mock-chatgpt-reply] {prompt[:40]}..."
        return text, {
            "model": "chatgpt-mock",
            "usage": {"in": 0, "out": len(text.split()), "total": len(text.split())},
            "latency_s": 0.001,
        }

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    body = {
        "model": DEFAULT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        # Remove temperature parameter - let it use default value
    }

    t0 = time.perf_counter()
    resp = requests.post(ENDPOINT, headers=headers, json=body, timeout=60)
    latency = time.perf_counter() - t0

    try:
        resp.raise_for_status()
    except Exception as e:
        try:
            data = resp.json()
        except Exception:
            data = None

        if data and "error" in data:
            error_msg = data["error"].get("message", "Unknown error")
        else:
            error_msg = f"HTTP {resp.status_code}, raw: {resp.text}"

        raise ValueError(f"OpenAI API error: {error_msg}") from e

    data = resp.json()
    try:
        text = data["choices"][0]["message"]["content"]
    except Exception:
        text = f"[chatgpt-wrapper-error] unexpected response: {data}"

    # Estimate tokens very roughly by word count
    out_tokens = len(text.split())
    in_tokens = len(prompt.split())
    total_tokens = in_tokens + out_tokens

    metadata = {
        "model": DEFAULT_MODEL,
        "usage": {"in": in_tokens, "out": out_tokens, "total": total_tokens},
        "latency_s": round(latency, 3),
    }

    return text, metadata


if __name__ == "__main__":
    t, m = send_to_chatgpt("Hello world", mock=True)
    print("ChatGPT(mock):", t, m)
