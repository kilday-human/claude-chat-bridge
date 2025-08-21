"""
chatgpt_wrapper.py — wrapper to call OpenAI ChatGPT API.
"""

import os
import requests
from dotenv import load_dotenv

# Ensure .env gets loaded
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

ENDPOINT = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"


def send_to_chatgpt(prompt: str, mock: bool = False) -> str:
    """
    Send a prompt to OpenAI ChatGPT.
    If mock=True, return a fake reply.
    """
    if mock:
        return f"[mock-chatgpt-reply] {prompt[:40]}..."

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    body = {
        "model": DEFAULT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }

    resp = requests.post(ENDPOINT, headers=headers, json=body)

    # Defensive error handling
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
        return data["choices"][0]["message"]["content"]
    except Exception:
        return f"[chatgpt-wrapper-error] unexpected response: {data}"
