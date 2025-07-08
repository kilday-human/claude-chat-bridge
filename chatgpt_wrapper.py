import os
import requests
import time
import logging
from tenacity import retry, stop_after_attempt
from backoff_utils import wait_fibonacci_jitter
from token_utils import count_gpt_tokens
from dotenv import load_dotenv

# 1) Load .env so the key is set
load_dotenv(dotenv_path="./.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
logger = logging.getLogger("bridge.chatgpt")

@retry(reraise=True, stop=stop_after_attempt(5), wait=wait_fibonacci_jitter(max_attempts=5))
def send_to_chatgpt(messages):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    # DEBUG: show key (masked) and payload
    print("DEBUG OPENAI_API_KEY startswith:", OPENAI_API_KEY[:4], "…")
    payload = {"model": "gpt-4o-mini", "messages": messages}
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    print("DEBUG chatgpt payload:", payload)
    print("DEBUG chatgpt headers:", headers)

    # time it
    start = time.perf_counter()
    resp = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
    latency = time.perf_counter() - start
    logger.info("ChatGPT call latency: %.3f sec", latency)
    print("DEBUG status_code:", resp.status_code)
    print("DEBUG response body:", resp.text)

    resp.raise_for_status()
    data = resp.json()

    # Validate
    choices = data.get("choices", [])
    if not choices or "message" not in choices[0] or "content" not in choices[0]["message"]:
        raise ValueError(f"Unexpected response format: {data!r}")
    content = choices[0]["message"]["content"]
    return content
