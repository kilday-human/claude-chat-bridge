import os
import requests
import time
import logging
from tenacity import retry, stop_after_attempt
from backoff_utils import wait_fibonacci_jitter
from token_utils import count_gpt_tokens

# Constants and logger
CLAUDE_API_URL = "https://api.anthropic.com/v1/complete"
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
logger = logging.getLogger("bridge.claude")

@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_fibonacci_jitter(max_attempts=5)
)
def send_to_claude(messages):
    if not CLAUDE_API_KEY:
        raise RuntimeError("CLAUDE_API_KEY environment variable is not set")

    # Count and log tokens sent
    token_in = count_gpt_tokens(messages)
    logger.info("Claude request tokens: %d", token_in)

    # Build payload and headers
    payload = {
        "model": "claude-v1",
        "messages": messages,
        "max_tokens_to_sample": 300
    }
    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "Content-Type": "application/json"
    }

    # Time the HTTP call
    start = time.perf_counter()
    response = requests.post(CLAUDE_API_URL, json=payload, headers=headers)
    latency = time.perf_counter() - start
    logger.info("Claude call latency: %.3f sec", latency)

    # Raise for status and parse JSON
    response.raise_for_status()
    data = response.json()

    # Extract the assistant’s response
    assistant_resp = data.get("assistant_response") or data.get("completion")
    if not isinstance(assistant_resp, dict):
        raise ValueError(f"Unexpected response format: {data!r}")
    content = assistant_resp.get("content", "")

    # Count and log tokens received
    token_out = count_gpt_tokens([{"role": "assistant", "content": content}])
    logger.info("Claude response tokens: %d", token_out)

    return {
        "role": assistant_resp.get("role", "assistant"),
        "content": content
    }

