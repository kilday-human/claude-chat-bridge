import os
import requests
import time
import logging
from tenacity import retry, stop_after_attempt
from backoff_utils import wait_fibonacci_jitter
from token_utils import count_gpt_tokens

# Constants and logger
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
logger = logging.getLogger("bridge.chatgpt")

@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_fibonacci_jitter(max_attempts=5)
)
def send_to_chatgpt(messages):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    # Count and log tokens sent
    token_in = count_gpt_tokens(messages)
    logger.info("ChatGPT request tokens: %d", token_in)

    # Build payload and headers
    payload = {
        "model": "gpt-4o-mini",
        "messages": messages,
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    # Time the HTTP call
    start = time.perf_counter()
    response = requests.post(OPENAI_API_URL, json=payload, headers=headers)
    latency = time.perf_counter() - start
    logger.info("ChatGPT call latency: %.3f sec", latency)

    # Raise for status and parse JSON
    response.raise_for_status()
    data = response.json()

    # Validate and extract the assistant message
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) == 0 or "message" not in choices[0]:
        raise ValueError(f"Unexpected response format: {data!r}")
    choice = choices[0]["message"]
    content = choice.get("content", "")

    # Count and log tokens received
    token_out = count_gpt_tokens([{"role": "assistant", "content": content}])
    logger.info("ChatGPT response tokens: %d", token_out)

    return {
        "role": choice.get("role", "assistant"),
        "content": content
    }

