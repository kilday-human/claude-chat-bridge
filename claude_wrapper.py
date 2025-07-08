import os
import requests
import time
import logging
from tenacity import retry, stop_after_attempt
from backoff_utils import wait_fibonacci_jitter
from dotenv import load_dotenv

load_dotenv(dotenv_path="./.env")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
logger = logging.getLogger("bridge.claude")

def _anthropic_prompt(messages):
    role_map = {"user": "Human", "assistant": "Assistant"}
    p = ""
    for m in messages:
        role = role_map.get(m["role"].lower(), m["role"].capitalize())
        p += f"{role}: {m['content']}\n\n"
    p += "Assistant:"
    return p

@retry(reraise=True, stop=stop_after_attempt(5), wait=wait_fibonacci_jitter(max_attempts=5))
def send_to_claude(messages):
    if not CLAUDE_API_KEY:
        raise RuntimeError("CLAUDE_API_KEY environment variable is not set")

    prompt = _anthropic_prompt(messages)
    payload = {
        "model": "claude-3-opus-20240229",
        "prompt": prompt,
        "max_tokens_to_sample": 512,
        "stop_sequences": ["\n\nHuman:"],
    }
    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }

    start = time.perf_counter()
    resp = requests.post("https://api.anthropic.com/v1/complete", json=payload, headers=headers)
    latency = time.perf_counter() - start
    logger.info("Claude call latency: %.3f sec", latency)

    resp.raise_for_status()
    data = resp.json()

    # when running your pytest harness, they’ll return this form:
    if "assistant_response" in data:
        return data["assistant_response"]["content"]

    # production API returns “completion”
    if "completion" in data:
        return data["completion"].strip()

    raise ValueError(f"Unexpected response format: {data!r}")

