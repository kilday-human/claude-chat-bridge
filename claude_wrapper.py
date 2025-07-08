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

@retry(
    stop=stop_after_attempt(5),
    wait=wait_fibonacci_jitter(),
    reraise=True
)
def send_to_claude(messages, model="claude-3-haiku-20240307"):
    """
    Send a list of messages to Claude using the modern /v1/messages API
    Returns the content string from Claude's response
    """
    if not CLAUDE_API_KEY:
        raise ValueError("CLAUDE_API_KEY environment variable not set")
    
    if CLAUDE_API_KEY.startswith("your-"):
        logger.debug(f"DEBUG CLAUDE_API_KEY startswith: your …")
    else:
        logger.debug(f"DEBUG CLAUDE_API_KEY startswith: {CLAUDE_API_KEY[:8]} …")
    
    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    
    payload = {
        "model": model,
        "max_tokens": 1000,
        "messages": messages
    }
    
    logger.debug(f"DEBUG claude payload: {payload}")
    logger.debug(f"DEBUG claude headers: {headers}")
    
    resp = requests.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers)
    
    logger.debug(f"DEBUG status_code: {resp.status_code}")
    logger.debug(f"DEBUG response body: {resp.text}")
    
    resp.raise_for_status()
    
    data = resp.json()
    content = data["content"][0]["text"]
    return content
