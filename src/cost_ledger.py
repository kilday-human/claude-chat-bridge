"""
cost_ledger.py — cost tracking utility with router decision logging.
Keeps a running log of token usage, cost, and router reasoning.
"""

import os
import json
from datetime import datetime
from typing import Optional

# Where to save logs
LOG_FILE = os.path.join(os.path.dirname(__file__), "..", "logs", "cost_ledger.jsonl")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Example price table (USD per 1k tokens)
PRICES = {
    "gpt-mini": 0.0005,     # GPT-4o-mini
    "gpt": 0.0020,          # GPT-4o
    "chatgpt": 0.0005,      # legacy alias
    "claude": 0.0020,       # Claude Sonnet
    "claude-lite": 0.0010,  # Claude Haiku (cheaper)
}

def log_cost(model: str, tokens: int, reason: Optional[str] = None) -> None:
    """Append a record of token usage, cost, and optional router reason."""
    cost = (tokens / 1000) * PRICES.get(model, 0.001)
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": model,
        "tokens": tokens,
        "cost": round(cost, 6),
    }
    if reason:
        record["router_reason"] = reason
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

def read_summary() -> dict:
    """Return total tokens + cost so far."""
    if not os.path.exists(LOG_FILE):
        return {"tokens": 0, "cost": 0.0}
    tokens, cost = 0, 0.0
    with open(LOG_FILE) as f:
        for line in f:
            rec = json.loads(line)
            tokens += rec.get("tokens", 0)
            cost += rec.get("cost", 0.0)
    return {"tokens": tokens, "cost": round(cost, 6)}
