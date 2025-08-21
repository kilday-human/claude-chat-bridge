"""
cost_ledger.py — simple cost tracking utility.
Keeps a running log of token usage and estimated spend.
"""

import os
import json
from datetime import datetime

# Where to save logs
LOG_FILE = os.path.join(os.path.dirname(__file__), "..", "logs", "cost_ledger.jsonl")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Example price table (USD per 1k tokens)
PRICES = {
    "chatgpt": 0.0005,   # e.g. GPT-4o-mini input/output
    "claude": 0.0020,    # e.g. Claude Sonnet input/output
}

def log_cost(model: str, tokens: int) -> None:
    """Append a record of token usage and cost."""
    cost = (tokens / 1000) * PRICES.get(model, 0.001)
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": model,
        "tokens": tokens,
        "cost": round(cost, 6),
    }
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
            tokens += rec["tokens"]
            cost += rec["cost"]
    return {"tokens": tokens, "cost": round(cost, 6)}
