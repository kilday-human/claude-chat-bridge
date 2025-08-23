#!/usr/bin/env python3
"""
cost_ledger.py — logs model usage costs.
"""

import os
import json
import datetime

LOG_DIR = "./logs"
LEDGER_FILE = os.path.join(LOG_DIR, "ledger.jsonl")


def log_cost_entry(model_used: str, tokens: int, cost_usd: float):
    """Append a cost entry to logs/ledger.jsonl"""
    os.makedirs(LOG_DIR, exist_ok=True)

    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "model": model_used,
        "tokens": tokens,
        "cost_usd": cost_usd
    }

    with open(LEDGER_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"[LEDGER] Logged cost: {entry}")
