#!/usr/bin/env python3
"""
Pretty-print recent cost ledger entries with router reasons.
"""

import json
from pathlib import Path

LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "cost_ledger.jsonl"

def tail_logs(n=10):
    if not LOG_FILE.exists():
        print("No logs found.")
        return
    with open(LOG_FILE) as f:
        lines = f.readlines()[-n:]
    for line in lines:
        rec = json.loads(line)
        ts   = rec.get("timestamp", "?")
        model = rec.get("model", "?")
        tokens = rec.get("tokens", "?")
        cost = rec.get("cost", "?")
        reason = rec.get("router_reason", "")
        reason_str = f" | reason={reason}" if reason else ""
        print(f"{ts} | {model:10} | tokens={tokens} | cost=${cost}{reason_str}")

if __name__ == "__main__":
    tail_logs()
