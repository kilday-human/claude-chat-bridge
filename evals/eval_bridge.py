#!/usr/bin/env python3
"""
Eval harness for Claude–GPT Bridge
Runs a small suite of canned prompts and shows receipts.
Use --live to hit real APIs. Use --show-json to dump raw receipts.
"""

import sys
import subprocess
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = ROOT / "logs" / "cost_ledger.jsonl"


def run_cli(args):
    cmd = ["python3", "cli_bridge.py"] + args
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    return result.stdout.strip()


def read_summary():
    if not LOG_FILE.exists():
        return {"tokens": 0, "cost": 0.0}
    tokens, cost = 0, 0.0
    with open(LOG_FILE) as f:
        for line in f:
            rec = json.loads(line)
            tokens += rec.get("tokens", rec.get("usage", {}).get("total", 0))
            cost += rec["cost"]
    return {"tokens": tokens, "cost": round(cost, 6)}


def tail_logs(n=5):
    """Print the last n log entries nicely and return them."""
    if not LOG_FILE.exists():
        print("No logs found.")
        return []
    with open(LOG_FILE) as f:
        lines = f.readlines()[-n:]
    print("\n📜 Recent Log Entries")
    records = []
    for line in lines:
        rec = json.loads(line)
        records.append(rec)
        ts = rec.get("timestamp", "?")
        model = rec.get("model", "?")
        tokens = rec.get("tokens", "?")
        cost = rec.get("cost", "?")
        reason = rec.get("router_reason", "")
        reason_str = f" | reason={reason}" if reason else ""
        print(f"{ts} | {model:10} | tokens={tokens} | cost=${cost}{reason_str}")
    return records


def main():
    live = "--live" in sys.argv
    show_json = "--show-json" in sys.argv
    mock_flag = [] if live else ["--mock"]

    mode_marker = "✅ (real API calls)" if live else "🎭 (mock mode)"
    print(f"🚀 Running evals... {mode_marker}")
    print()

    scenarios = [
        ["Explain quantum entanglement", "--dual"] + mock_flag,
        ["What is the capital of France?", "--router"] + mock_flag,
        ["Write a haiku about bridges."] + mock_flag,
    ]

    for i, args in enumerate(scenarios, 1):
        print(f"\n=== Scenario {i}: {' '.join(args)} ===")
        out = run_cli(args)
        print(out)

    summary = read_summary()
    print("\n📊 Cost Summary")
    print(f"  Tokens: {summary['tokens']}")
    print(f"  Cost:   ${summary['cost']}")

    # Always show last few receipts
    records = tail_logs(5)

    # Optionally show raw JSON
    if show_json and records:
        print("\n🪵 Raw JSON Receipts")
        for rec in records:
            print(json.dumps(rec))
    

if __name__ == "__main__":
    main()
