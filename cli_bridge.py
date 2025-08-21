# cli_bridge.py

from __future__ import annotations  # must be first!

# Load environment variables early
from dotenv import load_dotenv
load_dotenv()

import os
import argparse

from src.router import choose_model
from src.cost_ledger import log_cost

# Wrappers
from src.wrappers.chatgpt_wrapper import send_to_chatgpt
try:
    from src.wrappers.claude_wrapper import send_to_claude
except ImportError:
    def send_to_claude(prompt: str) -> str:
        return "[claude-wrapper-missing] " + prompt

def run_once(prompt: str, use_router: bool = False, mock: bool = False) -> str:
    model = "chatgpt"
    if use_router:
        model = choose_model(prompt)

    if model == "claude":
        reply = send_to_claude(prompt)
        tokens_used = len(prompt.split())
    else:
        reply = send_to_chatgpt(prompt, mock=mock)
        tokens_used = len(prompt.split())

    log_cost(model, tokens_used)
    return reply

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", help="Prompt to send")
    parser.add_argument("n", type=int, nargs="?", default=1, help="Number of runs")
    parser.add_argument("--router", action="store_true", help="Enable router")
    parser.add_argument("--mock", action="store_true", help="Use mock replies")
    args = parser.parse_args()

    for i in range(args.n):
        reply = run_once(args.prompt, use_router=args.router, mock=args.mock)
        print(f"\n=== Reply {i+1} ===")
        print(reply)

if __name__ == "__main__":
    main()
