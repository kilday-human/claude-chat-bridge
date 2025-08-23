#!/usr/bin/env python3
import sys
import argparse

from bridge import run_bridge
from router import choose_model
from cost_ledger import log_cost_entry


def parse_args():
    parser = argparse.ArgumentParser(description="Claude–GPT Bridge CLI")
    parser.add_argument("prompt", type=str, help="Prompt to send")
    parser.add_argument("turns", type=int, nargs="?", default=1, help="Conversation turns")
    parser.add_argument("--gpt-model", type=str, default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--claude-model", type=str, default="claude-haiku", help="Anthropic model name")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per call")
    parser.add_argument("--ensure-output", action="store_true", help="Ensure output is returned")
    parser.add_argument("--log-cost", action="store_true", help="Enable cost logging")
    parser.add_argument("--parallel", action="store_true", help="Run Claude + GPT in parallel")
    parser.add_argument("--mock", action="store_true", help="Use mock responses (no API cost)")
    parser.add_argument("--router", choices=["on", "off"], default="off", help="Enable router for cheap→strong fallback")
    return parser.parse_args()


def _print_debug_header(args):
    print("\n=== Claude–GPT Bridge ===")
    print(f"Prompt: {args.prompt}")
    print(f"Turns: {args.turns}")
    print(f"Router: {args.router}")
    print(f"GPT model: {args.gpt_model}")
    print(f"Claude model: {args.claude_model}")
    print("==========================\n")


def main():
    args = parse_args()
    _print_debug_header(args)

    try:
        # Step 1: Pick model via router
        chosen_gpt = args.gpt_model
        chosen_claude = args.claude_model
        if args.router == "on":
            chosen_gpt, chosen_claude = choose_model(
                args.prompt,
                cheap_gpt="gpt-4o-mini",
                strong_gpt="gpt-5",
                cheap_claude="claude-haiku",
                strong_claude="claude-sonnet-4"
            )
            print(f"[ROUTER] Selected GPT={chosen_gpt}, Claude={chosen_claude}")

        # Step 2: Run the bridge
        result, usage = run_bridge(
            prompt=args.prompt,
            turns=args.turns,
            gpt_model=chosen_gpt,
            claude_model=chosen_claude,
            max_tokens=args.max_tokens,
            ensure_output=args.ensure_output,
            log_cost=args.log_cost,
            parallel=args.parallel,
            mock=args.mock,
        )

        # Step 3: Log cost entry
        if args.log_cost and usage:
            log_cost_entry(
                model_used=usage.get("model"),
                tokens=usage.get("tokens", 0),
                cost_usd=usage.get("cost", 0.0)
            )

        print("\n=== Output ===\n")
        print(result)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
