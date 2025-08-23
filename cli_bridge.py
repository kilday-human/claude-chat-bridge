from __future__ import annotations  # must be first!

from dotenv import load_dotenv
load_dotenv()

import argparse
import concurrent.futures

from src.router import choose_model
from src.cost_ledger import log_cost

from src.wrappers.chatgpt_wrapper import send_to_chatgpt
try:
    from src.wrappers.claude_wrapper import send_to_claude
except ImportError:
    def send_to_claude(prompt: str, mock: bool = False, **kwargs):
        return "[claude-wrapper-missing] " + prompt, {
            "model": "claude-missing",
            "usage": {"in": 0, "out": 0, "total": 0},
        }


def run_once(prompt: str, use_router: bool = False, mock: bool = False, max_tokens: int = 512, dual: bool = False) -> str:
    outputs = []

    if dual:
        gpt_text, gpt_meta = send_to_chatgpt(prompt, mock=mock)
        claude_text, claude_meta = send_to_claude(prompt, mock=mock, max_tokens=max_tokens)

        log_cost("chatgpt", gpt_meta["usage"]["total"])
        log_cost("claude", claude_meta["usage"]["total"])

        outputs.append(f"[GPT] {gpt_text}")
        outputs.append(f"[CLAUDE] {claude_text}")
        outputs.append("Router Decision: dual mode (both models)")
    else:
        if use_router:
            decision = choose_model(prompt)
            model = decision["model"]
            reason = decision["reason"]
        else:
            decision = {"model": "gpt-mini", "reason": "router disabled (default gpt-mini)"}
            model = decision["model"]
            reason = decision["reason"]

        if model.startswith("claude"):
            text, meta = send_to_claude(prompt, mock=mock, max_tokens=max_tokens)
            label = "CLAUDE"
        else:
            text, meta = send_to_chatgpt(prompt, mock=mock)
            label = "GPT" if model == "gpt" else "GPT-MINI"

        log_cost(model, meta["usage"]["total"])
        outputs.append(f"[{label}] {text}")
        outputs.append(f"Router Decision: {reason}")

    return "\n".join(outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", help="Prompt to send")
    parser.add_argument("n", type=int, nargs="?", default=1, help="Number of runs")
    parser.add_argument("--router", action="store_true", help="Enable router")
    parser.add_argument("--mock", action="store_true", help="Use mock replies")
    parser.add_argument("--parallel", action="store_true", help="Run requests in parallel")
    parser.add_argument("--no-parallel", action="store_true", help="Force sequential execution")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens for generation")
    parser.add_argument("--dual", action="store_true", help="Send to both GPT and Claude")
    args = parser.parse_args()

    def _worker(i: int):
        reply = run_once(
            args.prompt,
            use_router=args.router,
            mock=args.mock,
            max_tokens=args.max_tokens,
            dual=args.dual,
        )
        output = f"\n=== Reply {i+1} ===\n{reply}\nBridge complete\n"
        return output

    if args.parallel and not args.no_parallel and args.n > 1:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(_worker, i) for i in range(args.n)]
            for f in concurrent.futures.as_completed(futures):
                print(f.result())
    else:
        for i in range(args.n):
            print(_worker(i))


if __name__ == "__main__":
    main()
