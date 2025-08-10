#!/usr/bin/env python3
"""
cli_bridge.py — run a small bridge between GPT and Claude models.

Usage examples:
  python3 cli_bridge.py "Reply 'bridge-ok' only." 1 --mock
  python3 cli_bridge.py "One-sentence proof you're alive; then echo 'done'." 3 --mock --no-parallel
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from typing import Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Local wrappers (you already created these)
from chatgpt_wrapper import send_to_chatgpt
try:
    from claude_wrapper import send_to_claude
except ImportError:
    # Minimal stub if claude_wrapper.py not present
    def send_to_claude(transcript: str, *, model: str = "claude-opus-4-1-20250805",
                       max_tokens: int = 512, mock: bool = False):
        if mock:
            if "bridge-ok" in transcript.lower():
                return "bridge-ok", {"model": "claude-mock"}
            if "echo 'done'" in transcript.lower():
                return "done", {"model": "claude-mock"}
            return "mock-claude", {"model": "claude-mock"}
        return "mock-claude", {"model": "claude-mock", "note": "real Claude call not implemented"}

DEFAULT_GPT_MODEL = "gpt-5"
DEFAULT_CLAUDE_MODEL = "claude-opus-4-1-20250805"


def _print_debug_header(args):
    parallel_flag = "True" if args.parallel else "False"
    print(f"[DEBUG] GPT model: {args.gpt_model} | Claude model: {args.claude_model} | parallel={parallel_flag}")
    if args.mock:
        print("[DEBUG] Running in MOCK mode (no external API calls).")
    if args.ensure_output:
        print("[DEBUG] ensure_output=True (empty responses will be replaced with a stub).")
    if args.log_cost:
        print("[DEBUG] log_cost=True (usage/cost will be logged if available).")


def _safe_text(text: str, ensure_output: bool) -> str:
    if text and text.strip():
        return text.strip()
    return "[no-text-returned]" if ensure_output else ""


def call_gpt(transcript: str, *, model: str, max_tokens: int,
             ensure_output: bool, log_cost: bool, mock: bool):
    # Try new keyword first; fall back to older wrappers that use max_tokens
    try:
        text, meta = send_to_chatgpt(
            transcript,
            model=model,
            max_output_tokens=max_tokens,
            ensure_output=ensure_output,
            log_cost=log_cost,
            mock=mock,
        )
    except TypeError as e:
        if "max_output_tokens" in str(e):
            text, meta = send_to_chatgpt(
                transcript,
                model=model,
                max_tokens=max_tokens,  # older wrappers
                ensure_output=ensure_output,
                log_cost=log_cost,
                mock=mock,
            )
        else:
            raise
    return _safe_text(text, ensure_output), meta


def call_claude(transcript: str, *, model: str, max_tokens: int,
                ensure_output: bool, log_cost: bool, mock: bool) -> Tuple[str, Dict]:
    # claude wrapper signature may ignore ensure_output/log_cost; we still pass for symmetry
    text, meta = send_to_claude(
        transcript,
        model=model,
        max_tokens=max_tokens,
        mock=mock,
    )
    return _safe_text(text, ensure_output), meta


def bridge_turn(transcript: str,
                gpt_model: str,
                claude_model: str,
                max_tokens: int,
                ensure_output: bool,
                log_cost: bool,
                parallel: bool,
                mock: bool) -> Tuple[str, Dict, str, Dict]:
    """
    One bridge iteration: get replies from both models against the same transcript.
    Returns: gpt_text, gpt_meta, claude_text, claude_meta
    """
    if parallel:
        with ThreadPoolExecutor(max_workers=2) as ex:
            gpt_f = ex.submit(call_gpt, transcript,
                              model=gpt_model, max_tokens=max_tokens,
                              ensure_output=ensure_output, log_cost=log_cost, mock=mock)
            claude_f = ex.submit(call_claude, transcript,
                                 model=claude_model, max_tokens=max_tokens,
                                 ensure_output=ensure_output, log_cost=log_cost, mock=mock)
            gpt_text, gpt_meta = gpt_f.result()
            claude_text, claude_meta = claude_f.result()
    else:
        gpt_text, gpt_meta = call_gpt(transcript,
                                      model=gpt_model, max_tokens=max_tokens,
                                      ensure_output=ensure_output, log_cost=log_cost, mock=mock)
        claude_text, claude_meta = call_claude(transcript,
                                               model=claude_model, max_tokens=max_tokens,
                                               ensure_output=ensure_output, log_cost=log_cost, mock=mock)
    return gpt_text, gpt_meta, claude_text, claude_meta


def run_bridge(prompt: str,
               turns: int,
               gpt_model: str,
               claude_model: str,
               max_tokens: int,
               ensure_output: bool,
               log_cost: bool,
               parallel: bool,
               mock: bool) -> None:
    """
    Execute N bridge iterations. Each iteration:
      - Calls both models on the current transcript
      - Appends both outputs back into the transcript for the next round
    Prints each round to stdout.
    """
    transcript = f"User: {prompt}".strip()

    for i in range(1, turns + 1):
        print(f"\n--- Iteration {i} ---\n")
        t0 = time.time()
        try:
            gpt_text, gpt_meta, claude_text, claude_meta = bridge_turn(
                transcript=transcript,
                gpt_model=gpt_model,
                claude_model=claude_model,
                max_tokens=max_tokens,
                ensure_output=ensure_output,
                log_cost=log_cost,
                parallel=parallel,
                mock=mock,
            )
        except Exception as e:
            print(f"[ERROR] bridge_turn failed: {e}")
            if ensure_output:
                gpt_text, claude_text = "[no-text-returned]", "[no-text-returned]"
                gpt_meta, claude_meta = {"error": str(e)}, {"error": str(e)}
            else:
                raise

        # Print GPT result
        print("[GPT OUTPUT]")
        print(gpt_text or "")
        if gpt_meta:
            # Keep debug concise
            model = gpt_meta.get("model")
            stop = gpt_meta.get("stop_reason")
            usage = gpt_meta.get("usage")
            cost = gpt_meta.get("estimated_cost_usd")
            lat = gpt_meta.get("latency_s")
            print(f"[DEBUG][GPT] model={model} stop={stop} usage={usage} cost={cost} latency={lat}")

        # Print Claude result
        print("\n[CLAUDE OUTPUT]")
        print(claude_text or "")
        if claude_meta:
            model = claude_meta.get("model")
            print(f"[DEBUG][Claude] model={model}")

        # Append both outputs back into transcript for the next round
        transcript += f"\nGPT: {gpt_text}\nClaude: {claude_text}"

        dt = time.time() - t0
        print(f"\n[DEBUG] Iteration {i} finished in {dt:.2f}s")

    # Final line to make it easy to scrape last outputs if needed
    print("\n--- Bridge complete ---")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("prompt", type=str, help="Initial user prompt to start the bridge")
    p.add_argument("turns", type=int, help="Number of bridge iterations to run")
    p.add_argument("--max-tokens", type=int, default=512, help="Max output tokens per model call")
    p.add_argument("--ensure-output", action="store_true", help="Replace empty outputs with a stub")
    p.add_argument("--log-cost", action="store_true", help="Log usage/cost if available")
    p.add_argument("--gpt-model", type=str, default=DEFAULT_GPT_MODEL, help="GPT model name")
    p.add_argument("--claude-model", type=str, default=DEFAULT_CLAUDE_MODEL, help="Claude model name")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--parallel", dest="parallel", action="store_true", help="Call both models concurrently")
    grp.add_argument("--no-parallel", dest="parallel", action="store_false", help="Call models sequentially")
    p.set_defaults(parallel=True)
    p.add_argument("--mock", action="store_true", help="Run without calling external APIs (canned outputs)")
    return p.parse_args()


def main():
    args = parse_args()
    _print_debug_header(args)

    try:
        run_bridge(
            prompt=args.prompt,
            turns=args.turns,
            gpt_model=args.gpt_model,
            claude_model=args.claude_model,
            max_tokens=args.max_tokens,
            ensure_output=args.ensure_output,
            log_cost=args.log_cost,
            parallel=args.parallel,
            mock=args.mock,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()

