#!/usr/bin/env python3
# cli_bridge.py — demo-ready parallel Claude ↔ GPT bridge
# Safe with old/new wrappers; supports GPT-5 / Claude Opus 4.1
import argparse
import os
import sys
import time
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- Wrapper imports (sync functions) ---
from chatgpt_wrapper import send_to_chatgpt
from claude_wrapper import send_to_claude

# --- Helpers ---------------------------------------------------------------

def _call_with_fallback(fn, *args, **kwargs):
    """
    Call wrapper function with kwargs; if the wrapper has an older signature,
    retry without unknown kwargs.
    Returns (text, meta_dict or None).
    """
    try:
        out = fn(*args, **kwargs)
    except TypeError:
        # Strip kwargs and retry
        out = fn(*args)
    # Normalize return shape
    if isinstance(out, tuple) and len(out) == 2:
        return out[0] or "", out[1] or {}
    return (out or ""), {}

def _ensure_min_output(text, ensure_output: bool):
    if ensure_output and (text is None or str(text).strip() == ""):
        return "(no content)"
    return text

def _print_metrics(prefix, meta, start_ts):
    try:
        model = meta.get("model") or meta.get("reported_model")
        usage = meta.get("usage") or {}
        latency = time.time() - start_ts
        if usage:
            print(f"[METRICS][{prefix}] model={model} latency={latency:.3f}s usage={usage}")
        else:
            print(f"[METRICS][{prefix}] model={model} latency={latency:.3f}s")
    except Exception:
        pass

# --- Core bridging ---------------------------------------------------------

def build_argparser():
    p = argparse.ArgumentParser(description="Claude ↔ GPT bridge (parallel-capable).")
    p.add_argument("prompt", help="The initial user prompt.")
    p.add_argument("turns", nargs="?", type=int, default=1, help="Number of back-and-forth turns (default: 1).")
    p.add_argument("--max-tokens", type=int, default=None, help="Max output tokens per model (if supported).")
    p.add_argument("--ensure-output", action="store_true", help="If a model returns empty, substitute '(no content)'.")
    p.add_argument("--log-cost", action="store_true", help="Ask wrappers to include usage/cost in meta if supported.")
    p.add_argument("--gpt-model", default=os.getenv("OPENAI_MODEL", os.getenv("MODEL_OPENAI", "gpt-5")),
                   help="Override GPT model (default from .env or gpt-5).")
    p.add_argument("--claude-model", default=os.getenv("CLAUDE_MODEL", os.getenv("MODEL_ANTHROPIC", "claude-opus-4-1-20250805")),
                   help="Override Claude model (default from .env or opus 4.1).")
    p.add_argument("--parallel", dest="parallel", action="store_true", help="Run GPT and Claude calls in parallel.")
    p.add_argument("--no-parallel", dest="parallel", action="store_false", help="Run sequentially.")
    p.set_defaults(parallel=True)
    return p

def bridge_turn(transcript, gpt_model, claude_model, max_tokens, ensure_output, log_cost, parallel=True):
    """
    Run one turn for both models, optionally in parallel.
    transcript: list of {role, content} (user/assistant)
    Returns (gpt_text, gpt_meta, claude_text, claude_meta)
    """
    # Prepare kwargs for wrappers; they may ignore unknown keys
    gpt_kwargs = {"model": gpt_model}
    claude_kwargs = {"model": claude_model}
    if max_tokens:
        # Different wrappers may expect different names; include both
        gpt_kwargs["max_tokens"] = max_tokens
        gpt_kwargs["max_output_tokens"] = max_tokens
        claude_kwargs["max_tokens"] = max_tokens
    if log_cost:
        gpt_kwargs["log_cost"] = True
        claude_kwargs["log_cost"] = True

    def call_gpt():
        t0 = time.time()
        text, meta = _call_with_fallback(send_to_chatgpt, transcript, **gpt_kwargs)
        text = _ensure_min_output(text, ensure_output)
        meta = meta or {}
        if "model" not in meta:
            meta["model"] = gpt_model
        _print_metrics("GPT", meta, t0)
        return text, meta

    def call_claude():
        t0 = time.time()
        text, meta = _call_with_fallback(send_to_claude, transcript, **claude_kwargs)
        text = _ensure_min_output(text, ensure_output)
        meta = meta or {}
        if "model" not in meta:
            meta["model"] = claude_model
        _print_metrics("Claude", meta, t0)
        return text, meta

    if parallel:
        with ThreadPoolExecutor(max_workers=2) as pool:
            gpt_f = pool.submit(call_gpt)
            claude_f = pool.submit(call_claude)
            gpt_text, gpt_meta = gpt_f.result()
            claude_text, claude_meta = claude_f.result()
    else:
        gpt_text, gpt_meta = call_gpt()
        claude_text, claude_meta = call_claude()

    return gpt_text, gpt_meta, claude_text, claude_meta

def run_bridge(prompt, turns, gpt_model, claude_model, max_tokens, ensure_output, log_cost, parallel):
    transcript = [{"role": "user", "content": prompt}]

    for it in range(1, turns + 1):
        print(f"\n--- Iteration {it} ---\n")
        gpt_text, gpt_meta, claude_text, claude_meta = bridge_turn(
            transcript, gpt_model, claude_model, max_tokens, ensure_output, log_cost, parallel
        )

        print(f"GPT: {gpt_text}\n")
        print(f"Claude: {claude_text}\n")

        # Grow transcript for next turn. We bias the next user message to prompt follow-ups.
        transcript.append({"role": "assistant", "content": gpt_text})
        transcript.append({"role": "assistant", "content": claude_text})
        if it < turns:
            # Nudge them to continue; this mirrors the “ensure-output” style we used earlier
            transcript.append({
                "role": "user",
                "content": "Now continue collaboratively based on both responses above."
            })

# --- Main ------------------------------------------------------------------

def main():
    args = build_argparser().parse_args()

    # Surface which models we’ll use
    print(f"[DEBUG] GPT model: {args.gpt_model} | Claude model: {args.claude_model} | parallel={args.parallel}")

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
        )
    except KeyboardInterrupt:
        print("\nAborted by user.", file=sys.stderr)
        sys.exit(130)

if __name__ == "__main__":
    main()

