#!/usr/bin/env python3
"""
bridge.py — runs a query across GPT and Claude models, with --mock support.
"""

import os
from typing import Tuple, Dict

from openai import OpenAI
import anthropic


def run_bridge(prompt: str,
               turns: int = 1,
               gpt_model: str = "gpt-4o-mini",
               claude_model: str = "claude-haiku",
               max_tokens: int = 512,
               ensure_output: bool = False,
               log_cost: bool = False,
               parallel: bool = False,
               mock: bool = False) -> Tuple[str, Dict]:
    """
    Returns:
        result (str)
        usage (dict)
    """

    if mock:
        result = f"[MOCK RESPONSE] {prompt[:50]}..."
        usage = {
            "model": gpt_model,
            "tokens": len(prompt.split()),
            "cost": 0.0001 * len(prompt.split())
        }
        return result, usage

    # === OpenAI client ===
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # === Anthropic client ===
    claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Step 1: Query GPT model
    gpt_resp = openai_client.chat.completions.create(
        model=gpt_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )

    gpt_text = gpt_resp.choices[0].message.content
    gpt_tokens = gpt_resp.usage.total_tokens
    gpt_cost = gpt_tokens * 0.000002  # placeholder cost calc, tune per model

    # Step 2: Query Claude model
    claude_resp = claude_client.messages.create(
        model=claude_model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )

    claude_text = claude_resp.content[0].text
    claude_tokens = claude_resp.usage.input_tokens + claude_resp.usage.output_tokens
    claude_cost = claude_tokens * 0.0000025  # placeholder cost calc

    # Combine results
    result = f"[GPT: {gpt_model}]\n{gpt_text}\n\n[Claude: {claude_model}]\n{claude_text}"

    usage = {
        "model": gpt_model,
        "tokens": gpt_tokens + claude_tokens,
        "cost": gpt_cost + claude_cost
    }

    return result, usage
