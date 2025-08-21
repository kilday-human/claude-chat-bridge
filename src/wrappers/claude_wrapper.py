#!/usr/bin/env python3
from __future__ import annotations
import os, json, time, requests
from typing import Dict, Any, Optional, Tuple

# Load keys from env
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=".env")
except ImportError:
    pass

CLAUDE_API_KEY_DEFAULT   = os.getenv("CLAUDE_API_KEY", "")
PUBLIC_ANTHROPIC_API_KEY = os.getenv("PUBLIC_ANTHROPIC_API_KEY", "")
CLAUDE_MODEL_DEFAULT     = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")
ANTHROPIC_URL            = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION        = "2023-06-01"
VERBOSE = os.getenv("BRIDGE_VERBOSE", "1") not in ("0", "false", "False", "")

def _mask(k: str) -> str:
    return f"{k[:4]}…{k[-4:]}" if k and len(k) > 8 else "***"

def _parse(data: Dict[str, Any]) -> Tuple[str, Dict[str,int]]:
    usage = data.get("usage") or {}
    text = ""
    for part in (data.get("content") or []):
        if isinstance(part, dict) and part.get("type") == "text":
            text += part.get("text") or ""
    return text.strip(), {
        "in": usage.get("input_tokens", 0),
        "out": usage.get("output_tokens", 0),
        "total": (usage.get("input_tokens", 0) or 0) + (usage.get("output_tokens", 0) or 0),
    }

def _normalize_for_claude(messages: list) -> tuple[list, str]:
    """Ensure last turn is user; repeat original request if needed."""
    system = None
    converted = []
    original_user_request = ""

    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            system = content if system is None else f"{system}\n{content}"
        elif role in ("user", "assistant"):
            if role == "user" and not original_user_request:
                original_user_request = content
            converted.append({"role": role, "content": content})

    if not converted:
        converted = [{"role": "user", "content": "Please respond."}]
        original_user_request = original_user_request or "Please respond."

    if converted[0]["role"] != "user":
        converted.insert(0, {"role": "user", "content": "Continue:"})

    if converted[-1]["role"] == "assistant":
        converted.append({"role": "user", "content": original_user_request})

    # Deduplicate exact role/content pairs
    seen, cleaned = set(), []
    for msg in converted:
        key = (msg["role"], msg["content"])
        if key not in seen:
            seen.add(key)
            cleaned.append(msg)

    if system:
        cleaned.insert(0, {"role": "system", "content": system})

    return cleaned, original_user_request

def _send_messages(messages: list, *, model: str, public: bool, max_tokens: int, temperature: float):
    api_key = PUBLIC_ANTHROPIC_API_KEY if public else CLAUDE_API_KEY_DEFAULT
    if not api_key:
        raise RuntimeError("Anthropic API key missing. Set CLAUDE_API_KEY or PUBLIC_ANTHROPIC_API_KEY")

    cleaned, _ = _normalize_for_claude(messages)
    system_msg = None
    payload_msgs = []
    for msg in cleaned:
        if msg["role"] == "system":
            system_msg = msg["content"]
        else:
            payload_msgs.append(msg)

    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": payload_msgs,
    }
    if system_msg:
        payload["system"] = system_msg

    if VERBOSE:
        print(f"[DEBUG][Claude] Using model: {model} | public={public}")
        print(f"[DEBUG][Claude] Key: {_mask(api_key)}")
        print(f"[DEBUG][Claude] Payload: {json.dumps(payload)[:300]}…")

    t0 = time.perf_counter()
    r = requests.post(ANTHROPIC_URL, json=payload, headers=headers, timeout=60)
    latency = time.perf_counter() - t0
    if VERBOSE:
        print(f"[DEBUG][Claude] HTTP {r.status_code} in {latency:.3f}s")
    r.raise_for_status()
    data = r.json()
    if VERBOSE:
        print(f"[DEBUG][Claude] API reports model: {data.get('model', '')}")

    text, usage = _parse(data)
    if not text:
        text = "(Claude provided no response)"
    return text, {"model": data.get("model", model), "latency_s": round(latency, 3), "usage": usage, "cost_usd": 0.0}

def send_to_claude(
    transcript: str,
    *,
    model: str = CLAUDE_MODEL_DEFAULT,
    max_tokens: int = 512,
    mock: bool = False,
    public: bool = False,
):
    if mock:
        low = (transcript or "").lower()
        if "bridge-ok" in low:
            return "bridge-ok", {"model": "claude-mock", "usage": {"in": 0, "out": 2}}
        if "echo 'done'" in low:
            return "done", {"model": "claude-mock", "usage": {"in": 0, "out": 2}}
        return "mock-claude", {"model": "claude-mock", "usage": {"in": 0, "out": 2}}

    messages = [{"role": "user", "content": transcript or "Please respond."}]
    return _send_messages(messages, model=model, public=public, max_tokens=max_tokens, temperature=0.2)

if __name__ == "__main__":
    t, m = send_to_claude("Reply 'bridge-ok' only.", mock=True)
    print("Claude(mock):", t, m)

