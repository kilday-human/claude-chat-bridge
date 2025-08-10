# chatgpt_wrapper.py
# Robust OpenAI wrapper that supports GPT-5 (Responses API) and legacy Chat Completions.
# Returns (text, meta) and prints concise debug info.

import os
import json
import time
from typing import Any, Dict, List, Tuple, Union

import requests

# Optional token counting (won't crash if tiktoken isn't available)
try:
    from token_utils import count_gpt_tokens  # repo helper
except Exception:
    def count_gpt_tokens(_payload: Union[str, List[Dict[str, str]]], _model: str) -> Union[int, None]:
        return None

OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

def _mask_key(k: str) -> str:
    if not k:
        return "MISSING"
    if len(k) <= 6:
        return k
    return f"{k[:4]}…{k[-4:]}"

def _get_key(public: bool) -> str:
    if public:
        return os.environ.get("OPENAI_API_KEY_PUBLIC", "")
    return os.environ.get("OPENAI_API_KEY", "")

def _is_gpt5(model: str) -> bool:
    return model.lower().startswith("gpt-5")

def _debug_print(title: str, value: Any) -> None:
    # Keep logs readable & small
    try:
        if isinstance(value, (dict, list)):
            s = json.dumps(value, ensure_ascii=False)[:500]
        else:
            s = str(value)[:500]
        print(f"[DEBUG][GPT] {title}: {s}")
    except Exception:
        print(f"[DEBUG][GPT] {title}: (unprintable)")

def _parse_gpt5_response(data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Parse OpenAI Responses API (GPT-5) format robustly:
    - data['output'] is a list containing items of type 'reasoning' and/or 'message'
    - a 'message' has content list entries with type 'output_text' { text: ... }
    - response may be 'incomplete' due to max_output_tokens; might include reasoning only
    """
    if "model" in data:
        print(f"[DEBUG][GPT] API reports model: {data['model']}")

    status = data.get("status")
    incomplete = status == "incomplete"
    incomplete_reason = data.get("incomplete_details", {}).get("reason")

    text_chunks: List[str] = []

    for item in data.get("output", []):
        t = item.get("type")
        # If it's a completed message block
        if t == "message":
            # Claude-like shape but in OpenAI responses: content is list of segments
            for seg in item.get("content", []):
                if seg.get("type") == "output_text":
                    txt = seg.get("text")
                    if isinstance(txt, str):
                        text_chunks.append(txt)
        # Some variants may emit top-level output_text items
        elif t == "output_text":
            txt = item.get("text")
            if isinstance(txt, str):
                text_chunks.append(txt)

    text = "".join(text_chunks).strip()

    usage_obj = data.get("usage", {}) or {}
    in_tok = usage_obj.get("input_tokens")
    out_tok = usage_obj.get("output_tokens")
    total_tok = None
    if isinstance(in_tok, int) and isinstance(out_tok, int):
        total_tok = in_tok + out_tok

    meta: Dict[str, Any] = {
        "model": data.get("model"),
        "usage": {
            "in": in_tok,
            "out": out_tok,
            "total": total_tok,
        },
        "status": status,
        "incomplete": incomplete,
        "incomplete_reason": incomplete_reason,
    }

    return text, meta

def _parse_chat_completions_response(data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Parse legacy Chat Completions format.
    """
    if "model" in data:
        print(f"[DEBUG][GPT] API reports model: {data['model']}")
    choices = data.get("choices") or []
    content = ""
    if choices:
        msg = choices[0].get("message") or {}
        content = (msg.get("content") or "").strip()

    usage_obj = data.get("usage", {}) or {}
    in_tok = usage_obj.get("prompt_tokens")
    out_tok = usage_obj.get("completion_tokens")
    total_tok = usage_obj.get("total_tokens")

    meta: Dict[str, Any] = {
        "model": data.get("model"),
        "usage": {
            "in": in_tok,
            "out": out_tok,
            "total": total_tok,
        },
        "status": "completed",
        "incomplete": False,
        "incomplete_reason": None,
    }
    return content, meta

def send_to_chatgpt(
    transcript: Union[str, List[Dict[str, str]]],
    model: str = None,
    public: bool = False,
    max_tokens: int = 512,
) -> Tuple[str, Dict[str, Any]]:
    """
    Send a prompt or messages to OpenAI.
    - If model starts with gpt-5 -> use Responses API with 'input' and 'max_output_tokens'
    - Else -> use Chat Completions with 'messages'
    Returns (text, meta)
    """
    env_model = os.environ.get("OPENAI_MODEL", "").strip()
    model = (model or env_model or "gpt-5").strip()

    api_key = _get_key(public)
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing (and OPENAI_API_KEY_PUBLIC if public=True).")

    print(f"[DEBUG][GPT] Using model: {model} | public={public}")
    print(f"[DEBUG][GPT] Key: {_mask_key(api_key)}")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Build request
    is_gpt5 = _is_gpt5(model)

    if is_gpt5:
        # Responses API: https://api.openai.com/v1/responses
        url = f"{OPENAI_API_BASE}/responses"
        # 'input' can be a string or a messages array; both worked in your logs.
        if isinstance(transcript, list):
            payload_input = transcript
        else:
            payload_input = str(transcript)

        payload: Dict[str, Any] = {
            "model": model,
            "input": payload_input,
            "max_output_tokens": max_tokens if isinstance(max_tokens, int) and max_tokens > 0 else 512,
        }
        print(f"[DEBUG][GPT] Endpoint: {url}")
        _debug_print("Payload", payload)

        t0 = time.time()
        resp = requests.post(url, headers=headers, json=payload, timeout=90)
        t1 = time.time()

        if resp.status_code != 200:
            print(f"[DEBUG][GPT] HTTP {resp.status_code} in {t1 - t0:.3f}s (responses API)")
            try:
                print("[DEBUG][GPT] Error body:", json.dumps(resp.json(), ensure_ascii=False))
            except Exception:
                print("[DEBUG][GPT] Error body: (unprintable)")
            resp.raise_for_status()

        print(f"[DEBUG][GPT] HTTP 200 in {t1 - t0:.3f}s")
        data = resp.json()
        text, meta = _parse_gpt5_response(data)
        meta["latency_s"] = round(t1 - t0, 3)

        # If it came back incomplete because the model used all tokens on reasoning,
        # return an empty string rather than crashing; caller can decide what to do.
        if meta.get("incomplete") and not text:
            # Provide a tiny fallback string so upstream doesn’t explode.
            text = ""

        return text, meta

    else:
        # Legacy Chat Completions
        url = f"{OPENAI_API_BASE}/chat/completions"

        if isinstance(transcript, list):
            messages = transcript
        else:
            messages = [{"role": "user", "content": str(transcript)}]

        # Keep params compatible with modern 4o/mini
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens if isinstance(max_tokens, int) and max_tokens > 0 else 512,
        }

        print(f"[DEBUG][GPT] Endpoint: {url}")
        _debug_print("Payload", payload)
        _debug_print("Headers", {k: (v if k != "Authorization" else f"Bearer {_mask_key(api_key)}") for k, v in headers.items()})

        t0 = time.time()
        resp = requests.post(url, headers=headers, json=payload, timeout=90)
        t1 = time.time()

        if resp.status_code != 200:
            print(f"[DEBUG][GPT] HTTP {resp.status_code} in {t1 - t0:.3f}s")
            try:
                print("[DEBUG][GPT] Error body:", json.dumps(resp.json(), ensure_ascii=False))
            except Exception:
                print("[DEBUG][GPT] Error body: (unprintable)")
            resp.raise_for_status()

        print(f"[DEBUG][GPT] HTTP 200 in {t1 - t0:.3f}s")
        data = resp.json()
        text, meta = _parse_chat_completions_response(data)
        meta["latency_s"] = round(t1 - t0, 3)
        return text, meta

