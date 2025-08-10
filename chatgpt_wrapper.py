#!/usr/bin/env python3
"""
Robust ChatGPT wrapper with GPT-5 Responses API support
Replaces the legacy chatgpt_wrapper.py with proper error handling
"""

import requests
import json
import os
from typing import Dict, Any, Tuple, Optional

def _text_from_responses_api(data: Dict[str, Any]) -> str:
    """Extract text from GPT-5 Responses API format"""
    output = data.get("output")
    if not isinstance(output, list):
        return ""
    
    parts = []
    for item in output or []:
        if not isinstance(item, dict):
            continue
            
        # Look for message type output
        if item.get("type") == "message":
            content = item.get("content", [])
            for content_item in content:
                if isinstance(content_item, dict) and content_item.get("type") == "output_text":
                    text = content_item.get("text", "")
                    if text:
                        parts.append(str(text))
    
    return "".join(parts).strip()

def _text_from_chat_completions(data: Dict[str, Any]) -> str:
    """Extract text from legacy Chat Completions API format"""
    choices = data.get("choices", [])
    if not choices:
        return ""
    
    choice = choices[0]
    message = choice.get("message", {})
    return message.get("content", "").strip()

def send_to_chatgpt(
    transcript: str,
    model: str = "gpt-5",
    public: bool = False,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    **kwargs  # Accept any additional kwargs for compatibility
) -> Tuple[str, Dict[str, Any]]:
    """
    Send transcript to ChatGPT and return response text + metadata
    
    Args:
        transcript: The input text to send
        model: Model name (gpt-5, gpt-4, etc.)
        public: Whether to use public endpoint (unused for now)
        max_tokens: Maximum output tokens
        temperature: Sampling temperature
        **kwargs: Additional parameters for compatibility
    
    Returns:
        Tuple of (response_text, full_api_response)
    """
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    # Determine endpoint and payload based on model
    if model.startswith("gpt-5"):
        # Use new Responses API for GPT-5
        endpoint = "https://api.openai.com/v1/responses"
        payload = {
            "model": model,
            "input": transcript,
            "max_output_tokens": max_tokens,
            "temperature": temperature
        }
    else:
        # Use Chat Completions API for GPT-4 and earlier
        endpoint = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": transcript}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Make the API call
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        raise ValueError(f"[GPT] API request failed: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"[GPT] Invalid JSON response: {e}")
    
    # Handle API errors
    if "error" in data:
        error_msg = data["error"].get("message", "Unknown error")
        raise ValueError(f"[GPT] API error: {error_msg}")
    
    # Check for incomplete responses (GPT-5 specific)
    status = data.get("status")
    if status == "incomplete":
        incomplete = data.get("incomplete_details") or {}
        incomplete_reason = incomplete.get("reason", "unknown")
        if incomplete_reason == "max_output_tokens":
            raise ValueError(f"[GPT] Response incomplete due to max_output_tokens limit. "
                           f"Try increasing max_tokens (current: {max_tokens})")
        else:
            raise ValueError(f"[GPT] Response incomplete: {incomplete_reason}")
    
    # Extract text based on API format
    if model.startswith("gpt-5"):
        text = _text_from_responses_api(data)
    else:
        text = _text_from_chat_completions(data)
    
    if not text:
        # Log the response for debugging
        print(f"[DEBUG][GPT] No text extracted from response: {json.dumps(data, indent=2)}")
        return "", data
    
    return text, data

def mock_send_to_chatgpt(
    transcript: str,
    model: str = "gpt-5",
    **kwargs
) -> Tuple[str, Dict[str, Any]]:
    """
    Mock version for testing without API calls
    """
    # Simple mock responses based on input
    if "bridge-ok" in transcript.lower():
        text = "bridge-ok"
    elif "alive" in transcript.lower() and "done" in transcript.lower():
        text = "I am not alive; I am an AI model.\ndone"
    elif "haiku" in transcript.lower():
        text = "Code flows between minds,\nAI voices in harmony—\nBridge spans silicon."
    elif "quantum" in transcript.lower():
        text = "Quantum computing harnesses quantum mechanical phenomena like superposition and entanglement to process information exponentially faster than classical computers for specific problems."
    elif "*" in transcript and "+" in transcript:  # Math operation
        text = "4890"  # Mock result for 17 * 234 + 892
    else:
        text = f"Mock response to: {transcript[:50]}{'...' if len(transcript) > 50 else ''}"
    
    # Mock metadata
    mock_data = {
        "id": "mock_response_123",
        "model": model,
        "usage": {"input_tokens": len(transcript.split()), "output_tokens": len(text.split()), "total_tokens": len(transcript.split()) + len(text.split())},
        "mock": True
    }
    
    return text, mock_data

# For backwards compatibility
def send_to_gpt(*args, **kwargs):
    """Legacy function name compatibility"""
    return send_to_chatgpt(*args, **kwargs)

if __name__ == "__main__":
    # Quick test
    try:
        result, meta = mock_send_to_chatgpt("Reply 'bridge-ok' only.")
        print(f"Mock test: {result}")
        
        # Test real API if key is available
        if os.getenv("OPENAI_API_KEY"):
            result, meta = send_to_chatgpt("Reply 'test-ok' only.", model="gpt-5", max_tokens=64)
            print(f"Real API test: {result}")
        else:
            print("No API key found, skipping real test")
            
    except Exception as e:
        print(f"Test failed: {e}")
