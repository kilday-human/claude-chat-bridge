#!/usr/bin/env python3
"""
Minimal Webhook Integration - 5% Complete
Just the absolute basics that actually work
"""

import requests
import json
import time
from datetime import datetime

def send_webhook(query, model, response_time_ms, tokens):
    """Send a simple webhook - that's it"""
    webhook_url = "http://localhost:8000/api/webhook/usage"
    
    payload = {
        "timestamp": datetime.now().isoformat(),
        "session_id": f"session_{int(time.time())}",
        "query": query[:50],
        "model": model,
        "response_time_ms": response_time_ms,
        "tokens": tokens,
        "success": True
    }
    
    try:
        response = requests.post(webhook_url, json=payload, timeout=5)
        if response.status_code == 200:
            print("📊 Webhook sent successfully")
            return True
        else:
            print(f"⚠️ Webhook failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"⚠️ Webhook error: {e}")
        return False

if __name__ == "__main__":
    # Test the minimal webhook
    print("Testing minimal webhook...")
    send_webhook("test query", "gpt-4o-mini", 1000, 50)
