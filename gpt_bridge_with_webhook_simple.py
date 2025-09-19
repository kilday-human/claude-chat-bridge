#!/usr/bin/env python3
"""
GPT-Bridge with Simple Webhook Integration
A simple wrapper that adds webhook tracking without async complexity
"""

import subprocess
import sys
import json
import requests
import time
import os
import threading
from datetime import datetime

def create_webhook_payload(session_id, query, model_name, response_time_ms, tokens, success):
    """Create a simple webhook payload"""
    return {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "query": {
            "summary": query[:50] + ("..." if len(query) > 50 else ""),
            "category": "general",
            "estimated_complexity": 1
        },
        "response": {
            "model_used": "gpt-4" if "gpt" in model_name.lower() else "claude-3",
            "response_time_ms": response_time_ms,
            "estimated_tokens": tokens,
            "success": success
        },
        "user_agent": "GPT-Bridge-Simple-Webhook"
    }

def send_webhook_sync(payload, webhook_url, timeout=5):
    """Send webhook synchronously"""
    try:
        response = requests.post(
            webhook_url,
            json=payload,
            timeout=timeout,
            headers={'Content-Type': 'application/json'}
        )
        if response.status_code == 200:
            print("📊 Usage data sent to webhook")
            return True
        else:
            print(f"⚠️ Webhook failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"⚠️ Webhook error (non-critical): {e}")
        return False

def send_webhook_async(payload, webhook_url):
    """Send webhook in background thread"""
    def _send():
        send_webhook_sync(payload, webhook_url)
    
    thread = threading.Thread(target=_send, daemon=True)
    thread.start()

def run_gpt_bridge_with_webhook(args):
    """Run GPT-Bridge and track usage with webhooks"""
    
    # Check if webhook is enabled
    webhook_enabled = os.getenv('ENABLE_WEBHOOK', 'false').lower() == 'true'
    webhook_url = os.getenv('ZAPIER_WEBHOOK_URL')
    
    if not webhook_enabled or not webhook_url:
        print("📊 Webhook tracking disabled - running GPT-Bridge normally")
        return subprocess.run([sys.executable, 'cli_bridge.py'] + args)
    
    print("📊 Webhook tracking enabled - running GPT-Bridge with usage tracking")
    
    # Extract query from arguments
    query = None
    for arg in args:
        if not arg.startswith('-'):
            query = arg
            break
    
    if not query:
        print("❌ No query found in arguments")
        return subprocess.run([sys.executable, 'cli_bridge.py'] + args)
    
    # Run GPT-Bridge and capture output
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, 'cli_bridge.py'] + args,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        end_time = time.time()
        response_time_ms = int((end_time - start_time) * 1000)
        
        # Parse the output to extract model and token info
        output = result.stdout
        model_used = "gpt-4o-mini"  # Default
        tokens_used = 0
        
        # Try to extract model from output
        for line in output.split('\n'):
            if 'Model:' in line:
                model_used = line.split('Model:')[1].strip()
            elif 'Tokens:' in line:
                try:
                    tokens_used = int(line.split('Tokens:')[1].strip())
                except:
                    tokens_used = 0
        
        # Determine success
        success = result.returncode == 0 and "ERROR" not in output.upper()
        
        # Send webhook in background if successful
        if success:
            payload = create_webhook_payload(
                session_id=f"session_{int(time.time())}",
                query=query,
                model_name=model_used,
                response_time_ms=response_time_ms,
                tokens=tokens_used,
                success=success
            )
            send_webhook_async(payload, webhook_url)
        
        # Print the original output
        print(output)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        return result
        
    except subprocess.TimeoutExpired:
        print("❌ GPT-Bridge timed out")
        return subprocess.TimeoutExpired(cmd=args, timeout=60)
    except Exception as e:
        print(f"❌ Error running GPT-Bridge: {e}")
        return subprocess.CompletedProcess(args, 1, "", str(e))

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python3 gpt_bridge_with_webhook_simple.py <gpt-bridge-args>")
        print("Example: python3 gpt_bridge_with_webhook_simple.py 'test query' --verbose")
        sys.exit(1)
    
    # Remove the script name from arguments
    args = sys.argv[1:]
    
    # Run GPT-Bridge with webhook tracking
    result = run_gpt_bridge_with_webhook(args)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
