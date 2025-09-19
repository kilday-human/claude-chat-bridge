#!/usr/bin/env python3
"""
GPT-Bridge with Webhook Integration
A wrapper that adds webhook tracking to GPT-Bridge without modifying core files
"""

import subprocess
import sys
import json
import asyncio
import aiohttp
import time
import os
from datetime import datetime

# Import the simple webhook functions
from simple_webhook_test import create_webhook_payload, send_webhook

def run_gpt_bridge_with_webhook(args):
    """Run GPT-Bridge and track usage with webhooks"""
    
    # Check if webhook is enabled
    webhook_enabled = os.getenv('ENABLE_WEBHOOK', 'false').lower() == 'true'
    webhook_url = os.getenv('ZAPIER_WEBHOOK_URL')
    
    if not webhook_enabled or not webhook_url:
        print("📊 Webhook tracking disabled - running GPT-Bridge normally")
        return subprocess.run([sys.executable, 'cli_bridge.py'] + args)
    
    print("📊 Webhook tracking enabled - running GPT-Bridge with usage tracking")
    
    # Extract query from arguments (first argument that doesn't start with -)
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
            timeout=60  # 60 second timeout
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
        
        # Send webhook asynchronously (fire and forget)
        if success:
            asyncio.create_task(send_usage_webhook(
                query, model_used, response_time_ms, tokens_used, success
            ))
        
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

async def send_usage_webhook(query, model_name, response_time_ms, tokens, success):
    """Send usage webhook asynchronously"""
    try:
        payload = create_webhook_payload(
            session_id=f"session_{int(time.time())}",
            query=query,
            model_name=model_name,
            response_time_ms=response_time_ms,
            tokens=tokens,
            success=success
        )
        
        webhook_url = os.getenv('ZAPIER_WEBHOOK_URL')
        await send_webhook(payload, webhook_url)
        print("📊 Usage data sent to webhook")
        
    except Exception as e:
        print(f"⚠️ Webhook tracking failed (non-critical): {e}")

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python3 gpt_bridge_with_webhook.py <gpt-bridge-args>")
        print("Example: python3 gpt_bridge_with_webhook.py 'test query' --verbose")
        sys.exit(1)
    
    # Remove the script name from arguments
    args = sys.argv[1:]
    
    # Run GPT-Bridge with webhook tracking
    result = run_gpt_bridge_with_webhook(args)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
