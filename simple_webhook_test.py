#!/usr/bin/env python3
"""
Simple Webhook Test - Minimal, Safe Integration
Tests basic webhook functionality without touching core GPT-Bridge code
"""

import os
import asyncio
import aiohttp
import json
import time
from datetime import datetime

# Simple webhook payload structure
def create_webhook_payload(session_id, query, model_name, response_time_ms, tokens, success):
    """Create a simple webhook payload"""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "session_id": session_id,
        "query": {
            "summary": query[:50] + ("..." if len(query) > 50 else ""),
            "category": "general",  # Simple categorization
            "estimated_complexity": 1
        },
        "response": {
            "model_used": "gpt-4" if "gpt" in model_name.lower() else "claude-3",
            "response_time_ms": response_time_ms,
            "estimated_tokens": tokens,
            "success": success
        },
        "user_agent": "GPT-Bridge-Simple-Test"
    }

async def send_webhook(payload, webhook_url, timeout=5):
    """Send webhook with simple error handling"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                webhook_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout),
                headers={'Content-Type': 'application/json'}
            ) as response:
                if response.status == 200:
                    print(f"✅ Webhook sent successfully: {response.status}")
                    return True
                else:
                    print(f"⚠️ Webhook failed: {response.status}")
                    return False
    except asyncio.TimeoutError:
        print(f"⚠️ Webhook timed out after {timeout} seconds")
        return False
    except Exception as e:
        print(f"❌ Webhook error: {e}")
        return False

async def test_webhook_integration():
    """Test basic webhook integration"""
    print("🧪 Testing Simple Webhook Integration...")
    
    # Check environment
    webhook_url = os.getenv('ZAPIER_WEBHOOK_URL')
    enabled = os.getenv('ENABLE_WEBHOOK', 'false').lower() == 'true'
    
    print(f"Webhook URL: {webhook_url}")
    print(f"Enabled: {enabled}")
    
    if not enabled or not webhook_url:
        print("❌ Webhook not configured. Set ZAPIER_WEBHOOK_URL and ENABLE_WEBHOOK=true")
        return False
    
    # Create test payload
    payload = create_webhook_payload(
        session_id="test_session_123",
        query="Test query for webhook integration",
        model_name="gpt-4o-mini",
        response_time_ms=1500,
        tokens=200,
        success=True
    )
    
    print(f"📦 Payload: {json.dumps(payload, indent=2)}")
    
    # Send webhook
    success = await send_webhook(payload, webhook_url)
    
    if success:
        print("🎉 Webhook test completed successfully!")
    else:
        print("❌ Webhook test failed")
    
    return success

def test_with_local_server():
    """Test with local webhook server"""
    print("\n🧪 Testing with local webhook server...")
    
    # Start local server in background
    import subprocess
    import threading
    import requests
    
    # Start the webhook server
    server_process = subprocess.Popen([
        'python3', 'webhook_server.py'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to start
    time.sleep(3)
    
    try:
        # Test webhook
        webhook_url = "http://localhost:8000/api/webhook/usage"
        payload = create_webhook_payload(
            session_id="local_test_123",
            query="Test with local server",
            model_name="gpt-4o-mini",
            response_time_ms=1000,
            tokens=150,
            success=True
        )
        
        # Send webhook
        response = requests.post(webhook_url, json=payload, timeout=5)
        print(f"Local server response: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Local webhook test successful!")
            
            # Check received webhooks
            webhooks_response = requests.get("http://localhost:8000/webhooks")
            webhooks = webhooks_response.json()
            print(f"📊 Received {len(webhooks['webhooks'])} webhooks")
            
            return True
        else:
            print(f"❌ Local webhook test failed: {response.status_code}")
            return False
            
    finally:
        # Clean up
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    print("🚀 Simple Webhook Integration Test")
    print("=" * 50)
    
    # Test 1: Check if webhook is configured
    webhook_url = os.getenv('ZAPIER_WEBHOOK_URL')
    enabled = os.getenv('ENABLE_WEBHOOK', 'false').lower() == 'true'
    
    if enabled and webhook_url and "localhost" not in webhook_url:
        print("Testing with configured webhook...")
        asyncio.run(test_webhook_integration())
    else:
        print("No real webhook configured, testing with local server...")
        test_with_local_server()
    
    print("\n✅ Test completed!")
