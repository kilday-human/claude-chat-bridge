#!/usr/bin/env python3
"""
Simple webhook test - verify basic functionality
"""

import requests
import json
from datetime import datetime

def test_webhook_payload():
    """Test webhook payload creation"""
    payload = {
        'timestamp': datetime.now().isoformat(),
        'session_id': 'test_123',
        'query': {
            'summary': 'test query',
            'category': 'general',
            'estimated_complexity': 1
        },
        'response': {
            'model_used': 'gpt-4',
            'response_time_ms': 1000,
            'estimated_tokens': 100,
            'success': True
        },
        'user_agent': 'test'
    }
    
    print("✅ Webhook payload created successfully")
    print(json.dumps(payload, indent=2))
    return payload

def test_webhook_sending(payload, webhook_url):
    """Test webhook sending"""
    try:
        response = requests.post(webhook_url, json=payload, timeout=5)
        print(f"✅ Webhook sent successfully: {response.status_code}")
        return True
    except requests.exceptions.ConnectionError:
        print("⚠️ Webhook server not available (expected)")
        return False
    except Exception as e:
        print(f"❌ Webhook error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Basic Webhook Functionality")
    print("=" * 50)
    
    # Test 1: Payload creation
    payload = test_webhook_payload()
    
    # Test 2: Webhook sending (will fail if no server)
    webhook_url = "http://localhost:8000/api/webhook/usage"
    test_webhook_sending(payload, webhook_url)
    
    print("\n✅ Basic webhook functionality test completed")
    print("📝 Next: Start webhook server and test end-to-end")
