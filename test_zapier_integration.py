#!/usr/bin/env python3
"""
Test script for Level 1 Zapier Integration
Tests webhook integration without requiring actual Zapier setup
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from webhook_integration import UsagePayload, QueryData, ResponseData, track_gpt_bridge_usage

async def test_webhook_integration():
    """Test the webhook integration functionality"""
    
    print("🧪 Testing Level 1 Zapier Integration...")
    
    # Test 1: Create usage payload
    print("\n1. Testing payload creation...")
    
    payload = UsagePayload(
        timestamp="2024-01-15T10:30:00Z",
        session_id="test_session_123",
        query=QueryData(
            summary="How to implement a webhook integration?",
            category="architecture",
            estimated_complexity=2
        ),
        response=ResponseData(
            model_used="gpt-4",
            response_time_ms=1500,
            estimated_tokens=250,
            success=True
        ),
        user_agent="GPT-Bridge-Test"
    )
    
    print(f"✅ Payload created: {payload.query.summary} -> {payload.response.model_used}")
    
    # Test 2: Test webhook tracking (will fail gracefully if no webhook URL)
    print("\n2. Testing webhook tracking...")
    
    # Set test environment
    os.environ['ENABLE_WEBHOOK'] = 'true'
    os.environ['ZAPIER_WEBHOOK_URL'] = 'https://hooks.zapier.com/test-url'  # This will fail, but gracefully
    
    try:
        result = await track_gpt_bridge_usage(
            session_id="test_session_456",
            query="Test query for webhook integration",
            model_name="gpt-4o-mini",
            response_time_ms=2000,
            estimated_tokens=300,
            success=True,
            user_agent="GPT-Bridge-Test"
        )
        
        print(f"✅ Webhook tracking completed (success: {result})")
        
    except Exception as e:
        print(f"⚠️ Webhook tracking failed as expected: {e}")
    
    # Test 3: Test with webhook disabled
    print("\n3. Testing with webhook disabled...")
    
    os.environ['ENABLE_WEBHOOK'] = 'false'
    
    result = await track_gpt_bridge_usage(
        session_id="test_session_789",
        query="Test query with webhook disabled",
        model_name="claude-3-haiku",
        response_time_ms=1000,
        estimated_tokens=150,
        success=True,
        user_agent="GPT-Bridge-Test"
    )
    
    print(f"✅ Webhook disabled test completed (success: {result})")
    
    print("\n🎉 Level 1 integration test completed!")
    print("\nNext steps:")
    print("1. Set up your Zapier webhook URL in .env file")
    print("2. Set ENABLE_WEBHOOK=true in .env file")
    print("3. Test with real GPT-Bridge calls: python3 cli_bridge.py 'test query' --verbose")

if __name__ == "__main__":
    asyncio.run(test_webhook_integration())
