#!/usr/bin/env python3
"""
Test webhook timeout specifically without AI API calls
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from webhook_integration import track_gpt_bridge_usage_sync

async def test_webhook_timeout():
    """Test webhook timeout with hanging endpoint"""
    
    print("🧪 Testing webhook timeout...")
    
    # Test with hanging endpoint
    os.environ['ZAPIER_WEBHOOK_URL'] = 'http://httpbin.org/delay/5'
    os.environ['ENABLE_WEBHOOK'] = 'true'
    
    print("Testing with 5-second delay endpoint (1-second timeout)...")
    start_time = asyncio.get_event_loop().time()
    
    try:
        result = track_gpt_bridge_usage_sync(
            session_id="timeout_test",
            query="Test timeout",
            model_name="gpt-4o-mini",
            response_time_ms=1000,
            estimated_tokens=10,
            success=True,
            user_agent="Timeout-Test"
        )
        
        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time
        
        print(f"Result: {result}")
        print(f"Duration: {duration:.2f} seconds")
        
        if duration < 2.0:
            print("✅ Timeout working correctly (under 2 seconds)")
        else:
            print("❌ Timeout not working (over 2 seconds)")
            
    except Exception as e:
        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time
        print(f"Exception: {e}")
        print(f"Duration: {duration:.2f} seconds")
        
        if duration < 2.0:
            print("✅ Timeout working correctly (under 2 seconds)")
        else:
            print("❌ Timeout not working (over 2 seconds)")

if __name__ == "__main__":
    asyncio.run(test_webhook_timeout())
