#!/usr/bin/env python3
"""
Test Level 1 webhook integration and show results
"""
import os
import sys
from minimal_webhook import send_webhook
from webhook_integration_minimal import log_usage_to_zapier

def test_level1():
    """Test Level 1 webhook integration and show results"""
    print("🧪 Testing Level 1 Zapier Integration")
    print("=" * 50)
    
    # Test 1: Check environment
    print("\n1️⃣ Environment Check:")
    webhook_enabled = os.getenv('ENABLE_WEBHOOK', 'false').lower() == 'true'
    webhook_url = os.getenv('ZAPIER_WEBHOOK_URL')
    print(f"   ENABLE_WEBHOOK: {webhook_enabled}")
    print(f"   ZAPIER_WEBHOOK_URL: {'Set' if webhook_url else 'Not set'}")
    
    # Test 2: Test minimal webhook function
    print("\n2️⃣ Testing minimal webhook function:")
    try:
        result = send_webhook("Test query", "gpt-4o-mini", 1000, 50)
        print(f"   Result: {'✅ Success' if result else '❌ Failed'}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Test integration function
    print("\n3️⃣ Testing integration function:")
    try:
        log_usage_to_zapier("Test integration", "gpt-4", 1500, 75)
        print("   ✅ Integration function called successfully")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Test imports
    print("\n4️⃣ Testing imports:")
    try:
        from minimal_webhook import send_webhook
        print("   ✅ minimal_webhook imported successfully")
    except Exception as e:
        print(f"   ❌ minimal_webhook import failed: {e}")
    
    try:
        from webhook_integration_minimal import log_usage_to_zapier
        print("   ✅ webhook_integration_minimal imported successfully")
    except Exception as e:
        print(f"   ❌ webhook_integration_minimal import failed: {e}")
    
    print("\n Level 1 testing completed!")
    return True

if __name__ == "__main__":
    test_level1()
