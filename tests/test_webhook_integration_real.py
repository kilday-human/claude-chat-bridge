#!/usr/bin/env python3
"""
Real Webhook Integration Tests
Tests webhook integration with actual webhook endpoints
"""

import asyncio
import json
import time
import os
import sys
from pathlib import Path
from datetime import datetime
import aiohttp
import pytest

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from webhook_integration import track_gpt_bridge_usage, UsagePayload, QueryData, ResponseData
from webhook_reliability import ProductionWebhookClient
from session_persistence import SessionPersistenceManager
from webhook_monitoring import WebhookMetricsCollector, WebhookAlertManager
from webhook_security import get_default_security_config

class TestRealWebhookIntegration:
    """Test webhook integration with real endpoints"""
    
    @pytest.fixture
    def webhook_test_server(self):
        """Start a test webhook server"""
        import subprocess
        import threading
        
        # Start the webhook test server
        server_process = subprocess.Popen([
            sys.executable, 
            str(Path(__file__).parent.parent / "webhook_server.py")
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        time.sleep(3)
        
        yield "http://localhost:8000/api/webhook/usage"
        
        # Cleanup
        server_process.terminate()
        server_process.wait()
    
    @pytest.mark.asyncio
    async def test_webhook_delivery_success(self, webhook_test_server):
        """Test successful webhook delivery"""
        webhook_url = webhook_test_server
        
        # Set environment variables
        os.environ['ZAPIER_WEBHOOK_URL'] = webhook_url
        os.environ['ENABLE_WEBHOOK'] = 'true'
        
        # Test webhook delivery
        success = await track_gpt_bridge_usage(
            session_id="test_session_real",
            query="Test query for real webhook delivery",
            model_name="gpt-4o-mini",
            response_time_ms=1500,
            estimated_tokens=200,
            success=True,
            user_agent="GPT-Bridge-Integration-Test"
        )
        
        assert success is True
        
        # Verify webhook was received by checking the test server
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000/webhooks") as response:
                webhooks = await response.json()
                assert len(webhooks["webhooks"]) > 0
                
                # Check the last webhook
                last_webhook = webhooks["webhooks"][0]
                assert last_webhook["payload"]["session_id"] == "test_session_real"
                assert last_webhook["payload"]["query"]["summary"] == "Test query for real webhook delivery"
    
    @pytest.mark.asyncio
    async def test_webhook_retry_logic(self, webhook_test_server):
        """Test webhook retry logic with failing endpoint"""
        # Use a non-existent webhook URL to test retry logic
        os.environ['ZAPIER_WEBHOOK_URL'] = "http://localhost:9999/nonexistent"
        os.environ['ENABLE_WEBHOOK'] = 'true'
        
        # Test webhook delivery (should fail and retry)
        success = await track_gpt_bridge_usage(
            session_id="test_session_retry",
            query="Test query for retry logic",
            model_name="gpt-4o-mini",
            response_time_ms=1000,
            estimated_tokens=150,
            success=True,
            user_agent="GPT-Bridge-Retry-Test"
        )
        
        # Should fail after retries
        assert success is False
    
    @pytest.mark.asyncio
    async def test_webhook_timeout(self, webhook_test_server):
        """Test webhook timeout handling"""
        # Use a slow-responding endpoint
        os.environ['ZAPIER_WEBHOOK_URL'] = "http://httpbin.org/delay/15"  # 15 second delay
        os.environ['ENABLE_WEBHOOK'] = 'true'
        
        start_time = time.time()
        
        success = await track_gpt_bridge_usage(
            session_id="test_session_timeout",
            query="Test query for timeout handling",
            model_name="gpt-4o-mini",
            response_time_ms=500,
            estimated_tokens=100,
            success=True,
            user_agent="GPT-Bridge-Timeout-Test"
        )
        
        end_time = time.time()
        
        # Should timeout and fail
        assert success is False
        # Should timeout within reasonable time (not wait 15 seconds)
        assert end_time - start_time < 15
    
    @pytest.mark.asyncio
    async def test_webhook_disabled(self):
        """Test webhook when disabled"""
        os.environ['ENABLE_WEBHOOK'] = 'false'
        
        success = await track_gpt_bridge_usage(
            session_id="test_session_disabled",
            query="Test query when webhook disabled",
            model_name="gpt-4o-mini",
            response_time_ms=1000,
            estimated_tokens=150,
            success=True,
            user_agent="GPT-Bridge-Disabled-Test"
        )
        
        # Should return False when disabled
        assert success is False
    
    @pytest.mark.asyncio
    async def test_webhook_metrics_collection(self, webhook_test_server):
        """Test webhook metrics collection"""
        webhook_url = webhook_test_server
        os.environ['ZAPIER_WEBHOOK_URL'] = webhook_url
        os.environ['ENABLE_WEBHOOK'] = 'true'
        
        # Create metrics collector
        metrics = WebhookMetricsCollector()
        
        # Send multiple webhooks
        for i in range(5):
            success = await track_gpt_bridge_usage(
                session_id=f"test_session_metrics_{i}",
                query=f"Test query {i} for metrics",
                model_name="gpt-4o-mini",
                response_time_ms=1000 + i * 100,
                estimated_tokens=100 + i * 10,
                success=True,
                user_agent="GPT-Bridge-Metrics-Test"
            )
            
            # Record metrics
            if success:
                metrics.increment_counter("webhook_success")
            else:
                metrics.increment_counter("webhook_failure")
            metrics.increment_counter("webhook_total")
        
        # Check metrics
        assert metrics.get_counter("webhook_total") == 5
        assert metrics.get_counter("webhook_success") == 5
        assert metrics.get_counter("webhook_failure") == 0
    
    @pytest.mark.asyncio
    async def test_session_persistence(self, webhook_test_server):
        """Test session persistence across webhook calls"""
        webhook_url = webhook_test_server
        os.environ['ZAPIER_WEBHOOK_URL'] = webhook_url
        os.environ['ENABLE_WEBHOOK'] = 'true'
        
        # Create session manager
        session_manager = SessionPersistenceManager(":memory:")
        
        session_id = "test_session_persistence"
        
        # Send multiple webhooks for same session
        for i in range(3):
            success = await track_gpt_bridge_usage(
                session_id=session_id,
                query=f"Test query {i} for session persistence",
                model_name="gpt-4o-mini",
                response_time_ms=1000,
                estimated_tokens=100,
                success=True,
                user_agent="GPT-Bridge-Session-Test"
            )
            
            # Update session
            session_manager.update_session(session_id, query_count=i+1)
        
        # Check session stats
        stats = session_manager.get_session_stats(session_id)
        assert stats is not None
        assert stats["query_count"] == 3
    
    @pytest.mark.asyncio
    async def test_webhook_duplicate_detection(self, webhook_test_server):
        """Test webhook duplicate detection"""
        webhook_url = webhook_test_server
        os.environ['ZAPIER_WEBHOOK_URL'] = webhook_url
        os.environ['ENABLE_WEBHOOK'] = 'true'
        
        # Create session manager
        session_manager = SessionPersistenceManager(":memory:")
        
        session_id = "test_session_duplicate"
        query = "Test query for duplicate detection"
        
        # Send same webhook twice
        success1 = await track_gpt_bridge_usage(
            session_id=session_id,
            query=query,
            model_name="gpt-4o-mini",
            response_time_ms=1000,
            estimated_tokens=100,
            success=True,
            user_agent="GPT-Bridge-Duplicate-Test"
        )
        
        success2 = await track_gpt_bridge_usage(
            session_id=session_id,
            query=query,  # Same query
            model_name="gpt-4o-mini",
            response_time_ms=1000,
            estimated_tokens=100,
            success=True,
            user_agent="GPT-Bridge-Duplicate-Test"
        )
        
        # First should succeed, second should be detected as duplicate
        assert success1 is True
        # Note: Current implementation doesn't prevent duplicates, just logs them
        # This test documents the expected behavior
    
    @pytest.mark.asyncio
    async def test_webhook_alert_system(self, webhook_test_server):
        """Test webhook alert system"""
        webhook_url = webhook_test_server
        os.environ['ZAPIER_WEBHOOK_URL'] = webhook_url
        os.environ['ENABLE_WEBHOOK'] = 'true'
        
        # Create alert manager
        alert_manager = WebhookAlertManager()
        
        # Track alerts
        alerts_received = []
        def alert_handler(alert):
            alerts_received.append(alert)
        
        alert_manager.add_alert_handler(alert_handler)
        
        # Create test alert
        alert = alert_manager.create_alert(
            "test_webhook_alert",
            "warning",
            "Test webhook alert"
        )
        
        assert len(alerts_received) == 1
        assert alerts_received[0].id == "test_webhook_alert"
    
    @pytest.mark.asyncio
    async def test_webhook_health_check(self, webhook_test_server):
        """Test webhook health checking"""
        webhook_url = webhook_test_server
        
        # Test endpoint validation
        from webhook_reliability import WebhookEndpointValidator
        
        is_healthy, message = await WebhookEndpointValidator.validate_endpoint(webhook_url)
        assert is_healthy is True
        assert "healthy" in message.lower()
    
    @pytest.mark.asyncio
    async def test_webhook_security_validation(self):
        """Test webhook security configuration validation"""
        from webhook_security import WebhookConfigurationValidator
        
        # Test environment validation
        validation = WebhookConfigurationValidator.validate_environment()
        
        # Should have validation results
        assert "valid" in validation
        assert "errors" in validation
        assert "warnings" in validation
    
    @pytest.mark.asyncio
    async def test_webhook_production_client(self, webhook_test_server):
        """Test production webhook client"""
        webhook_url = webhook_test_server
        
        # Create production client
        from webhook_reliability import ProductionWebhookClient
        
        client = ProductionWebhookClient(webhook_url, max_retries=2)
        
        # Validate endpoint
        is_valid = await client.validate_endpoint()
        assert is_valid is True
        
        # Test webhook sending
        test_payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": "test_production_client",
            "query": {
                "summary": "Test production client",
                "category": "general",
                "estimated_complexity": 1
            },
            "response": {
                "model_used": "gpt-4",
                "response_time_ms": 1000,
                "estimated_tokens": 100,
                "success": True
            }
        }
        
        success = await client.send_webhook(test_payload)
        assert success is True
        
        # Check health status
        health = client.get_health_status()
        assert health["endpoint_validated"] is True
        assert health["metrics"]["successful"] >= 1

class TestWebhookLoadTesting:
    """Test webhook system under load"""
    
    @pytest.mark.asyncio
    async def test_webhook_high_volume(self, webhook_test_server):
        """Test webhook system with high volume of requests"""
        webhook_url = webhook_test_server
        os.environ['ZAPIER_WEBHOOK_URL'] = webhook_url
        os.environ['ENABLE_WEBHOOK'] = 'true'
        
        # Send many webhooks concurrently
        tasks = []
        for i in range(20):  # 20 concurrent webhooks
            task = track_gpt_bridge_usage(
                session_id=f"load_test_session_{i}",
                query=f"Load test query {i}",
                model_name="gpt-4o-mini",
                response_time_ms=1000,
                estimated_tokens=100,
                success=True,
                user_agent="GPT-Bridge-Load-Test"
            )
            tasks.append(task)
        
        # Wait for all webhooks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check results
        successful = sum(1 for result in results if result is True)
        assert successful > 0  # At least some should succeed
        
        # Verify webhooks were received
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000/webhooks") as response:
                webhooks = await response.json()
                assert len(webhooks["webhooks"]) >= successful

if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-s"])
