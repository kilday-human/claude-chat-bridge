#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Webhook Integration
Tests all components of the production-ready webhook system
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from webhook_integration import UsagePayload, QueryData, ResponseData, track_gpt_bridge_usage
from webhook_reliability import WebhookRetryManager, WebhookDeadLetterQueue, WebhookEndpointValidator
from session_persistence import SessionPersistenceManager
from webhook_monitoring import WebhookMetricsCollector, WebhookAlertManager, WebhookHealthChecker
from webhook_security import WebhookSecurityManager, WebhookConfigurationValidator, get_default_security_config

class TestWebhookIntegration:
    """Test webhook integration components"""
    
    def test_usage_payload_creation(self):
        """Test UsagePayload creation and validation"""
        query_data = QueryData(
            summary="Test query for webhook integration",
            category="architecture",
            estimated_complexity=2
        )
        
        response_data = ResponseData(
            model_used="gpt-4",
            response_time_ms=1500,
            estimated_tokens=250,
            success=True
        )
        
        payload = UsagePayload(
            timestamp="2024-01-15T10:30:00Z",
            session_id="test_session_123",
            query=query_data,
            response=response_data,
            user_agent="GPT-Bridge-Test"
        )
        
        assert payload.session_id == "test_session_123"
        assert payload.query.category == "architecture"
        assert payload.response.model_used == "gpt-4"
        assert payload.response.success is True
    
    def test_query_categorization(self):
        """Test query categorization logic"""
        from webhook_integration import UsageTracker
        
        tracker = UsageTracker()
        
        # Test code review categorization
        assert tracker.categorize_query("Please review this pull request") == "code_review"
        assert tracker.categorize_query("Code review for feature branch") == "code_review"
        
        # Test debugging categorization
        assert tracker.categorize_query("Debug this error in the code") == "debugging"
        assert tracker.categorize_query("Fix the bug in authentication") == "debugging"
        
        # Test architecture categorization
        assert tracker.categorize_query("Design the system architecture") == "architecture"
        assert tracker.categorize_query("What's the best pattern for this?") == "architecture"
        
        # Test general categorization
        assert tracker.categorize_query("What is machine learning?") == "general"
        assert tracker.categorize_query("Hello world") == "general"
    
    def test_complexity_estimation(self):
        """Test complexity estimation logic"""
        from webhook_integration import UsageTracker
        
        tracker = UsageTracker()
        
        # High complexity
        assert tracker.estimate_complexity("A very long query that goes on and on with lots of details", 5000) == 3
        assert tracker.estimate_complexity("Short query", 4000) == 3
        
        # Medium complexity
        assert tracker.estimate_complexity("Medium length query with some details", 2000) == 2
        assert tracker.estimate_complexity("Short query", 1500) == 2
        
        # Low complexity
        assert tracker.estimate_complexity("Short", 500) == 1
        assert tracker.estimate_complexity("Medium length", 800) == 1

class TestWebhookReliability:
    """Test webhook reliability components"""
    
    def test_retry_manager_creation(self):
        """Test WebhookRetryManager initialization"""
        retry_manager = WebhookRetryManager(max_retries=3, base_delay=1, max_delay=300)
        
        assert retry_manager.max_retries == 3
        assert retry_manager.base_delay == 1
        assert retry_manager.max_delay == 300
    
    def test_delay_calculation(self):
        """Test exponential backoff delay calculation"""
        retry_manager = WebhookRetryManager(max_retries=3, base_delay=1, max_delay=300)
        
        # Test exponential backoff
        assert retry_manager.calculate_delay(1) == 1
        assert retry_manager.calculate_delay(2) == 2
        assert retry_manager.calculate_delay(3) == 4
        assert retry_manager.calculate_delay(4) == 8
        
        # Test max delay cap
        assert retry_manager.calculate_delay(10) == 300
    
    def test_dead_letter_queue(self):
        """Test dead letter queue functionality"""
        dlq = WebhookDeadLetterQueue(":memory:")  # Use in-memory database for testing
        
        # Create test message
        from webhook_reliability import WebhookMessage, WebhookAttempt, WebhookStatus
        from dataclasses import asdict
        
        message = WebhookMessage(
            id="test_message_123",
            payload={"test": "data"},
            webhook_url="https://test.com/webhook",
            created_at=datetime.utcnow(),
            attempts=[
                WebhookAttempt(1, datetime.utcnow(), WebhookStatus.FAILED, "Test error")
            ],
            max_retries=3
        )
        
        # Add to DLQ
        dlq.add_message(message, "Test final error")
        
        # Retrieve from DLQ
        messages = dlq.get_messages()
        assert len(messages) == 1
        assert messages[0].id == "test_message_123"
        assert messages[0].status == WebhookStatus.DEAD_LETTER

class TestSessionPersistence:
    """Test session persistence components"""
    
    def test_session_creation(self):
        """Test session creation and retrieval"""
        session_manager = SessionPersistenceManager(":memory:")
        
        # Create new session
        session = session_manager.get_or_create_session("test_session_123")
        
        assert session.session_id == "test_session_123"
        assert session.query_count == 0
        assert session.total_tokens == 0
        assert session.total_cost == 0.0
        assert isinstance(session.models_used, set)
        assert isinstance(session.categories, set)
    
    def test_session_update(self):
        """Test session data updates"""
        session_manager = SessionPersistenceManager(":memory:")
        
        # Create and update session
        session_manager.get_or_create_session("test_session_123")
        session_manager.update_session("test_session_123", 
                                     query_count=5, 
                                     total_tokens=1000,
                                     total_cost=0.05)
        
        session = session_manager.get_or_create_session("test_session_123")
        assert session.query_count == 5
        assert session.total_tokens == 1000
        assert session.total_cost == 0.05
    
    def test_duplicate_detection(self):
        """Test webhook duplicate detection"""
        session_manager = SessionPersistenceManager(":memory:")
        
        payload1 = {"test": "data", "timestamp": "2024-01-15T10:30:00Z"}
        payload2 = {"test": "data", "timestamp": "2024-01-15T10:30:00Z"}  # Same payload
        payload3 = {"test": "different", "timestamp": "2024-01-15T10:30:00Z"}  # Different payload
        
        # First payload should not be duplicate
        assert not session_manager.is_duplicate_webhook(payload1)
        
        # Second identical payload should be duplicate
        assert session_manager.is_duplicate_webhook(payload2)
        
        # Third different payload should not be duplicate
        assert not session_manager.is_duplicate_webhook(payload3)
    
    def test_webhook_delivery_recording(self):
        """Test webhook delivery recording"""
        session_manager = SessionPersistenceManager(":memory:")
        
        payload = {"test": "data"}
        session_id = "test_session_123"
        
        # Record successful delivery
        session_manager.record_webhook_delivery(session_id, payload, success=True, retry_count=0)
        
        # Record failed delivery
        session_manager.record_webhook_delivery(session_id, payload, success=False, retry_count=2)
        
        # Check session stats
        stats = session_manager.get_session_stats(session_id)
        assert stats is not None
        assert stats["webhook_stats"]["total_deliveries"] == 2
        assert stats["webhook_stats"]["successful_deliveries"] == 1
        assert stats["webhook_stats"]["total_retries"] == 2

class TestWebhookMonitoring:
    """Test webhook monitoring components"""
    
    def test_metrics_collector(self):
        """Test metrics collection"""
        metrics = WebhookMetricsCollector()
        
        # Test counter metrics
        metrics.increment_counter("webhook_success", 5)
        metrics.increment_counter("webhook_failure", 2)
        
        assert metrics.get_counter("webhook_success") == 5
        assert metrics.get_counter("webhook_failure") == 2
        
        # Test gauge metrics
        metrics.set_gauge("avg_response_time", 1500.5)
        assert metrics.get_gauge("avg_response_time") == 1500.5
        
        # Test timing metrics
        metrics.record_timing("webhook_duration", 1200)
        metrics.record_timing("webhook_duration", 1800)
        
        timing_summary = metrics.get_metric_summary("webhook_duration", minutes=60)
        assert timing_summary["count"] == 2
        assert timing_summary["avg"] == 1500.0
    
    def test_alert_manager(self):
        """Test alert management"""
        alert_manager = WebhookAlertManager()
        
        # Create test alert handler
        alerts_received = []
        def test_handler(alert):
            alerts_received.append(alert)
        
        alert_manager.add_alert_handler(test_handler)
        
        # Create alert
        alert = alert_manager.create_alert(
            "test_alert_1",
            "critical",
            "Test critical alert"
        )
        
        assert alert.id == "test_alert_1"
        assert alert.level == "critical"
        assert alert.message == "Test critical alert"
        assert not alert.resolved
        
        # Check handler was called
        assert len(alerts_received) == 1
        assert alerts_received[0].id == "test_alert_1"
        
        # Resolve alert
        alert_manager.resolve_alert("test_alert_1")
        assert alert.resolved
        assert alert.resolved_at is not None
        
        # Check active alerts
        active_alerts = alert_manager.get_active_alerts()
        assert len(active_alerts) == 0

class TestWebhookSecurity:
    """Test webhook security components"""
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        # Test valid configuration
        validation = WebhookConfigurationValidator.validate_environment()
        # This will depend on actual environment variables
        
        assert "valid" in validation
        assert "errors" in validation
        assert "warnings" in validation
    
    def test_security_config_validation(self):
        """Test security configuration validation"""
        from webhook_security import SecurityConfig, RateLimitConfig
        
        # Test valid security config
        config = SecurityConfig(
            webhook_secret="test_secret_123456789",
            signature_algorithm="sha256",
            rate_limits={
                "webhook_calls": RateLimitConfig(max_requests=100, window_seconds=3600)
            }
        )
        
        validation = WebhookConfigurationValidator.validate_security_config(config)
        assert validation["valid"] is True
        assert len(validation["errors"]) == 0
    
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        from webhook_security import WebhookRateLimiter, RateLimitConfig
        
        rate_limiter = WebhookRateLimiter()
        
        # Add rate limit
        config = RateLimitConfig(max_requests=2, window_seconds=60)
        rate_limiter.add_rate_limit("test_limit", config)
        
        # Test rate limiting
        assert rate_limiter.is_allowed("test_limit", "user1") is True
        assert rate_limiter.is_allowed("test_limit", "user1") is True
        assert rate_limiter.is_allowed("test_limit", "user1") is False  # Should be rate limited
        
        # Test different user
        assert rate_limiter.is_allowed("test_limit", "user2") is True
        
        # Test remaining requests
        remaining = rate_limiter.get_remaining_requests("test_limit", "user1")
        assert remaining == 0

class TestIntegration:
    """Test integration between components"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_webhook_flow(self):
        """Test complete webhook flow"""
        # This would test the full integration
        # For now, we'll test individual components work together
        
        # Create session manager
        session_manager = SessionPersistenceManager(":memory:")
        
        # Create metrics collector
        metrics = WebhookMetricsCollector()
        
        # Test session creation and metrics
        session_id = "integration_test_session"
        session = session_manager.get_or_create_session(session_id)
        
        # Simulate webhook tracking
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "query": {"summary": "Test query", "category": "general", "estimated_complexity": 1},
            "response": {"model_used": "gpt-4", "response_time_ms": 1000, "estimated_tokens": 100, "success": True}
        }
        
        # Check for duplicates
        is_duplicate = session_manager.is_duplicate_webhook(payload)
        assert not is_duplicate
        
        # Record metrics
        metrics.increment_counter("webhook_total")
        metrics.increment_counter("webhook_success")
        
        # Record delivery
        session_manager.record_webhook_delivery(session_id, payload, success=True)
        
        # Verify metrics
        assert metrics.get_counter("webhook_total") == 1
        assert metrics.get_counter("webhook_success") == 1
        
        # Verify session stats
        stats = session_manager.get_session_stats(session_id)
        assert stats["webhook_stats"]["total_deliveries"] == 1
        assert stats["webhook_stats"]["successful_deliveries"] == 1

# Test fixtures
@pytest.fixture
def sample_usage_payload():
    """Sample usage payload for testing"""
    return UsagePayload(
        timestamp="2024-01-15T10:30:00Z",
        session_id="test_session_123",
        query=QueryData(
            summary="Test query for webhook integration",
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

@pytest.fixture
def sample_webhook_config():
    """Sample webhook configuration for testing"""
    return {
        "webhook_url": "https://hooks.zapier.com/test",
        "enabled": True,
        "max_retries": 3,
        "timeout_seconds": 10
    }

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
