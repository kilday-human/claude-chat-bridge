#!/usr/bin/env python3
"""
Production-Ready Webhook Integration for GPT-Bridge
Integrates all reliability, monitoring, security, and persistence components
"""

import os
import asyncio
import threading
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# Import all production components
from .webhook_reliability import ProductionWebhookClient, WebhookRetryManager
from .session_persistence import SessionPersistenceManager
from .webhook_monitoring import WebhookMetricsCollector, WebhookAlertManager, WebhookHealthChecker, WebhookDashboard
from .webhook_security import get_default_security_config, WebhookSecurityManager
from .webhook_integration import UsagePayload, QueryData, ResponseData, UsageTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionWebhookIntegration:
    """Production-ready webhook integration with all reliability features"""
    
    def __init__(self):
        self.webhook_url = os.getenv('ZAPIER_WEBHOOK_URL')
        self.enabled = os.getenv('ENABLE_WEBHOOK', 'false').lower() == 'true'
        
        # Initialize components
        self.session_manager = SessionPersistenceManager()
        self.metrics_collector = WebhookMetricsCollector()
        self.alert_manager = WebhookAlertManager()
        self.security_config = get_default_security_config()
        self.security_manager = WebhookSecurityManager(self.security_config)
        
        # Initialize webhook client if enabled
        self.webhook_client = None
        if self.enabled and self.webhook_url:
            self.webhook_client = ProductionWebhookClient(self.webhook_url)
        
        # Initialize health checker
        self.health_checker = WebhookHealthChecker(self.metrics_collector, self.alert_manager)
        self.dashboard = WebhookDashboard(self.metrics_collector, self.alert_manager)
        
        # Set up alert handlers
        self._setup_alert_handlers()
        
        # Start health checking
        self.health_checker.start_health_checks()
        
        logger.info("Production webhook integration initialized")
    
    def _setup_alert_handlers(self):
        """Set up alert handlers for monitoring"""
        def console_alert_handler(alert):
            logger.warning(f"🚨 ALERT [{alert.level.value.upper()}] {alert.message}")
        
        def metrics_alert_handler(alert):
            # Update metrics for alert tracking
            self.metrics_collector.increment_counter(f"alerts_{alert.level.value}")
        
        self.alert_manager.add_alert_handler(console_alert_handler)
        self.alert_manager.add_alert_handler(metrics_alert_handler)
    
    async def track_usage(self, 
                         session_id: str,
                         query: str, 
                         model_name: str,
                         response_time_ms: int,
                         estimated_tokens: int,
                         success: bool,
                         user_agent: Optional[str] = None) -> bool:
        """
        Track GPT-Bridge usage with full production reliability
        """
        if not self.enabled or not self.webhook_client:
            logger.info("Webhook tracking disabled")
            return False
        
        try:
            # Get or create session
            session = self.session_manager.get_or_create_session(session_id)
            
            # Create usage payload
            usage_tracker = UsageTracker()
            payload = UsagePayload(
                timestamp=datetime.utcnow().isoformat() + 'Z',
                session_id=session_id,
                query=QueryData(
                    summary=query[:50] + ('...' if len(query) > 50 else ''),
                    category=usage_tracker.categorize_query(query),
                    estimated_complexity=usage_tracker.estimate_complexity(query, response_time_ms)
                ),
                response=ResponseData(
                    model_used=usage_tracker.determine_model_type(model_name),
                    response_time_ms=response_time_ms,
                    estimated_tokens=estimated_tokens,
                    success=success
                ),
                user_agent=user_agent or "GPT-Bridge-Production"
            )
            
            # Check for duplicates
            if self.session_manager.is_duplicate_webhook(payload.__dict__):
                logger.info(f"Duplicate webhook detected for session {session_id}")
                return True
            
            # Record metrics
            self.metrics_collector.increment_counter("webhook_total")
            self.metrics_collector.record_timing("webhook_duration", response_time_ms)
            
            # Send webhook
            webhook_success = await self.webhook_client.send_webhook(payload.__dict__)
            
            # Update metrics
            if webhook_success:
                self.metrics_collector.increment_counter("webhook_success")
                self.metrics_collector.set_gauge("last_success", datetime.utcnow().timestamp())
            else:
                self.metrics_collector.increment_counter("webhook_failure")
                self.metrics_collector.set_gauge("last_failure", datetime.utcnow().timestamp())
            
            # Record delivery
            self.session_manager.record_webhook_delivery(
                session_id, 
                payload.__dict__, 
                webhook_success
            )
            
            # Update session
            self.session_manager.update_session(session_id, 
                query_count=session.query_count + 1,
                total_tokens=session.total_tokens + estimated_tokens,
                models_used=session.models_used | {model_name},
                categories=session.categories | {payload.query.category}
            )
            
            logger.info(f"Webhook tracking completed for session {session_id}: {'success' if webhook_success else 'failed'}")
            return webhook_success
            
        except Exception as e:
            logger.error(f"Webhook tracking error: {e}")
            self.metrics_collector.increment_counter("webhook_error")
            return False
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        health_summary = self.dashboard.get_health_summary()
        metrics_dashboard = self.dashboard.get_metrics_dashboard()
        alerts_dashboard = self.dashboard.get_alerts_dashboard()
        
        return {
            "webhook_integration": {
                "enabled": self.enabled,
                "webhook_url": self.webhook_url,
                "endpoint_validated": self.webhook_client.endpoint_validated if self.webhook_client else False
            },
            "health": health_summary,
            "metrics": metrics_dashboard,
            "alerts": alerts_dashboard,
            "session_stats": self.session_manager.get_analytics_summary()
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get webhook metrics"""
        return self.metrics_collector.get_metrics()
    
    def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session statistics"""
        return self.session_manager.get_session_stats(session_id)
    
    def cleanup_old_data(self, retention_days: int = 30):
        """Clean up old data"""
        self.session_manager.cleanup_old_data(retention_days)
        logger.info(f"Cleaned up data older than {retention_days} days")
    
    def shutdown(self):
        """Shutdown webhook integration"""
        self.health_checker.stop_health_checks()
        logger.info("Webhook integration shutdown complete")

# Global production webhook integration instance
_production_webhook = None

def get_production_webhook() -> ProductionWebhookIntegration:
    """Get or create global production webhook integration"""
    global _production_webhook
    if _production_webhook is None:
        _production_webhook = ProductionWebhookIntegration()
    return _production_webhook

# Convenience function for easy integration
async def track_gpt_bridge_usage_production(session_id: str,
                                          query: str,
                                          model_name: str, 
                                          response_time_ms: int,
                                          estimated_tokens: int,
                                          success: bool,
                                          user_agent: Optional[str] = None) -> bool:
    """
    Production-ready convenience function to track GPT-Bridge usage
    """
    webhook_integration = get_production_webhook()
    return await webhook_integration.track_usage(
        session_id=session_id,
        query=query,
        model_name=model_name,
        response_time_ms=response_time_ms,
        estimated_tokens=estimated_tokens,
        success=success,
        user_agent=user_agent
    )

# CLI interface for webhook management
def webhook_cli():
    """CLI interface for webhook management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Webhook Integration CLI")
    parser.add_argument("--status", action="store_true", help="Show webhook status")
    parser.add_argument("--metrics", action="store_true", help="Show webhook metrics")
    parser.add_argument("--health", action="store_true", help="Show health status")
    parser.add_argument("--cleanup", type=int, metavar="DAYS", help="Clean up data older than N days")
    parser.add_argument("--test", action="store_true", help="Test webhook endpoint")
    
    args = parser.parse_args()
    
    webhook = get_production_webhook()
    
    if args.status:
        status = webhook.get_health_status()
        print(json.dumps(status, indent=2))
    
    elif args.metrics:
        metrics = webhook.get_metrics()
        print(json.dumps(metrics, indent=2))
    
    elif args.health:
        health = webhook.get_health_status()
        print(f"Status: {health['health']['status']}")
        print(f"Success Rate: {health['metrics']['success_rate_1hour']:.1%}")
        print(f"Active Alerts: {health['alerts']['alert_counts']['total']}")
    
    elif args.cleanup:
        webhook.cleanup_old_data(args.cleanup)
        print(f"Cleaned up data older than {args.cleanup} days")
    
    elif args.test:
        if webhook.webhook_client:
            is_valid = asyncio.run(webhook.webhook_client.validate_endpoint())
            print(f"Endpoint validation: {'PASS' if is_valid else 'FAIL'}")
        else:
            print("Webhook client not initialized")

if __name__ == "__main__":
    webhook_cli()
