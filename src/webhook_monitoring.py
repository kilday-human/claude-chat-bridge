#!/usr/bin/env python3
"""
Webhook Monitoring and Observability System
Tracks metrics, implements alerting, and provides health dashboards
"""

import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import logging
import threading
from enum import Enum

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class Alert:
    id: str
    level: AlertLevel
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None

@dataclass
class MetricPoint:
    timestamp: datetime
    value: float
    tags: Dict[str, str]

class WebhookMetricsCollector:
    """Collects and aggregates webhook metrics"""
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.metrics = defaultdict(lambda: deque(maxlen=1000))  # Keep last 1000 points
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self._lock = threading.Lock()
    
    def increment_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """Increment a counter metric"""
        with self._lock:
            self.counters[name] += value
            self._record_metric_point(f"{name}_total", value, tags)
    
    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge metric"""
        with self._lock:
            self.gauges[name] = value
            self._record_metric_point(name, value, tags)
    
    def record_timing(self, name: str, duration_ms: float, tags: Dict[str, str] = None):
        """Record a timing metric"""
        with self._lock:
            self._record_metric_point(f"{name}_duration_ms", duration_ms, tags)
            self._record_metric_point(f"{name}_count", 1, tags)
    
    def _record_metric_point(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a metric point with timestamp"""
        point = MetricPoint(
            timestamp=datetime.utcnow(),
            value=value,
            tags=tags or {}
        )
        self.metrics[name].append(point)
    
    def get_counter(self, name: str) -> int:
        """Get current counter value"""
        with self._lock:
            return self.counters[name]
    
    def get_gauge(self, name: str) -> float:
        """Get current gauge value"""
        with self._lock:
            return self.gauges[name]
    
    def get_metric_series(self, name: str, minutes: int = 60) -> List[MetricPoint]:
        """Get metric series for the last N minutes"""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        
        with self._lock:
            return [point for point in self.metrics[name] if point.timestamp >= cutoff]
    
    def get_metric_summary(self, name: str, minutes: int = 60) -> Dict[str, Any]:
        """Get metric summary (min, max, avg, count) for the last N minutes"""
        series = self.get_metric_series(name, minutes)
        
        if not series:
            return {"count": 0, "min": 0, "max": 0, "avg": 0}
        
        values = [point.value for point in series]
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values)
        }

class WebhookAlertManager:
    """Manages webhook alerts and notifications"""
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self._lock = threading.Lock()
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add alert handler function"""
        self.alert_handlers.append(handler)
    
    def create_alert(self, alert_id: str, level: AlertLevel, message: str) -> Alert:
        """Create a new alert"""
        with self._lock:
            alert = Alert(
                id=alert_id,
                level=level,
                message=message,
                timestamp=datetime.utcnow()
            )
            self.alerts[alert_id] = alert
            
            # Notify handlers
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Alert handler failed: {e}")
            
            logger.warning(f"Alert created: {level.value.upper()} - {message}")
            return alert
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        with self._lock:
            if alert_id in self.alerts:
                self.alerts[alert_id].resolved = True
                self.alerts[alert_id].resolved_at = datetime.utcnow()
                logger.info(f"Alert resolved: {alert_id}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts"""
        with self._lock:
            return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def get_alerts_by_level(self, level: AlertLevel) -> List[Alert]:
        """Get alerts by level"""
        with self._lock:
            return [alert for alert in self.alerts.values() if alert.level == level and not alert.resolved]

class WebhookHealthChecker:
    """Performs health checks and generates alerts"""
    
    def __init__(self, metrics_collector: WebhookMetricsCollector, alert_manager: WebhookAlertManager):
        self.metrics = metrics_collector
        self.alerts = alert_manager
        self.last_health_check = None
        self._check_interval = 60  # Check every 60 seconds
        self._running = False
        self._thread = None
    
    def start_health_checks(self):
        """Start background health checking"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self._thread.start()
        logger.info("Health checker started")
    
    def stop_health_checks(self):
        """Stop background health checking"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Health checker stopped")
    
    def _health_check_loop(self):
        """Main health check loop"""
        while self._running:
            try:
                self._perform_health_checks()
                time.sleep(self._check_interval)
            except Exception as e:
                logger.error(f"Health check error: {e}")
                time.sleep(10)  # Short sleep on error
    
    def _perform_health_checks(self):
        """Perform all health checks"""
        self.last_health_check = datetime.utcnow()
        
        # Check webhook success rate
        self._check_success_rate()
        
        # Check response times
        self._check_response_times()
        
        # Check error rates
        self._check_error_rates()
        
        # Check dead letter queue
        self._check_dead_letter_queue()
    
    def _check_success_rate(self):
        """Check webhook success rate"""
        success_count = self.metrics.get_counter("webhook_success")
        failure_count = self.metrics.get_counter("webhook_failure")
        total_count = success_count + failure_count
        
        if total_count > 0:
            success_rate = success_count / total_count
            
            if success_rate < 0.8:  # Less than 80% success rate
                self.alerts.create_alert(
                    "webhook_success_rate_low",
                    AlertLevel.CRITICAL,
                    f"Webhook success rate is {success_rate:.1%} (below 80% threshold)"
                )
            elif success_rate < 0.9:  # Less than 90% success rate
                self.alerts.create_alert(
                    "webhook_success_rate_warning",
                    AlertLevel.WARNING,
                    f"Webhook success rate is {success_rate:.1%} (below 90% threshold)"
                )
            else:
                # Resolve alerts if success rate is good
                self.alerts.resolve_alert("webhook_success_rate_low")
                self.alerts.resolve_alert("webhook_success_rate_warning")
    
    def _check_response_times(self):
        """Check webhook response times"""
        timing_summary = self.metrics.get_metric_summary("webhook_duration_ms", minutes=5)
        
        if timing_summary["count"] > 0:
            avg_response_time = timing_summary["avg"]
            
            if avg_response_time > 10000:  # More than 10 seconds
                self.alerts.create_alert(
                    "webhook_response_time_high",
                    AlertLevel.CRITICAL,
                    f"Average webhook response time is {avg_response_time:.0f}ms (above 10s threshold)"
                )
            elif avg_response_time > 5000:  # More than 5 seconds
                self.alerts.create_alert(
                    "webhook_response_time_warning",
                    AlertLevel.WARNING,
                    f"Average webhook response time is {avg_response_time:.0f}ms (above 5s threshold)"
                )
            else:
                # Resolve alerts if response times are good
                self.alerts.resolve_alert("webhook_response_time_high")
                self.alerts.resolve_alert("webhook_response_time_warning")
    
    def _check_error_rates(self):
        """Check error rates"""
        error_count = self.metrics.get_counter("webhook_error")
        total_count = self.metrics.get_counter("webhook_total")
        
        if total_count > 0:
            error_rate = error_count / total_count
            
            if error_rate > 0.1:  # More than 10% error rate
                self.alerts.create_alert(
                    "webhook_error_rate_high",
                    AlertLevel.CRITICAL,
                    f"Webhook error rate is {error_rate:.1%} (above 10% threshold)"
                )
            else:
                self.alerts.resolve_alert("webhook_error_rate_high")
    
    def _check_dead_letter_queue(self):
        """Check dead letter queue size"""
        dlq_size = self.metrics.get_gauge("dead_letter_queue_size")
        
        if dlq_size > 100:  # More than 100 messages in DLQ
            self.alerts.create_alert(
                "dead_letter_queue_large",
                AlertLevel.CRITICAL,
                f"Dead letter queue has {dlq_size} messages (above 100 threshold)"
            )
        elif dlq_size > 50:  # More than 50 messages in DLQ
            self.alerts.create_alert(
                "dead_letter_queue_warning",
                AlertLevel.WARNING,
                f"Dead letter queue has {dlq_size} messages (above 50 threshold)"
            )
        else:
            # Resolve alerts if DLQ is manageable
            self.alerts.resolve_alert("dead_letter_queue_large")
            self.alerts.resolve_alert("dead_letter_queue_warning")

class WebhookDashboard:
    """Provides webhook health dashboard data"""
    
    def __init__(self, metrics_collector: WebhookMetricsCollector, alert_manager: WebhookAlertManager):
        self.metrics = metrics_collector
        self.alerts = alert_manager
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary"""
        success_count = self.metrics.get_counter("webhook_success")
        failure_count = self.metrics.get_counter("webhook_failure")
        total_count = success_count + failure_count
        
        success_rate = success_count / max(total_count, 1)
        
        active_alerts = self.alerts.get_active_alerts()
        critical_alerts = [a for a in active_alerts if a.level == AlertLevel.CRITICAL]
        warning_alerts = [a for a in active_alerts if a.level == AlertLevel.WARNING]
        
        return {
            "status": "healthy" if not critical_alerts else "critical",
            "success_rate": success_rate,
            "total_webhooks": total_count,
            "active_alerts": {
                "critical": len(critical_alerts),
                "warning": len(warning_alerts),
                "total": len(active_alerts)
            },
            "last_updated": datetime.utcnow().isoformat()
        }
    
    def get_metrics_dashboard(self) -> Dict[str, Any]:
        """Get detailed metrics for dashboard"""
        return {
            "counters": dict(self.metrics.counters),
            "gauges": dict(self.metrics.gauges),
            "response_times": self.metrics.get_metric_summary("webhook_duration_ms", minutes=60),
            "success_rate_5min": self._calculate_success_rate(5),
            "success_rate_1hour": self._calculate_success_rate(60),
            "success_rate_24hour": self._calculate_success_rate(1440)
        }
    
    def _calculate_success_rate(self, minutes: int) -> float:
        """Calculate success rate for given time period"""
        success_series = self.metrics.get_metric_series("webhook_success", minutes)
        failure_series = self.metrics.get_metric_series("webhook_failure", minutes)
        
        success_count = sum(point.value for point in success_series)
        failure_count = sum(point.value for point in failure_series)
        total_count = success_count + failure_count
        
        return success_count / max(total_count, 1)
    
    def get_alerts_dashboard(self) -> Dict[str, Any]:
        """Get alerts for dashboard"""
        active_alerts = self.alerts.get_active_alerts()
        
        return {
            "active_alerts": [asdict(alert) for alert in active_alerts],
            "alert_counts": {
                "critical": len([a for a in active_alerts if a.level == AlertLevel.CRITICAL]),
                "warning": len([a for a in active_alerts if a.level == AlertLevel.WARNING]),
                "info": len([a for a in active_alerts if a.level == AlertLevel.INFO])
            }
        }

# Example alert handlers
def console_alert_handler(alert: Alert):
    """Simple console alert handler"""
    print(f"🚨 ALERT [{alert.level.value.upper()}] {alert.message}")

def slack_alert_handler(alert: Alert):
    """Slack alert handler (placeholder)"""
    # In production, this would send to Slack
    logger.info(f"Slack alert: {alert.level.value.upper()} - {alert.message}")

def email_alert_handler(alert: Alert):
    """Email alert handler (placeholder)"""
    # In production, this would send email
    logger.info(f"Email alert: {alert.level.value.upper()} - {alert.message}")
