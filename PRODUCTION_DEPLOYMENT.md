# Production Deployment Guide - Zapier Integration

## 🚀 Production-Ready Webhook System

This guide covers deploying the production-ready Zapier integration for GPT-Bridge with full reliability, monitoring, and security features.

## 📋 Pre-Deployment Checklist

### ✅ Core Features Implemented

- **Webhook Reliability**: Retry logic with exponential backoff, dead letter queue
- **Data Persistence**: Session storage, duplicate detection, data retention
- **Monitoring & Observability**: Metrics collection, alerting, health checks
- **Security & Configuration**: Signature verification, rate limiting, config validation
- **TypeScript Implementation**: Full TypeScript types and interfaces
- **Comprehensive Testing**: Unit tests, integration tests, load testing

### ✅ Production Components

1. **WebhookReliability** (`src/webhook_reliability.py`)
   - Retry logic with exponential backoff
   - Dead letter queue for failed webhooks
   - Endpoint validation and health checks

2. **SessionPersistence** (`src/session_persistence.py`)
   - SQLite-based session storage
   - Duplicate webhook detection
   - Data retention and cleanup

3. **WebhookMonitoring** (`src/webhook_monitoring.py`)
   - Metrics collection and aggregation
   - Alert management system
   - Health checking and dashboard

4. **WebhookSecurity** (`src/webhook_security.py`)
   - Signature verification
   - Rate limiting
   - Configuration validation

5. **ProductionIntegration** (`src/production_webhook_integration.py`)
   - Unified production interface
   - CLI management tools
   - Health status and metrics

## 🔧 Environment Configuration

### Required Environment Variables

```bash
# Core Configuration
ZAPIER_WEBHOOK_URL=https://hooks.zapier.com/hooks/catch/your-webhook-id
ENABLE_WEBHOOK=true

# Reliability Configuration
WEBHOOK_MAX_RETRIES=3
WEBHOOK_TIMEOUT_SECONDS=10
WEBHOOK_RETRY_DELAY_SECONDS=60

# Security Configuration
WEBHOOK_SECRET=your-webhook-secret-key
WEBHOOK_SIGNATURE_HEADER=X-Webhook-Signature
WEBHOOK_SIGNATURE_ALGORITHM=sha256
WEBHOOK_RATE_LIMIT=100
WEBHOOK_RATE_WINDOW=3600
WEBHOOK_BURST_ALLOWANCE=10

# Optional Security
WEBHOOK_ALLOWED_IPS=192.168.1.0/24,10.0.0.0/8
REQUIRE_HTTPS=true

# Monitoring Configuration
SENTRY_DSN=your-sentry-dsn
LOG_LEVEL=INFO
```

### Environment Validation

```bash
# Validate configuration
python3 -c "from src.webhook_security import WebhookConfigurationValidator; print(WebhookConfigurationValidator.validate_environment())"
```

## 🚀 Deployment Steps

### 1. Install Dependencies

```bash
# Install production dependencies
pip install -r requirements.txt

# Install additional production dependencies
pip install sentry-sdk[fastapi] prometheus-client
```

### 2. Database Setup

```bash
# Initialize databases (automatic on first run)
# Sessions database: sessions.db
# Dead letter queue: webhook_dlq.db
# Metrics database: webhook_metrics.db
```

### 3. Webhook Endpoint Setup

#### Zapier Webhook Configuration

1. **Create Zapier Webhook**:
   - Go to Zapier.com → Create Zap
   - Choose "Webhooks by Zapier" → "Catch Hook"
   - Copy the webhook URL

2. **Configure Actions**:
   - **Google Sheets**: Log usage data
   - **Slack**: Send notifications
   - **Email**: Alert on failures

3. **Test Webhook**:
   ```bash
   # Test webhook endpoint
   python3 -c "
   import asyncio
   from src.production_webhook_integration import get_production_webhook
   webhook = get_production_webhook()
   result = asyncio.run(webhook.webhook_client.validate_endpoint())
   print(f'Endpoint validation: {result}')
   "
   ```

### 4. Security Configuration

#### Webhook Signature Verification

```bash
# Generate webhook secret
WEBHOOK_SECRET=$(openssl rand -hex 32)
echo "WEBHOOK_SECRET=$WEBHOOK_SECRET" >> .env

# Configure Zapier to use signature verification
# Add X-Webhook-Signature header with HMAC-SHA256 signature
```

#### Rate Limiting

```bash
# Configure rate limits
WEBHOOK_RATE_LIMIT=100        # 100 requests per hour
WEBHOOK_RATE_WINDOW=3600      # 1 hour window
WEBHOOK_BURST_ALLOWANCE=10    # Allow 10 burst requests
```

### 5. Monitoring Setup

#### Sentry Integration

```bash
# Add Sentry DSN
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id
```

#### Prometheus Metrics (Optional)

```bash
# Install Prometheus client
pip install prometheus-client

# Expose metrics endpoint
python3 -c "
from src.webhook_monitoring import WebhookMetricsCollector
from prometheus_client import start_http_server, Gauge, Counter

# Start Prometheus metrics server
start_http_server(8000)
"
```

### 6. Production Testing

#### Unit Tests

```bash
# Run unit tests
python3 -m pytest tests/test_webhook_integration.py -v
```

#### Integration Tests

```bash
# Run integration tests (requires test webhook server)
python3 -m pytest tests/test_webhook_integration_real.py -v
```

#### Load Testing

```bash
# Run load tests
python3 -c "
import asyncio
from tests.test_webhook_integration_real import TestWebhookLoadTesting
test = TestWebhookLoadTesting()
asyncio.run(test.test_webhook_high_volume('http://localhost:8000/api/webhook/usage'))
"
```

## 📊 Monitoring and Observability

### Health Dashboard

```bash
# Check webhook health
python3 -c "
from src.production_webhook_integration import get_production_webhook
webhook = get_production_webhook()
health = webhook.get_health_status()
print(f'Status: {health[\"health\"][\"status\"]}')
print(f'Success Rate: {health[\"metrics\"][\"success_rate_1hour\"]:.1%}')
print(f'Active Alerts: {health[\"alerts\"][\"alert_counts\"][\"total\"]}')
"
```

### Metrics Collection

```bash
# View webhook metrics
python3 -c "
from src.production_webhook_integration import get_production_webhook
webhook = get_production_webhook()
metrics = webhook.get_metrics()
print(f'Total Webhooks: {metrics[\"total_sent\"]}')
print(f'Success Rate: {metrics[\"success_rate\"]:.1%}')
print(f'Avg Response Time: {metrics[\"avg_response_time_ms\"]:.0f}ms')
"
```

### Alert Management

```bash
# Check active alerts
python3 -c "
from src.production_webhook_integration import get_production_webhook
webhook = get_production_webhook()
alerts = webhook.alert_manager.get_active_alerts()
for alert in alerts:
    print(f'{alert.level.value.upper()}: {alert.message}')
"
```

## 🔧 Maintenance and Operations

### Data Cleanup

```bash
# Clean up old data (retention: 30 days)
python3 -c "
from src.production_webhook_integration import get_production_webhook
webhook = get_production_webhook()
webhook.cleanup_old_data(30)
print('Data cleanup completed')
"
```

### Dead Letter Queue Management

```bash
# View dead letter queue
python3 -c "
from src.webhook_reliability import WebhookDeadLetterQueue
dlq = WebhookDeadLetterQueue()
messages = dlq.get_messages(limit=10)
print(f'Dead letter queue has {len(messages)} messages')
for msg in messages[:5]:
    print(f'  {msg.id}: {msg.attempts[-1].error_message}')
"
```

### Session Analytics

```bash
# View session analytics
python3 -c "
from src.production_webhook_integration import get_production_webhook
webhook = get_production_webhook()
analytics = webhook.session_manager.get_analytics_summary()
print(f'Total Sessions (7 days): {analytics[\"sessions\"][\"total\"]}')
print(f'Total Queries: {analytics[\"sessions\"][\"total_queries\"]}')
print(f'Webhook Success Rate: {analytics[\"webhooks\"][\"success_rate\"]:.1%}')
"
```

## 🚨 Troubleshooting

### Common Issues

1. **Webhook Endpoint Not Responding**
   ```bash
   # Check endpoint health
   python3 -c "
   from src.webhook_reliability import WebhookEndpointValidator
   import asyncio
   result = asyncio.run(WebhookEndpointValidator.validate_endpoint('YOUR_WEBHOOK_URL'))
   print(result)
   "
   ```

2. **High Failure Rate**
   ```bash
   # Check metrics and alerts
   python3 -c "
   from src.production_webhook_integration import get_production_webhook
   webhook = get_production_webhook()
   health = webhook.get_health_status()
   print('Health:', health['health'])
   print('Alerts:', health['alerts']['active_alerts'])
   "
   ```

3. **Rate Limiting Issues**
   ```bash
   # Check rate limit status
   python3 -c "
   from src.webhook_security import get_default_security_config
   from src.webhook_security import WebhookSecurityManager
   config = get_default_security_config()
   security = WebhookSecurityManager(config)
   status = security.get_rate_limit_status('default')
   print('Rate Limit Status:', status)
   "
   ```

### Log Analysis

```bash
# View webhook logs
tail -f logs/webhook.log | grep -E "(ERROR|WARNING|webhook)"

# Check specific session
grep "session_123" logs/webhook.log

# Monitor success rate
grep "webhook_success" logs/webhook.log | wc -l
```

## 📈 Performance Optimization

### Database Optimization

```sql
-- Optimize SQLite databases
VACUUM;
ANALYZE;

-- Check database sizes
SELECT name, page_count * page_size as size FROM pragma_page_count(), pragma_page_size();
```

### Memory Management

```bash
# Monitor memory usage
python3 -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"
```

### Network Optimization

```bash
# Test webhook response times
python3 -c "
import time
import requests
start = time.time()
response = requests.post('YOUR_WEBHOOK_URL', json={'test': True}, timeout=10)
print(f'Response time: {(time.time() - start) * 1000:.0f}ms')
print(f'Status: {response.status_code}')
"
```

## 🔒 Security Best Practices

1. **Use HTTPS**: Always use HTTPS for webhook URLs
2. **Signature Verification**: Enable webhook signature verification
3. **Rate Limiting**: Configure appropriate rate limits
4. **IP Allowlisting**: Restrict webhook access to known IPs
5. **Secret Management**: Use secure secret management
6. **Regular Updates**: Keep dependencies updated
7. **Monitoring**: Monitor for security alerts

## 📚 API Reference

### Production Webhook Client

```python
from src.production_webhook_integration import get_production_webhook

# Get webhook instance
webhook = get_production_webhook()

# Track usage
await webhook.track_usage(
    session_id="session_123",
    query="Test query",
    model_name="gpt-4o-mini",
    response_time_ms=1000,
    estimated_tokens=100,
    success=True
)

# Get health status
health = webhook.get_health_status()

# Get metrics
metrics = webhook.get_metrics()

# Get session stats
stats = webhook.get_session_stats("session_123")
```

### CLI Management

```bash
# Webhook status
python3 src/production_webhook_integration.py --status

# Webhook metrics
python3 src/production_webhook_integration.py --metrics

# Health check
python3 src/production_webhook_integration.py --health

# Test endpoint
python3 src/production_webhook_integration.py --test

# Cleanup data
python3 src/production_webhook_integration.py --cleanup 30
```

## ✅ Production Readiness Checklist

- [ ] Environment variables configured
- [ ] Webhook endpoint validated
- [ ] Security configuration enabled
- [ ] Monitoring and alerting setup
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Load testing completed
- [ ] Documentation reviewed
- [ ] Backup strategy implemented
- [ ] Incident response plan ready

## 🎯 Success Metrics

- **Reliability**: >99% webhook delivery success rate
- **Performance**: <5 second average response time
- **Security**: Zero security incidents
- **Monitoring**: 100% alert coverage
- **Data Quality**: <1% duplicate webhook rate

---

**Production deployment complete!** 🚀

The webhook integration is now production-ready with full reliability, monitoring, security, and observability features.
