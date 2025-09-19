#!/usr/bin/env python3
"""
Production-Ready Webhook Reliability System
Handles retry logic, dead letter queue, and endpoint validation
"""

import asyncio
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import aiohttp
import sqlite3
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebhookStatus(Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"

@dataclass
class WebhookAttempt:
    attempt_number: int
    timestamp: datetime
    status: WebhookStatus
    error_message: Optional[str] = None
    response_code: Optional[int] = None
    response_time_ms: Optional[int] = None

@dataclass
class WebhookMessage:
    id: str
    payload: Dict[str, Any]
    webhook_url: str
    created_at: datetime
    attempts: List[WebhookAttempt]
    max_retries: int = 3
    retry_delay_seconds: int = 60
    status: WebhookStatus = WebhookStatus.PENDING
    
    def add_attempt(self, attempt: WebhookAttempt):
        self.attempts.append(attempt)
        self.status = attempt.status

class WebhookDeadLetterQueue:
    """Handles permanently failed webhooks with persistence"""
    
    def __init__(self, db_path: str = "webhook_dlq.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database for dead letter queue"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dead_letter_queue (
                    id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    webhook_url TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    failed_at TEXT NOT NULL,
                    attempts TEXT NOT NULL,
                    error_message TEXT
                )
            """)
    
    def add_message(self, message: WebhookMessage, final_error: str):
        """Add message to dead letter queue"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO dead_letter_queue 
                (id, payload, webhook_url, created_at, failed_at, attempts, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                message.id,
                json.dumps(message.payload),
                message.webhook_url,
                message.created_at.isoformat(),
                datetime.utcnow().isoformat(),
                json.dumps([asdict(attempt) for attempt in message.attempts]),
                final_error
            ))
        logger.warning(f"Message {message.id} moved to dead letter queue: {final_error}")
    
    def get_messages(self, limit: int = 100) -> List[WebhookMessage]:
        """Retrieve messages from dead letter queue"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, payload, webhook_url, created_at, failed_at, attempts, error_message
                FROM dead_letter_queue
                ORDER BY failed_at DESC
                LIMIT ?
            """, (limit,))
            
            messages = []
            for row in cursor.fetchall():
                message = WebhookMessage(
                    id=row[0],
                    payload=json.loads(row[1]),
                    webhook_url=row[2],
                    created_at=datetime.fromisoformat(row[3]),
                    attempts=[WebhookAttempt(**attempt) for attempt in json.loads(row[4])],
                    status=WebhookStatus.DEAD_LETTER
                )
                messages.append(message)
            return messages

class WebhookRetryManager:
    """Handles retry logic with exponential backoff"""
    
    def __init__(self, max_retries: int = 3, base_delay: int = 1, max_delay: int = 300):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.dlq = WebhookDeadLetterQueue()
    
    def calculate_delay(self, attempt_number: int) -> int:
        """Calculate exponential backoff delay"""
        delay = min(self.base_delay * (2 ** (attempt_number - 1)), self.max_delay)
        return delay
    
    async def send_with_retry(self, message: WebhookMessage) -> bool:
        """Send webhook with retry logic and exponential backoff"""
        
        for attempt_num in range(1, self.max_retries + 1):
            try:
                # Calculate delay for this attempt
                if attempt_num > 1:
                    delay = self.calculate_delay(attempt_num)
                    logger.info(f"Retrying webhook {message.id} in {delay} seconds (attempt {attempt_num})")
                    await asyncio.sleep(delay)
                
                # Attempt to send webhook
                start_time = time.time()
                success, error_msg, response_code = await self._send_webhook(message)
                response_time = int((time.time() - start_time) * 1000)
                
                # Record attempt
                attempt = WebhookAttempt(
                    attempt_number=attempt_num,
                    timestamp=datetime.utcnow(),
                    status=WebhookStatus.SUCCESS if success else WebhookStatus.FAILED,
                    error_message=error_msg,
                    response_code=response_code,
                    response_time_ms=response_time
                )
                message.add_attempt(attempt)
                
                if success:
                    logger.info(f"Webhook {message.id} sent successfully on attempt {attempt_num}")
                    return True
                else:
                    logger.warning(f"Webhook {message.id} failed on attempt {attempt_num}: {error_msg}")
                    
            except Exception as e:
                logger.error(f"Unexpected error sending webhook {message.id}: {e}")
                attempt = WebhookAttempt(
                    attempt_number=attempt_num,
                    timestamp=datetime.utcnow(),
                    status=WebhookStatus.FAILED,
                    error_message=str(e),
                    response_time_ms=None
                )
                message.add_attempt(attempt)
        
        # All retries failed, move to dead letter queue
        final_error = f"Failed after {self.max_retries} attempts"
        self.dlq.add_message(message, final_error)
        return False
    
    async def _send_webhook(self, message: WebhookMessage) -> tuple[bool, Optional[str], Optional[int]]:
        """Send single webhook attempt"""
        try:
            timeout = aiohttp.ClientTimeout(total=10)  # 10 second timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    message.webhook_url,
                    json=message.payload,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if 200 <= response.status < 300:
                        return True, None, response.status
                    else:
                        return False, f"HTTP {response.status}", response.status
                        
        except asyncio.TimeoutError:
            return False, "Timeout after 10 seconds", None
        except Exception as e:
            return False, str(e), None

class WebhookEndpointValidator:
    """Validates webhook endpoints and performs health checks"""
    
    @staticmethod
    async def validate_endpoint(webhook_url: str) -> tuple[bool, str]:
        """Validate webhook endpoint is reachable and responds correctly"""
        try:
            # Test with a simple ping payload
            test_payload = {
                "test": True,
                "timestamp": datetime.utcnow().isoformat(),
                "message": "Health check from GPT-Bridge"
            }
            
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    webhook_url,
                    json=test_payload,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if 200 <= response.status < 300:
                        return True, f"Endpoint healthy (HTTP {response.status})"
                    else:
                        return False, f"Endpoint returned HTTP {response.status}"
                        
        except asyncio.TimeoutError:
            return False, "Endpoint timeout after 5 seconds"
        except Exception as e:
            return False, f"Endpoint validation failed: {str(e)}"
    
    @staticmethod
    async def health_check(webhook_url: str) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        is_healthy, message = await WebhookEndpointValidator.validate_endpoint(webhook_url)
        
        return {
            "endpoint": webhook_url,
            "healthy": is_healthy,
            "message": message,
            "checked_at": datetime.utcnow().isoformat()
        }

class WebhookMetrics:
    """Tracks webhook performance metrics"""
    
    def __init__(self):
        self.metrics = {
            "total_sent": 0,
            "successful": 0,
            "failed": 0,
            "dead_letter": 0,
            "avg_response_time_ms": 0,
            "last_success": None,
            "last_failure": None
        }
        self.response_times = []
    
    def record_attempt(self, success: bool, response_time_ms: Optional[int] = None):
        """Record webhook attempt metrics"""
        self.metrics["total_sent"] += 1
        
        if success:
            self.metrics["successful"] += 1
            self.metrics["last_success"] = datetime.utcnow().isoformat()
        else:
            self.metrics["failed"] += 1
            self.metrics["last_failure"] = datetime.utcnow().isoformat()
        
        if response_time_ms:
            self.response_times.append(response_time_ms)
            # Keep only last 1000 response times for rolling average
            if len(self.response_times) > 1000:
                self.response_times = self.response_times[-1000:]
            
            self.metrics["avg_response_time_ms"] = sum(self.response_times) / len(self.response_times)
    
    def record_dead_letter(self):
        """Record message moved to dead letter queue"""
        self.metrics["dead_letter"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        success_rate = 0
        if self.metrics["total_sent"] > 0:
            success_rate = self.metrics["successful"] / self.metrics["total_sent"]
        
        return {
            **self.metrics,
            "success_rate": success_rate,
            "failure_rate": 1 - success_rate
        }

class ProductionWebhookClient:
    """Production-ready webhook client with all reliability features"""
    
    def __init__(self, webhook_url: str, max_retries: int = 3):
        self.webhook_url = webhook_url
        self.retry_manager = WebhookRetryManager(max_retries=max_retries)
        self.validator = WebhookEndpointValidator()
        self.metrics = WebhookMetrics()
        self._endpoint_validated = False
    
    async def validate_endpoint(self) -> bool:
        """Validate webhook endpoint on startup"""
        is_healthy, message = await self.validator.validate_endpoint(self.webhook_url)
        self._endpoint_validated = is_healthy
        
        if is_healthy:
            logger.info(f"Webhook endpoint validated: {message}")
        else:
            logger.error(f"Webhook endpoint validation failed: {message}")
        
        return is_healthy
    
    async def send_webhook(self, payload: Dict[str, Any]) -> bool:
        """Send webhook with full production reliability"""
        if not self._endpoint_validated:
            logger.warning("Webhook endpoint not validated, attempting validation...")
            if not await self.validate_endpoint():
                logger.error("Cannot send webhook: endpoint validation failed")
                return False
        
        # Create webhook message
        message_id = hashlib.md5(f"{payload}{time.time()}".encode()).hexdigest()[:16]
        message = WebhookMessage(
            id=message_id,
            payload=payload,
            webhook_url=self.webhook_url,
            created_at=datetime.utcnow(),
            attempts=[],
            max_retries=self.retry_manager.max_retries
        )
        
        # Send with retry logic
        success = await self.retry_manager.send_with_retry(message)
        
        # Record metrics
        if success:
            self.metrics.record_attempt(True, message.attempts[-1].response_time_ms)
        else:
            self.metrics.record_attempt(False)
            self.metrics.record_dead_letter()
        
        return success
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        return {
            "endpoint_validated": self._endpoint_validated,
            "metrics": self.metrics.get_metrics(),
            "dead_letter_count": len(self.retry_manager.dlq.get_messages())
        }
