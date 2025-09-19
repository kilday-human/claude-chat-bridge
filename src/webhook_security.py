#!/usr/bin/env python3
"""
Webhook Security and Configuration System
Handles signature verification, rate limiting, and configuration validation
"""

import hmac
import hashlib
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from collections import defaultdict, deque
import logging
import os
from enum import Enum

logger = logging.getLogger(__name__)

class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded"""
    pass

class InvalidSignature(Exception):
    """Raised when webhook signature is invalid"""
    pass

class ConfigurationError(Exception):
    """Raised when configuration is invalid"""
    pass

@dataclass
class RateLimitConfig:
    max_requests: int
    window_seconds: int
    burst_allowance: int = 0

@dataclass
class SecurityConfig:
    webhook_secret: Optional[str] = None
    signature_header: str = "X-Webhook-Signature"
    signature_algorithm: str = "sha256"
    rate_limits: Dict[str, RateLimitConfig] = None
    allowed_ips: Optional[List[str]] = None
    require_https: bool = True

class WebhookSignatureVerifier:
    """Handles webhook signature verification"""
    
    def __init__(self, secret: str, algorithm: str = "sha256"):
        self.secret = secret.encode() if isinstance(secret, str) else secret
        self.algorithm = algorithm
    
    def generate_signature(self, payload: str) -> str:
        """Generate signature for payload"""
        if self.algorithm == "sha256":
            signature = hmac.new(
                self.secret,
                payload.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            return f"sha256={signature}"
        elif self.algorithm == "sha1":
            signature = hmac.new(
                self.secret,
                payload.encode('utf-8'),
                hashlib.sha1
            ).hexdigest()
            return f"sha1={signature}"
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def verify_signature(self, payload: str, signature: str) -> bool:
        """Verify webhook signature"""
        if not signature:
            return False
        
        try:
            # Extract algorithm and signature from header
            if '=' in signature:
                algorithm, provided_signature = signature.split('=', 1)
            else:
                algorithm = self.algorithm
                provided_signature = signature
            
            # Generate expected signature
            expected_signature = self.generate_signature(payload)
            expected_algorithm, expected_sig = expected_signature.split('=', 1)
            
            # Compare signatures using constant-time comparison
            return hmac.compare_digest(provided_signature, expected_sig)
            
        except Exception as e:
            logger.error(f"Signature verification error: {e}")
            return False

class WebhookRateLimiter:
    """Implements rate limiting for webhook calls"""
    
    def __init__(self):
        self.rate_limits: Dict[str, RateLimitConfig] = {}
        self.request_counts: Dict[str, deque] = defaultdict(lambda: deque())
        self.burst_counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
    
    def add_rate_limit(self, key: str, config: RateLimitConfig):
        """Add rate limit configuration"""
        self.rate_limits[key] = config
    
    def is_allowed(self, key: str, identifier: str = "default") -> bool:
        """Check if request is allowed under rate limits"""
        if key not in self.rate_limits:
            return True  # No rate limit configured
        
        config = self.rate_limits[key]
        now = time.time()
        window_start = now - config.window_seconds
        
        with self._lock:
            # Clean old requests
            request_key = f"{key}:{identifier}"
            requests = self.request_counts[request_key]
            
            # Remove requests outside the window
            while requests and requests[0] < window_start:
                requests.popleft()
            
            # Check if we're within limits
            if len(requests) >= config.max_requests:
                # Check burst allowance
                if config.burst_allowance > 0:
                    burst_key = f"{key}:{identifier}:burst"
                    if self.burst_counts[burst_key] < config.burst_allowance:
                        self.burst_counts[burst_key] += 1
                        return True
                
                return False
            
            # Add current request
            requests.append(now)
            return True
    
    def get_remaining_requests(self, key: str, identifier: str = "default") -> int:
        """Get remaining requests in current window"""
        if key not in self.rate_limits:
            return float('inf')
        
        config = self.rate_limits[key]
        now = time.time()
        window_start = now - config.window_seconds
        
        with self._lock:
            request_key = f"{key}:{identifier}"
            requests = self.request_counts[request_key]
            
            # Clean old requests
            while requests and requests[0] < window_start:
                requests.popleft()
            
            remaining = config.max_requests - len(requests)
            return max(0, remaining)
    
    def reset_rate_limit(self, key: str, identifier: str = "default"):
        """Reset rate limit for specific key and identifier"""
        with self._lock:
            request_key = f"{key}:{identifier}"
            burst_key = f"{key}:{identifier}:burst"
            
            if request_key in self.request_counts:
                del self.request_counts[request_key]
            if burst_key in self.burst_counts:
                del self.burst_counts[burst_key]

class WebhookConfigurationValidator:
    """Validates webhook configuration"""
    
    @staticmethod
    def validate_environment() -> Dict[str, Any]:
        """Validate environment configuration"""
        errors = []
        warnings = []
        
        # Check required environment variables
        required_vars = ["ZAPIER_WEBHOOK_URL"]
        for var in required_vars:
            if not os.getenv(var):
                errors.append(f"Missing required environment variable: {var}")
        
        # Check webhook URL format
        webhook_url = os.getenv("ZAPIER_WEBHOOK_URL")
        if webhook_url:
            if not webhook_url.startswith(("http://", "https://")):
                errors.append("ZAPIER_WEBHOOK_URL must start with http:// or https://")
            elif webhook_url.startswith("http://") and os.getenv("REQUIRE_HTTPS", "true").lower() == "true":
                warnings.append("Using HTTP instead of HTTPS for webhook URL")
        
        # Check boolean configurations
        enable_webhook = os.getenv("ENABLE_WEBHOOK", "false").lower()
        if enable_webhook not in ["true", "false"]:
            errors.append("ENABLE_WEBHOOK must be 'true' or 'false'")
        
        # Check numeric configurations
        try:
            max_retries = int(os.getenv("WEBHOOK_MAX_RETRIES", "3"))
            if max_retries < 0 or max_retries > 10:
                errors.append("WEBHOOK_MAX_RETRIES must be between 0 and 10")
        except ValueError:
            errors.append("WEBHOOK_MAX_RETRIES must be a valid integer")
        
        try:
            timeout_seconds = int(os.getenv("WEBHOOK_TIMEOUT_SECONDS", "10"))
            if timeout_seconds < 1 or timeout_seconds > 60:
                errors.append("WEBHOOK_TIMEOUT_SECONDS must be between 1 and 60")
        except ValueError:
            errors.append("WEBHOOK_TIMEOUT_SECONDS must be a valid integer")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    @staticmethod
    def validate_security_config(config: SecurityConfig) -> Dict[str, Any]:
        """Validate security configuration"""
        errors = []
        warnings = []
        
        # Check webhook secret
        if not config.webhook_secret:
            warnings.append("No webhook secret configured - signature verification disabled")
        elif len(config.webhook_secret) < 16:
            errors.append("Webhook secret must be at least 16 characters long")
        
        # Check signature algorithm
        if config.signature_algorithm not in ["sha1", "sha256"]:
            errors.append("Signature algorithm must be 'sha1' or 'sha256'")
        
        # Check rate limits
        if config.rate_limits:
            for key, rate_limit in config.rate_limits.items():
                if rate_limit.max_requests <= 0:
                    errors.append(f"Rate limit '{key}' max_requests must be positive")
                if rate_limit.window_seconds <= 0:
                    errors.append(f"Rate limit '{key}' window_seconds must be positive")
                if rate_limit.burst_allowance < 0:
                    errors.append(f"Rate limit '{key}' burst_allowance must be non-negative")
        
        # Check allowed IPs
        if config.allowed_ips:
            for ip in config.allowed_ips:
                if not WebhookConfigurationValidator._is_valid_ip(ip):
                    errors.append(f"Invalid IP address: {ip}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    @staticmethod
    def _is_valid_ip(ip: str) -> bool:
        """Basic IP address validation"""
        import socket
        try:
            socket.inet_aton(ip)
            return True
        except socket.error:
            return False

class WebhookSecurityManager:
    """Main security manager for webhook operations"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.signature_verifier = None
        self.rate_limiter = WebhookRateLimiter()
        
        # Initialize signature verifier if secret is provided
        if config.webhook_secret:
            self.signature_verifier = WebhookSignatureVerifier(
                config.webhook_secret,
                config.signature_algorithm
            )
        
        # Set up rate limits
        if config.rate_limits:
            for key, rate_limit in config.rate_limits.items():
                self.rate_limiter.add_rate_limit(key, rate_limit)
    
    def verify_request(self, payload: str, headers: Dict[str, str], 
                      client_ip: str = None, identifier: str = "default") -> bool:
        """Verify webhook request security"""
        
        # Check IP allowlist
        if self.config.allowed_ips and client_ip:
            if client_ip not in self.config.allowed_ips:
                logger.warning(f"Request from disallowed IP: {client_ip}")
                return False
        
        # Check rate limits
        for rate_limit_key in self.rate_limiter.rate_limits:
            if not self.rate_limiter.is_allowed(rate_limit_key, identifier):
                logger.warning(f"Rate limit exceeded for {rate_limit_key}:{identifier}")
                raise RateLimitExceeded(f"Rate limit exceeded for {rate_limit_key}")
        
        # Verify signature if verifier is configured
        if self.signature_verifier:
            signature = headers.get(self.config.signature_header)
            if not signature:
                logger.warning("Missing webhook signature")
                return False
            
            if not self.signature_verifier.verify_signature(payload, signature):
                logger.warning("Invalid webhook signature")
                return False
        
        return True
    
    def generate_signature(self, payload: str) -> str:
        """Generate signature for outgoing webhook"""
        if not self.signature_verifier:
            return ""
        return self.signature_verifier.generate_signature(payload)
    
    def get_rate_limit_status(self, identifier: str = "default") -> Dict[str, Any]:
        """Get rate limit status for identifier"""
        status = {}
        for key in self.rate_limiter.rate_limits:
            status[key] = {
                "remaining": self.rate_limiter.get_remaining_requests(key, identifier),
                "limit": self.rate_limiter.rate_limits[key].max_requests
            }
        return status

# Default security configurations
def get_default_security_config() -> SecurityConfig:
    """Get default security configuration from environment"""
    return SecurityConfig(
        webhook_secret=os.getenv("WEBHOOK_SECRET"),
        signature_header=os.getenv("WEBHOOK_SIGNATURE_HEADER", "X-Webhook-Signature"),
        signature_algorithm=os.getenv("WEBHOOK_SIGNATURE_ALGORITHM", "sha256"),
        rate_limits={
            "webhook_calls": RateLimitConfig(
                max_requests=int(os.getenv("WEBHOOK_RATE_LIMIT", "100")),
                window_seconds=int(os.getenv("WEBHOOK_RATE_WINDOW", "3600")),  # 1 hour
                burst_allowance=int(os.getenv("WEBHOOK_BURST_ALLOWANCE", "10"))
            )
        },
        allowed_ips=os.getenv("WEBHOOK_ALLOWED_IPS", "").split(",") if os.getenv("WEBHOOK_ALLOWED_IPS") else None,
        require_https=os.getenv("REQUIRE_HTTPS", "true").lower() == "true"
    )

# Import threading for rate limiter
import threading
