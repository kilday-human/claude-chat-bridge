#!/usr/bin/env python3
"""
Zapier Integration for GPT-Bridge - Level 1 Foundation
Webhook endpoint and usage tracking for Zapier integration
"""

import os
import json
import time
import asyncio
import aiohttp
from typing import Dict, Any, Optional, Literal
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# Sentry integration for error tracking
try:
    import sentry_sdk
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type definitions matching the TypeScript interfaces
@dataclass
class QueryData:
    summary: str  # First 50 chars
    category: Literal['code_review', 'debugging', 'architecture', 'general']
    estimated_complexity: Literal[1, 2, 3]  # simple categorization

@dataclass
class ResponseData:
    model_used: Literal['gpt-4', 'claude-3']
    response_time_ms: int
    estimated_tokens: int
    success: bool

@dataclass
class UsagePayload:
    timestamp: str
    session_id: str
    query: QueryData
    response: ResponseData
    user_agent: Optional[str] = None

class ZapierWebhookClient:
    """Handles webhook calls to Zapier with proper error handling and timeouts"""
    
    def __init__(self):
        self.webhook_url = os.getenv('ZAPIER_WEBHOOK_URL')
        self.enabled = os.getenv('ENABLE_WEBHOOK', 'false').lower() == 'true'
        self.timeout = 5.0  # 5-second timeout as specified
        
    async def send_usage_data(self, payload: UsagePayload) -> bool:
        """
        Send usage data to Zapier webhook with proper error handling
        Returns True if successful, False otherwise
        """
        if not self.enabled or not self.webhook_url:
            logger.info("Webhook disabled or URL not configured")
            return False
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=asdict(payload),
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status == 200:
                        logger.info(f"Successfully sent usage data to Zapier: {payload.session_id}")
                        return True
                    else:
                        logger.warning(f"Zapier webhook returned status {response.status}")
                        return False
                        
        except asyncio.TimeoutError:
            logger.warning("Zapier webhook timed out after 5 seconds")
            if SENTRY_AVAILABLE:
                sentry_sdk.capture_exception(Exception("Zapier webhook timeout"))
            return False
        except Exception as e:
            logger.error(f"Failed to send data to Zapier: {str(e)}")
            if SENTRY_AVAILABLE:
                sentry_sdk.capture_exception(e)
            return False

class UsageTracker:
    """Tracks and categorizes usage data for Zapier integration"""
    
    def __init__(self):
        self.webhook_client = ZapierWebhookClient()
        
    def categorize_query(self, query: str) -> str:
        """Simple query categorization based on content analysis"""
        query_lower = query.lower()
        
        # Code review patterns
        if any(word in query_lower for word in ['review', 'code review', 'pull request', 'pr', 'refactor']):
            return 'code_review'
        
        # Debugging patterns  
        if any(word in query_lower for word in ['debug', 'error', 'bug', 'fix', 'issue', 'problem']):
            return 'debugging'
            
        # Architecture patterns
        if any(word in query_lower for word in ['architecture', 'design', 'structure', 'system', 'pattern']):
            return 'architecture'
            
        # Default to general
        return 'general'
    
    def estimate_complexity(self, query: str, response_time_ms: int) -> int:
        """Simple complexity estimation based on query length and response time"""
        query_length = len(query)
        
        # Simple heuristic: longer queries and slower responses = higher complexity
        if query_length > 200 or response_time_ms > 3000:
            return 3  # High complexity
        elif query_length > 100 or response_time_ms > 1000:
            return 2  # Medium complexity
        else:
            return 1  # Low complexity
    
    def determine_model_type(self, model_name: str) -> str:
        """Map model names to standardized types"""
        if 'gpt' in model_name.lower():
            return 'gpt-4'
        elif 'claude' in model_name.lower():
            return 'claude-3'
        else:
            return 'gpt-4'  # Default fallback
    
    async def track_usage(self, 
                         session_id: str,
                         query: str, 
                         model_name: str,
                         response_time_ms: int,
                         estimated_tokens: int,
                         success: bool,
                         user_agent: Optional[str] = None) -> bool:
        """
        Track usage and send to Zapier webhook
        Returns True if webhook was sent successfully
        """
        
        # Create usage payload
        payload = UsagePayload(
            timestamp=datetime.utcnow().isoformat() + 'Z',
            session_id=session_id,
            query=QueryData(
                summary=query[:50] + ('...' if len(query) > 50 else ''),
                category=self.categorize_query(query),
                estimated_complexity=self.estimate_complexity(query, response_time_ms)
            ),
            response=ResponseData(
                model_used=self.determine_model_type(model_name),
                response_time_ms=response_time_ms,
                estimated_tokens=estimated_tokens,
                success=success
            ),
            user_agent=user_agent
        )
        
        # Send to Zapier webhook
        return await self.webhook_client.send_usage_data(payload)

# Global usage tracker instance
usage_tracker = UsageTracker()

# Convenience function for easy integration
async def track_gpt_bridge_usage(session_id: str,
                                query: str,
                                model_name: str, 
                                response_time_ms: int,
                                estimated_tokens: int,
                                success: bool,
                                user_agent: Optional[str] = None) -> bool:
    """
    Convenience function to track GPT-Bridge usage
    Call this after successful responses in the main bridge logic
    """
    return await usage_tracker.track_usage(
        session_id=session_id,
        query=query,
        model_name=model_name,
        response_time_ms=response_time_ms,
        estimated_tokens=estimated_tokens,
        success=success,
        user_agent=user_agent
    )
