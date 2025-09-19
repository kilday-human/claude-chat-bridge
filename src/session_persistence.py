#!/usr/bin/env python3
"""
Session Persistence System
Handles session data storage, duplicate detection, and data retention
"""

import json
import sqlite3
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class SessionData:
    session_id: str
    created_at: datetime
    last_activity: datetime
    query_count: int
    total_tokens: int
    total_cost: float
    models_used: Set[str]
    categories: Set[str]
    metadata: Dict[str, Any]

@dataclass
class WebhookDelivery:
    delivery_id: str
    session_id: str
    payload_hash: str
    delivered_at: datetime
    success: bool
    retry_count: int

class SessionPersistenceManager:
    """Manages session data persistence and duplicate detection"""
    
    def __init__(self, db_path: str = "sessions.db"):
        self.db_path = db_path
        self._init_db()
        self._active_sessions: Dict[str, SessionData] = {}
        self._delivery_cache: Set[str] = set()  # In-memory cache for fast duplicate detection
    
    def _init_db(self):
        """Initialize SQLite database for session persistence"""
        with sqlite3.connect(self.db_path) as conn:
            # Sessions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    last_activity TEXT NOT NULL,
                    query_count INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    total_cost REAL DEFAULT 0.0,
                    models_used TEXT NOT NULL,
                    categories TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
            """)
            
            # Webhook deliveries table for duplicate detection
            conn.execute("""
                CREATE TABLE IF NOT EXISTS webhook_deliveries (
                    delivery_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    delivered_at TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    retry_count INTEGER DEFAULT 0,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_last_activity ON sessions (last_activity)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_deliveries_session_id ON webhook_deliveries (session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_deliveries_payload_hash ON webhook_deliveries (payload_hash)")
    
    def _load_active_sessions(self):
        """Load active sessions from database into memory"""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)  # Load sessions active in last 24 hours
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT session_id, created_at, last_activity, query_count, total_tokens, 
                       total_cost, models_used, categories, metadata
                FROM sessions
                WHERE last_activity > ?
            """, (cutoff_time.isoformat(),))
            
            for row in cursor.fetchall():
                session = SessionData(
                    session_id=row[0],
                    created_at=datetime.fromisoformat(row[1]),
                    last_activity=datetime.fromisoformat(row[2]),
                    query_count=row[3],
                    total_tokens=row[4],
                    total_cost=row[5],
                    models_used=set(json.loads(row[6])),
                    categories=set(json.loads(row[7])),
                    metadata=json.loads(row[8])
                )
                self._active_sessions[session.session_id] = session
    
    def _load_delivery_cache(self):
        """Load recent delivery hashes into memory for fast duplicate detection"""
        cutoff_time = datetime.utcnow() - timedelta(hours=1)  # Cache last hour of deliveries
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT payload_hash
                FROM webhook_deliveries
                WHERE delivered_at > ?
            """, (cutoff_time.isoformat(),))
            
            for row in cursor.fetchall():
                self._delivery_cache.add(row[0])
    
    def get_or_create_session(self, session_id: str) -> SessionData:
        """Get existing session or create new one"""
        if not self._active_sessions:
            self._load_active_sessions()
        
        if session_id not in self._active_sessions:
            # Create new session
            session = SessionData(
                session_id=session_id,
                created_at=datetime.utcnow(),
                last_activity=datetime.utcnow(),
                query_count=0,
                total_tokens=0,
                total_cost=0.0,
                models_used=set(),
                categories=set(),
                metadata={}
            )
            self._active_sessions[session_id] = session
            self._save_session(session)
            logger.info(f"Created new session: {session_id}")
        
        return self._active_sessions[session_id]
    
    def update_session(self, session_id: str, **updates):
        """Update session data"""
        if session_id not in self._active_sessions:
            self.get_or_create_session(session_id)
        
        session = self._active_sessions[session_id]
        session.last_activity = datetime.utcnow()
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(session, key):
                if key in ['models_used', 'categories'] and isinstance(value, (list, set)):
                    getattr(session, key).update(value)
                else:
                    setattr(session, key, value)
        
        self._save_session(session)
    
    def _save_session(self, session: SessionData):
        """Save session to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO sessions 
                (session_id, created_at, last_activity, query_count, total_tokens, 
                 total_cost, models_used, categories, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.session_id,
                session.created_at.isoformat(),
                session.last_activity.isoformat(),
                session.query_count,
                session.total_tokens,
                session.total_cost,
                json.dumps(list(session.models_used)),
                json.dumps(list(session.categories)),
                json.dumps(session.metadata)
            ))
    
    def is_duplicate_webhook(self, payload: Dict[str, Any]) -> bool:
        """Check if webhook payload is a duplicate"""
        if not self._delivery_cache:
            self._load_delivery_cache()
        
        # Create hash of payload for duplicate detection
        payload_str = json.dumps(payload, sort_keys=True)
        payload_hash = hashlib.sha256(payload_str.encode()).hexdigest()
        
        if payload_hash in self._delivery_cache:
            logger.info(f"Duplicate webhook detected: {payload_hash[:16]}...")
            return True
        
        return False
    
    def record_webhook_delivery(self, session_id: str, payload: Dict[str, Any], success: bool, retry_count: int = 0):
        """Record webhook delivery for duplicate detection"""
        payload_str = json.dumps(payload, sort_keys=True)
        payload_hash = hashlib.sha256(payload_str.encode()).hexdigest()
        delivery_id = f"{session_id}_{payload_hash[:16]}"
        
        delivery = WebhookDelivery(
            delivery_id=delivery_id,
            session_id=session_id,
            payload_hash=payload_hash,
            delivered_at=datetime.utcnow(),
            success=success,
            retry_count=retry_count
        )
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO webhook_deliveries
                (delivery_id, session_id, payload_hash, delivered_at, success, retry_count)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                delivery.delivery_id,
                delivery.session_id,
                delivery.payload_hash,
                delivery.delivered_at.isoformat(),
                delivery.success,
                delivery.retry_count
            ))
        
        # Add to cache
        self._delivery_cache.add(payload_hash)
        
        # Clean old cache entries (keep last 1000)
        if len(self._delivery_cache) > 1000:
            self._delivery_cache = set(list(self._delivery_cache)[-1000:])
    
    def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive session statistics"""
        if session_id not in self._active_sessions:
            return None
        
        session = self._active_sessions[session_id]
        
        # Get webhook delivery stats for this session
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) as total_deliveries,
                       SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_deliveries,
                       SUM(retry_count) as total_retries
                FROM webhook_deliveries
                WHERE session_id = ?
            """, (session_id,))
            
            row = cursor.fetchone()
            webhook_stats = {
                "total_deliveries": row[0] or 0,
                "successful_deliveries": row[1] or 0,
                "total_retries": row[2] or 0
            }
        
        return {
            "session_id": session.session_id,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "query_count": session.query_count,
            "total_tokens": session.total_tokens,
            "total_cost": session.total_cost,
            "models_used": list(session.models_used),
            "categories": list(session.categories),
            "webhook_stats": webhook_stats,
            "metadata": session.metadata
        }
    
    def cleanup_old_data(self, retention_days: int = 30):
        """Clean up old session and delivery data"""
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        with sqlite3.connect(self.db_path) as conn:
            # Clean old sessions
            cursor = conn.execute("""
                DELETE FROM sessions 
                WHERE last_activity < ?
            """, (cutoff_date.isoformat(),))
            sessions_deleted = cursor.rowcount
            
            # Clean old webhook deliveries
            cursor = conn.execute("""
                DELETE FROM webhook_deliveries 
                WHERE delivered_at < ?
            """, (cutoff_date.isoformat(),))
            deliveries_deleted = cursor.rowcount
            
            logger.info(f"Cleaned up {sessions_deleted} old sessions and {deliveries_deleted} old deliveries")
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get analytics summary across all sessions"""
        with sqlite3.connect(self.db_path) as conn:
            # Session analytics
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_sessions,
                    SUM(query_count) as total_queries,
                    SUM(total_tokens) as total_tokens,
                    SUM(total_cost) as total_cost,
                    AVG(query_count) as avg_queries_per_session
                FROM sessions
                WHERE last_activity > datetime('now', '-7 days')
            """)
            session_stats = cursor.fetchone()
            
            # Webhook analytics
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_deliveries,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_deliveries,
                    AVG(CASE WHEN success = 1 THEN retry_count ELSE NULL END) as avg_retries_success,
                    AVG(CASE WHEN success = 0 THEN retry_count ELSE NULL END) as avg_retries_failure
                FROM webhook_deliveries
                WHERE delivered_at > datetime('now', '-7 days')
            """)
            webhook_stats = cursor.fetchone()
        
        return {
            "period": "last_7_days",
            "sessions": {
                "total": session_stats[0] or 0,
                "total_queries": session_stats[1] or 0,
                "total_tokens": session_stats[2] or 0,
                "total_cost": session_stats[3] or 0.0,
                "avg_queries_per_session": session_stats[4] or 0.0
            },
            "webhooks": {
                "total_deliveries": webhook_stats[0] or 0,
                "successful_deliveries": webhook_stats[1] or 0,
                "success_rate": (webhook_stats[1] or 0) / max(webhook_stats[0] or 1, 1),
                "avg_retries_success": webhook_stats[2] or 0.0,
                "avg_retries_failure": webhook_stats[3] or 0.0
            }
        }
