#!/usr/bin/env python3
"""
Session Manager for GPT-Bridge
Provides persistent session tracking across application restarts
"""

import os
import json
import time
import uuid
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime, timedelta

class SessionManager:
    """Manages persistent sessions for GPT-Bridge"""
    
    def __init__(self, session_file: str = "data/sessions.json"):
        self.session_file = Path(session_file)
        self.sessions: Dict[str, Dict] = {}
        self.current_session_id: Optional[str] = None
        self._load_sessions()
    
    def _load_sessions(self):
        """Load existing sessions from file"""
        if self.session_file.exists():
            try:
                with open(self.session_file, 'r') as f:
                    self.sessions = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.sessions = {}
        else:
            self.sessions = {}
    
    def _save_sessions(self):
        """Save sessions to file"""
        self.session_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.session_file, 'w') as f:
            json.dump(self.sessions, f, indent=2)
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """
        Get existing session or create new one
        Returns session ID
        """
        if session_id and session_id in self.sessions:
            # Update last activity
            self.sessions[session_id]['last_activity'] = datetime.utcnow().isoformat()
            self.sessions[session_id]['request_count'] = self.sessions[session_id].get('request_count', 0) + 1
            self.current_session_id = session_id
            self._save_sessions()
            return session_id
        
        # Create new session
        new_session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"
        self.sessions[new_session_id] = {
            'session_id': new_session_id,
            'created_at': datetime.utcnow().isoformat(),
            'last_activity': datetime.utcnow().isoformat(),
            'request_count': 1,
            'total_tokens': 0,
            'total_cost': 0.0,
            'queries': []
        }
        self.current_session_id = new_session_id
        self._save_sessions()
        return new_session_id
    
    def add_query_to_session(self, session_id: str, query: str, response_time_ms: int, tokens: int, cost: float):
        """Add query data to session"""
        if session_id in self.sessions:
            self.sessions[session_id]['queries'].append({
                'query': query[:100] + ('...' if len(query) > 100 else ''),
                'timestamp': datetime.utcnow().isoformat(),
                'response_time_ms': response_time_ms,
                'tokens': tokens,
                'cost': cost
            })
            self.sessions[session_id]['total_tokens'] += tokens
            self.sessions[session_id]['total_cost'] += cost
            self.sessions[session_id]['last_activity'] = datetime.utcnow().isoformat()
            self._save_sessions()
    
    def get_session_stats(self, session_id: str) -> Dict:
        """Get statistics for a session"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            return {
                'session_id': session_id,
                'created_at': session['created_at'],
                'last_activity': session['last_activity'],
                'request_count': session['request_count'],
                'total_tokens': session['total_tokens'],
                'total_cost': session['total_cost'],
                'avg_response_time': sum(q['response_time_ms'] for q in session['queries']) / len(session['queries']) if session['queries'] else 0
            }
        return {}
    
    def cleanup_old_sessions(self, days: int = 7):
        """Remove sessions older than specified days"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        sessions_to_remove = []
        
        for session_id, session_data in self.sessions.items():
            last_activity = datetime.fromisoformat(session_data['last_activity'])
            if last_activity < cutoff:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.sessions[session_id]
        
        if sessions_to_remove:
            self._save_sessions()
            print(f"Cleaned up {len(sessions_to_remove)} old sessions")

# Global session manager instance
session_manager = SessionManager()
