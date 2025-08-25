"""
Cache Manager for Claude-GPT Bridge
Provides response caching with TTL, LRU eviction, and cost optimization
"""

import hashlib
import json
import time
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
import threading
from collections import OrderedDict


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    response: str
    model: str
    timestamp: float
    cost: float
    tokens_used: int
    hit_count: int = 0
    last_accessed: float = None
    
    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.timestamp


class CacheManager:
    """
    Multi-level cache with:
    - In-memory LRU cache for hot responses
    - Disk persistence for session continuity
    - TTL-based expiration
    - Cost and performance analytics
    """
    
    def __init__(
        self,
        max_memory_size: int = 100,
        max_disk_size: int = 1000,
        default_ttl: int = 3600,  # 1 hour
        cache_dir: str = "data/cache"
    ):
        self.max_memory_size = max_memory_size
        self.max_disk_size = max_disk_size
        self.default_ttl = default_ttl
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # In-memory LRU cache
        self.memory_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Disk cache index
        self.disk_index: Dict[str, Dict[str, Any]] = {}
        
        # Performance metrics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'memory_hits': 0,
            'disk_hits': 0,
            'evictions': 0,
            'cost_saved': 0.0,
            'tokens_saved': 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Load existing disk cache
        self._load_disk_index()
    
    def _generate_key(self, prompt: str, model: str, context: Dict[str, Any] = None) -> str:
        """Generate cache key from prompt, model, and context"""
        cache_input = {
            'prompt': prompt.strip(),
            'model': model,
            'context': context or {}
        }
        
        # Create deterministic hash
        cache_str = json.dumps(cache_input, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()[:16]
    
    def get(self, prompt: str, model: str, context: Dict[str, Any] = None) -> Optional[CacheEntry]:
        """Retrieve cached response if available and valid"""
        with self._lock:
            key = self._generate_key(prompt, model, context)
            
            # Check memory cache first (hot path)
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if self._is_valid(entry):
                    # Move to end (LRU)
                    self.memory_cache.move_to_end(key)
                    entry.hit_count += 1
                    entry.last_accessed = time.time()
                    
                    self.stats['hits'] += 1
                    self.stats['memory_hits'] += 1
                    self.stats['cost_saved'] += entry.cost
                    self.stats['tokens_saved'] += entry.tokens_used
                    
                    return entry
                else:
                    # Expired - remove
                    del self.memory_cache[key]
            
            # Check disk cache
            if key in self.disk_index:
                entry = self._load_from_disk(key)
                if entry and self._is_valid(entry):
                    # Promote to memory cache
                    self._add_to_memory(key, entry)
                    entry.hit_count += 1
                    entry.last_accessed = time.time()
                    
                    self.stats['hits'] += 1
                    self.stats['disk_hits'] += 1
                    self.stats['cost_saved'] += entry.cost
                    self.stats['tokens_saved'] += entry.tokens_used
                    
                    return entry
                else:
                    # Expired or corrupted - remove
                    self._remove_from_disk(key)
            
            self.stats['misses'] += 1
            return None
    
    def put(
        self,
        prompt: str,
        model: str,
        response: str,
        cost: float,
        tokens_used: int,
        context: Dict[str, Any] = None,
        ttl: Optional[int] = None
    ):
        """Cache response with metadata"""
        with self._lock:
            key = self._generate_key(prompt, model, context)
            
            entry = CacheEntry(
                response=response,
                model=model,
                timestamp=time.time(),
                cost=cost,
                tokens_used=tokens_used
            )
            
            # Add to memory cache
            self._add_to_memory(key, entry)
            
            # Persist to disk
            self._save_to_disk(key, entry)
    
    def _add_to_memory(self, key: str, entry: CacheEntry):
        """Add entry to memory cache with LRU eviction"""
        self.memory_cache[key] = entry
        
        # LRU eviction
        if len(self.memory_cache) > self.max_memory_size:
            oldest_key, oldest_entry = self.memory_cache.popitem(last=False)
            self.stats['evictions'] += 1
    
    def _is_valid(self, entry: CacheEntry) -> bool:
        """Check if cache entry is still valid"""
        return (time.time() - entry.timestamp) < self.default_ttl
    
    def _load_disk_index(self):
        """Load disk cache index on startup"""
        index_path = self.cache_dir / "cache_index.json"
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    self.disk_index = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.disk_index = {}
    
    def _save_disk_index(self):
        """Save disk cache index"""
        index_path = self.cache_dir / "cache_index.json"
        try:
            with open(index_path, 'w') as f:
                json.dump(self.disk_index, f)
        except IOError:
            pass  # Graceful degradation
    
    def _load_from_disk(self, key: str) -> Optional[CacheEntry]:
        """Load cache entry from disk"""
        cache_path = self.cache_dir / f"{key}.pkl"
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except (pickle.PickleError, IOError):
            return None
    
    def _save_to_disk(self, key: str, entry: CacheEntry):
        """Save cache entry to disk"""
        cache_path = self.cache_dir / f"{key}.pkl"
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(entry, f)
            
            # Update index
            self.disk_index[key] = {
                'timestamp': entry.timestamp,
                'model': entry.model,
                'cost': entry.cost,
                'tokens': entry.tokens_used
            }
            
            # Disk size management
            if len(self.disk_index) > self.max_disk_size:
                self._evict_oldest_disk_entries()
            
            self._save_disk_index()
            
        except (pickle.PickleError, IOError):
            pass  # Graceful degradation
    
    def _remove_from_disk(self, key: str):
        """Remove entry from disk cache"""
        cache_path = self.cache_dir / f"{key}.pkl"
        
        try:
            if cache_path.exists():
                cache_path.unlink()
            if key in self.disk_index:
                del self.disk_index[key]
            self._save_disk_index()
        except (OSError, IOError):
            pass  # Graceful degradation
    
    def _evict_oldest_disk_entries(self):
        """Evict oldest disk entries when over limit"""
        # Sort by timestamp and remove oldest 10%
        sorted_entries = sorted(
            self.disk_index.items(),
            key=lambda x: x[1]['timestamp']
        )
        
        evict_count = max(1, len(sorted_entries) // 10)
        for key, _ in sorted_entries[:evict_count]:
            self._remove_from_disk(key)
    
    def clear(self, memory_only: bool = False):
        """Clear cache"""
        with self._lock:
            self.memory_cache.clear()
            
            if not memory_only:
                # Clear disk cache
                for key in list(self.disk_index.keys()):
                    self._remove_from_disk(key)
            
            # Reset stats
            self.stats = {
                'hits': 0,
                'misses': 0,
                'memory_hits': 0,
                'disk_hits': 0,
                'evictions': 0,
                'cost_saved': 0.0,
                'tokens_saved': 0
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        with self._lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                **self.stats,
                'hit_rate': hit_rate,
                'memory_size': len(self.memory_cache),
                'disk_size': len(self.disk_index),
                'total_requests': total_requests
            }
    
    def cleanup_expired(self):
        """Remove expired entries"""
        with self._lock:
            # Clean memory cache
            expired_keys = [
                key for key, entry in self.memory_cache.items()
                if not self._is_valid(entry)
            ]
            for key in expired_keys:
                del self.memory_cache[key]
            
            # Clean disk cache
            expired_disk_keys = [
                key for key, meta in self.disk_index.items()
                if (time.time() - meta['timestamp']) > self.default_ttl
            ]
            for key in expired_disk_keys:
                self._remove_from_disk(key)
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information"""
        with self._lock:
            return {
                'config': {
                    'max_memory_size': self.max_memory_size,
                    'max_disk_size': self.max_disk_size,
                    'default_ttl': self.default_ttl,
                    'cache_dir': str(self.cache_dir)
                },
                'current_state': {
                    'memory_entries': len(self.memory_cache),
                    'disk_entries': len(self.disk_index),
                    'memory_utilization': len(self.memory_cache) / self.max_memory_size,
                    'disk_utilization': len(self.disk_index) / self.max_disk_size
                },
                'performance': self.get_stats()
            }


# Global cache instance
_cache_manager = None


def get_cache_manager(**kwargs) -> CacheManager:
    """Get or create global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(**kwargs)
    return _cache_manager


def cache_response(func):
    """Decorator for caching API responses"""
    def wrapper(*args, **kwargs):
        cache = get_cache_manager()
        
        # Extract cache key components
        prompt = args[0] if args else kwargs.get('prompt', '')
        model = kwargs.get('model', 'unknown')
        
        # Check cache first
        cached = cache.get(prompt, model)
        if cached:
            return cached.response, {
                'cached': True,
                'cost': 0.0,  # No new cost for cached response
                'tokens_used': 0,
                'original_cost': cached.cost,
                'hit_count': cached.hit_count
            }
        
        # Call original function
        response, metadata = func(*args, **kwargs)
        
        # Cache the response
        cache.put(
            prompt=prompt,
            model=model,
            response=response,
            cost=metadata.get('cost', 0.0),
            tokens_used=metadata.get('tokens_used', 0)
        )
        
        return response, {**metadata, 'cached': False}
    
    return wrapper
