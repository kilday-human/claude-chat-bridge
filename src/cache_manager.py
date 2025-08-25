"""
Advanced Cache Manager for Claude-GPT Bridge
Multi-level caching with memory, disk persistence, and intelligent eviction
"""

import json
import time
import hashlib
import pickle
import os
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict
from collections import OrderedDict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    data: Any
    timestamp: float
    access_count: int
    ttl: Optional[float]
    size_bytes: int
    cache_key: str
    metadata: Dict[str, Any]

    def is_expired(self) -> bool:
        """Check if entry has exceeded TTL"""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl

    def touch(self):
        """Update access timestamp and count"""
        self.access_count += 1
        return self

class CacheManager:
    """Multi-level cache with LRU eviction, TTL, and disk persistence"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_memory_size = config.get('max_memory_size', 100)  # number of entries
        self.max_disk_size = config.get('max_disk_size', 1000)    # number of entries
        self.default_ttl = config.get('default_ttl', 3600)        # seconds
        self.cache_dir = Path(config.get('cache_dir', 'data/cache'))
        
        # Memory cache (LRU OrderedDict)
        self._memory_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Disk cache index (tracks what's on disk)
        self._disk_index: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self._stats = {
            'memory_hits': 0,
            'disk_hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0,
            'memory_size': 0,
            'disk_size': 0
        }
        
        # Initialize cache directory and load disk index
        self._setup_cache_directory()
        self._load_disk_index()
    
    def _setup_cache_directory(self):
        """Create cache directory structure"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / 'entries').mkdir(exist_ok=True)
        (self.cache_dir / 'metadata').mkdir(exist_ok=True)
    
    def _load_disk_index(self):
        """Load disk cache index from file"""
        index_path = self.cache_dir / 'index.json'
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    self._disk_index = json.load(f)
                    self._stats['disk_size'] = len(self._disk_index)
                logger.info(f"Loaded disk cache index: {len(self._disk_index)} entries")
            except Exception as e:
                logger.error(f"Error loading disk cache index: {e}")
                self._disk_index = {}
    
    def _save_disk_index(self):
        """Save disk cache index to file"""
        index_path = self.cache_dir / 'index.json'
        try:
            with open(index_path, 'w') as f:
                json.dump(self._disk_index, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving disk cache index: {e}")
    
    def _generate_cache_key(self, prompt: str, model: str, **kwargs) -> str:
        """Generate deterministic cache key from inputs"""
        # Create hash input from all parameters
        hash_input = {
            'prompt': prompt,
            'model': model,
            **kwargs
        }
        
        # Sort keys for consistent hashing
        hash_str = json.dumps(hash_input, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()[:32]
    
    def _get_entry_size(self, data: Any) -> int:
        """Estimate size of cache entry in bytes"""
        try:
            return len(pickle.dumps(data))
        except:
            return len(str(data))
    
    def _evict_from_memory(self):
        """Evict least recently used entry from memory cache"""
        if not self._memory_cache:
            return
        
        # Remove oldest (least recently used) entry
        cache_key, entry = self._memory_cache.popitem(last=False)
        self._stats['evictions'] += 1
        self._stats['memory_size'] -= 1
        
        # Move to disk if disk cache has space
        if len(self._disk_index) < self.max_disk_size:
            self._save_to_disk(cache_key, entry)
        
        logger.debug(f"Evicted from memory cache: {cache_key}")
    
    def _save_to_disk(self, cache_key: str, entry: CacheEntry):
        """Save entry to disk cache"""
        try:
            entry_path = self.cache_dir / 'entries' / f"{cache_key}.pkl"
            metadata_path = self.cache_dir / 'metadata' / f"{cache_key}.json"
            
            # Save data
            with open(entry_path, 'wb') as f:
                pickle.dump(entry.data, f)
            
            # Save metadata
            metadata = {
                'timestamp': entry.timestamp,
                'access_count': entry.access_count,
                'ttl': entry.ttl,
                'size_bytes': entry.size_bytes,
                'cache_key': entry.cache_key,
                'metadata': entry.metadata
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update disk index
            self._disk_index[cache_key] = metadata
            self._stats['disk_size'] += 1
            self._save_disk_index()
            
            logger.debug(f"Saved to disk cache: {cache_key}")
            
        except Exception as e:
            logger.error(f"Error saving to disk cache: {e}")
    
    def _load_from_disk(self, cache_key: str) -> Optional[CacheEntry]:
        """Load entry from disk cache"""
        try:
            entry_path = self.cache_dir / 'entries' / f"{cache_key}.pkl"
            metadata_path = self.cache_dir / 'metadata' / f"{cache_key}.json"
            
            if not entry_path.exists() or not metadata_path.exists():
                return None
            
            # Load data
            with open(entry_path, 'rb') as f:
                data = pickle.load(f)
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Create entry
            entry = CacheEntry(
                data=data,
                timestamp=metadata['timestamp'],
                access_count=metadata['access_count'],
                ttl=metadata['ttl'],
                size_bytes=metadata['size_bytes'],
                cache_key=metadata['cache_key'],
                metadata=metadata['metadata']
            )
            
            return entry
            
        except Exception as e:
            logger.error(f"Error loading from disk cache: {e}")
            return None
    
    def get(self, prompt: str, model: str, **kwargs) -> Tuple[Optional[Any], bool, str]:
        """
        Get cached response
        Returns: (data, cache_hit, cache_level)
        """
        cache_key = self._generate_cache_key(prompt, model, **kwargs)
        self._stats['total_requests'] += 1
        
        # Check memory cache first
        if cache_key in self._memory_cache:
            entry = self._memory_cache[cache_key]
            
            # Check if expired
            if entry.is_expired():
                del self._memory_cache[cache_key]
                self._stats['memory_size'] -= 1
            else:
                # Move to end (most recently used)
                self._memory_cache.move_to_end(cache_key)
                entry.touch()
                self._stats['memory_hits'] += 1
                return entry.data, True, 'memory'
        
        # Check disk cache
        if cache_key in self._disk_index:
            entry = self._load_from_disk(cache_key)
            
            if entry and not entry.is_expired():
                # Move to memory cache (promote hot data)
                if len(self._memory_cache) >= self.max_memory_size:
                    self._evict_from_memory()
                
                entry.touch()
                self._memory_cache[cache_key] = entry
                self._stats['memory_size'] += 1
                self._stats['disk_hits'] += 1
                
                return entry.data, True, 'disk'
            elif entry:
                # Remove expired entry from disk
                self._remove_from_disk(cache_key)
        
        # Cache miss
        self._stats['misses'] += 1
        return None, False, 'none'
    
    def put(self, prompt: str, model: str, data: Any, ttl: Optional[float] = None, **kwargs):
        """Store response in cache"""
        cache_key = self._generate_cache_key(prompt, model, **kwargs)
        
        if ttl is None:
            ttl = self.default_ttl
        
        # Create cache entry
        entry = CacheEntry(
            data=data,
            timestamp=time.time(),
            access_count=1,
            ttl=ttl,
            size_bytes=self._get_entry_size(data),
            cache_key=cache_key,
            metadata={'model': model, 'prompt_length': len(prompt)}
        )
        
        # Add to memory cache
        if len(self._memory_cache) >= self.max_memory_size:
            self._evict_from_memory()
        
        self._memory_cache[cache_key] = entry
        self._stats['memory_size'] += 1
        
        logger.debug(f"Cached response: {cache_key} (model: {model})")
    
    def _remove_from_disk(self, cache_key: str):
        """Remove entry from disk cache"""
        try:
            entry_path = self.cache_dir / 'entries' / f"{cache_key}.pkl"
            metadata_path = self.cache_dir / 'metadata' / f"{cache_key}.json"
            
            if entry_path.exists():
                entry_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
            
            if cache_key in self._disk_index:
                del self._disk_index[cache_key]
                self._stats['disk_size'] -= 1
                self._save_disk_index()
            
        except Exception as e:
            logger.error(f"Error removing from disk cache: {e}")
    
    def clear_memory_cache(self):
        """Clear memory cache"""
        self._memory_cache.clear()
        self._stats['memory_size'] = 0
        logger.info("Memory cache cleared")
    
    def clear_disk_cache(self):
        """Clear disk cache"""
        try:
            # Remove all cache files
            entries_dir = self.cache_dir / 'entries'
            metadata_dir = self.cache_dir / 'metadata'
            
            for file in entries_dir.glob('*.pkl'):
                file.unlink()
            for file in metadata_dir.glob('*.json'):
                file.unlink()
            
            # Clear index
            self._disk_index.clear()
            self._stats['disk_size'] = 0
            self._save_disk_index()
            
            logger.info("Disk cache cleared")
            
        except Exception as e:
            logger.error(f"Error clearing disk cache: {e}")
    
    def clear_all(self):
        """Clear both memory and disk caches"""
        self.clear_memory_cache()
        self.clear_disk_cache()
        
        # Reset statistics
        self._stats.update({
            'memory_hits': 0,
            'disk_hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0,
            'memory_size': 0,
            'disk_size': 0
        })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_hits = self._stats['memory_hits'] + self._stats['disk_hits']
        total_requests = self._stats['total_requests']
        
        hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
        memory_hit_rate = (self._stats['memory_hits'] / total_requests * 100) if total_requests > 0 else 0
        disk_hit_rate = (self._stats['disk_hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self._stats,
            'hit_rate_percent': round(hit_rate, 2),
            'memory_hit_rate_percent': round(memory_hit_rate, 2),
            'disk_hit_rate_percent': round(disk_hit_rate, 2),
            'memory_utilization_percent': round((self._stats['memory_size'] / self.max_memory_size * 100), 2),
            'disk_utilization_percent': round((self._stats['disk_size'] / self.max_disk_size * 100), 2)
        }
    
    def cleanup_expired(self):
        """Remove expired entries from both caches"""
        current_time = time.time()
        expired_keys = []
        
        # Check memory cache
        for cache_key, entry in self._memory_cache.items():
            if entry.is_expired():
                expired_keys.append(cache_key)
        
        for cache_key in expired_keys:
            del self._memory_cache[cache_key]
            self._stats['memory_size'] -= 1
        
        # Check disk cache
        disk_expired_keys = []
        for cache_key in self._disk_index:
            entry = self._load_from_disk(cache_key)
            if entry and entry.is_expired():
                disk_expired_keys.append(cache_key)
        
        for cache_key in disk_expired_keys:
            self._remove_from_disk(cache_key)
        
        logger.info(f"Cleaned up {len(expired_keys)} memory entries and {len(disk_expired_keys)} disk entries")
        
        return len(expired_keys) + len(disk_expired_keys)

def create_cache_manager(config_path: str = "config/cache_config.json") -> CacheManager:
    """Factory function to create cache manager with configuration"""
    
    # Default configuration
    default_config = {
        "max_memory_size": 100,
        "max_disk_size": 1000,
        "default_ttl": 3600,
        "cache_dir": "data/cache"
    }
    
    # Load configuration if file exists
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Merge with defaults
                default_config.update(config)
        except Exception as e:
            logger.error(f"Error loading cache config: {e}")
    
    return CacheManager(default_config)
