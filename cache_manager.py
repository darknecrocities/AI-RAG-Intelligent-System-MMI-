import logging
import json
import time
from typing import Optional, Any
import config

logger = logging.getLogger(__name__)

class MemoryCache:
    """
    Fallback in-memory cache with TTL (Time To Live).
    """
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            val, expire_time = self.cache[key]
            if expire_time is None or expire_time > time.time():
                return val
            else:
                del self.cache[key] # Expired
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        # Enforce max size limit
        if len(self.cache) >= self.max_size:
            # Evict first key (FIFO-ish fallback)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            
        expire_time = (time.time() + ttl) if ttl else None
        self.cache[key] = (value, expire_time)

    def clear(self):
        self.cache.clear()

class CacheManager:
    def __init__(self):
        self.redis_client = None
        self.redis_available = False
        
        try:
            import redis
            logger.info(f"Connecting to Redis at {config.REDIS_URL}...")
            self.redis_client = redis.from_url(config.REDIS_URL, socket_timeout=2)
            # Test connection
            self.redis_client.ping()
            self.redis_available = True
            logger.info("Connected to Redis successfully.")
        except Exception as e:
            logger.warning(f"Redis is unavailable: {e}. Falling back to In-Memory cache.")
            self.memory_cache = MemoryCache()

    def get(self, key: str) -> Optional[Any]:
        if self.redis_available:
            try:
                val = self.redis_client.get(key)
                if val:
                    return json.loads(val)
            except Exception as e:
                logger.error(f"Redis get error: {e}")
        else:
            return self.memory_cache.get(key)
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = 3600):
        """
        Saves key-value pair. Defaults TTL to 1 hour (3600s).
        """
        if self.redis_available:
            try:
                self.redis_client.set(
                    key, 
                    json.dumps(value, ensure_ascii=False), 
                    ex=ttl
                )
            except Exception as e:
                logger.error(f"Redis set error: {e}")
        else:
            self.memory_cache.set(key, value, ttl)

    def clear(self):
        if self.redis_available:
            try:
                self.redis_client.flushdb()
                logger.info("Redis cache cleared.")
            except Exception as e:
                logger.error(f"Redis flush error: {e}")
        else:
            self.memory_cache.clear()
            logger.info("In-memory cache cleared.")
