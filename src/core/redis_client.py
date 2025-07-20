"""
Redis client configuration for caching, sessions, and task queues
"""

import redis.asyncio as redis
from loguru import logger
from typing import Optional, Any
import json
import pickle
from .config import settings

# Global Redis connection
redis_client: Optional[redis.Redis] = None

async def init_redis():
    """Initialize Redis connection"""
    global redis_client
    
    try:
        redis_client = redis.from_url(
            settings.REDIS_URL,
            max_connections=settings.REDIS_MAX_CONNECTIONS,
            decode_responses=True,
            socket_keepalive=True,
            socket_keepalive_options={},
            health_check_interval=30,
        )
        
        # Test connection
        await redis_client.ping()
        logger.info("✅ Redis connected successfully")
        
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        raise

async def get_redis() -> redis.Redis:
    """Get Redis client instance"""
    if not redis_client:
        raise RuntimeError("Redis not initialized")
    return redis_client

async def close_redis():
    """Close Redis connections"""
    global redis_client
    if redis_client:
        await redis_client.close()
        logger.info("✅ Redis connections closed")

class RedisCache:
    """Redis cache wrapper with JSON and pickle serialization"""
    
    def __init__(self, client: redis.Redis):
        self.client = client
    
    async def get(self, key: str, use_pickle: bool = False) -> Any:
        """Get value from cache"""
        try:
            value = await self.client.get(key)
            if value is None:
                return None
            
            if use_pickle:
                return pickle.loads(value.encode('latin1'))
            else:
                return json.loads(value)
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        expire: Optional[int] = None,
        use_pickle: bool = False
    ) -> bool:
        """Set value in cache"""
        try:
            if use_pickle:
                serialized = pickle.dumps(value).decode('latin1')
            else:
                serialized = json.dumps(value, default=str)
            
            result = await self.client.set(key, serialized, ex=expire)
            return result
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            result = await self.client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            result = await self.client.exists(key)
            return result > 0
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment counter"""
        try:
            result = await self.client.incrby(key, amount)
            return result
        except Exception as e:
            logger.error(f"Cache increment error for key {key}: {e}")
            return None
    
    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration for key"""
        try:
            result = await self.client.expire(key, seconds)
            return result
        except Exception as e:
            logger.error(f"Cache expire error for key {key}: {e}")
            return False

# Global cache instance
cache: Optional[RedisCache] = None

async def get_cache() -> RedisCache:
    """Get cache instance"""
    global cache
    if not cache:
        redis_instance = await get_redis()
        cache = RedisCache(redis_instance)
    return cache

# Redis health check
async def check_redis_health() -> bool:
    """Check Redis connectivity"""
    try:
        if not redis_client:
            return False
        await redis_client.ping()
        return True
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return False

# Utility functions for common cache patterns
async def cache_model_metadata(model_name: str, metadata: dict, expire: int = 3600):
    """Cache model metadata"""
    cache_instance = await get_cache()
    key = f"model_metadata:{model_name}"
    await cache_instance.set(key, metadata, expire=expire)

async def get_cached_model_metadata(model_name: str) -> Optional[dict]:
    """Get cached model metadata"""
    cache_instance = await get_cache()
    key = f"model_metadata:{model_name}"
    return await cache_instance.get(key)

async def cache_conversation(session_id: str, conversation: list, expire: int = 86400):
    """Cache conversation history"""
    cache_instance = await get_cache()
    key = f"conversation:{session_id}"
    await cache_instance.set(key, conversation, expire=expire)

async def get_cached_conversation(session_id: str) -> Optional[list]:
    """Get cached conversation"""
    cache_instance = await get_cache()
    key = f"conversation:{session_id}"
    return await cache_instance.get(key)
