"""Embedding Cache — 嵌入向量缓存.

缓存文本→向量的转换结果，避免对相同内容进行重复的嵌入计算，
从而减少 Embedding 模型调用成本和延迟。

基于 Redis HASH 存储，以文本 SHA-256 为 Key，支持 TTL 自动过期。

Usage:
    from redis import Redis
    from vector_platform.embedding_cache import EmbeddingsCache

    redis_client = Redis(host="localhost", port=6379, decode_responses=False)
    cache = EmbeddingsCache(redis_client, ttl=86400, prefix="embcache:")

    # 查询缓存
    cached_vec = cache.get("你好世界")
    if cached_vec is not None:
        embedding = cached_vec  # 命中！跳过嵌入计算
    else:
        embedding = some_vectorizer.encode("你好世界")
        cache.set("你好世界", embedding)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np
from redis import Redis

from vector_platform.types import CacheStats, compute_hash

logger = logging.getLogger(__name__)


class EmbeddingsCache:
    """嵌入缓存 — 避免对相同文本重复进行嵌入计算.

    基于 Redis HASH 存储，以文本 SHA-256 为 Key，支持 TTL 自动过期。

    Usage:
        cache = EmbeddingsCache(redis_client, ttl=86400, prefix="embcache:")

        # 存储
        cache.set("你好世界", embedding_vector)

        # 查询
        cached = cache.get("你好世界")
        if cached is not None:
            embedding = cached  # 命中！
    """

    def __init__(
        self,
        client: Redis,
        ttl: int = 86400,  # 默认24小时
        prefix: str = "embcache:",
        enable_stats: bool = True,
    ):
        """初始化嵌入缓存.

        Args:
            client: Redis 客户端
            ttl: 缓存过期时间（秒），0 表示永不过期
            prefix: Key 前缀，用于命名空间隔离
            enable_stats: 是否启用统计
        """
        self._client = client
        self._ttl = ttl
        self._prefix = prefix
        self._enable_stats = enable_stats
        self._stats = CacheStats()

    # ─── 属性 ────────────────────────────────────────────────────

    @property
    def stats(self) -> CacheStats:
        """只读统计信息."""
        return self._stats

    @property
    def ttl(self) -> int:
        return self._ttl

    @property
    def prefix(self) -> str:
        return self._prefix

    # ─── Key 生成 ─────────────────────────────────────────────────

    def _make_key(self, text: str) -> str:
        """生成缓存Key: prefix + SHA-256(text)."""
        return f"{self._prefix}{compute_hash(text)}"

    # ─── 核心操作 ─────────────────────────────────────────────────

    def get(self, text: str) -> Optional[np.ndarray]:
        """查询缓存的嵌入向量.

        Args:
            text: 输入文本

        Returns:
            缓存的向量（命中）或 None（未命中）
        """
        key = self._make_key(text)
        start = time.perf_counter()

        try:
            blob = self._client.hget(key, "embedding")
        except Exception as e:
            logger.warning("Failed to get embedding for text: %s", e)
            return None

        elapsed_ms = (time.perf_counter() - start) * 1000

        if blob is not None:
            if self._enable_stats:
                self._stats.hits_exact += 1
                self._stats.total_latency_ms += elapsed_ms
            return np.frombuffer(blob, dtype=np.float32)

        if self._enable_stats:
            self._stats.misses += 1
            self._stats.total_latency_ms += elapsed_ms
        return None

    def set(
        self,
        text: str,
        embedding: np.ndarray,
        model_name: Optional[str] = None,
        extra_meta: Optional[dict[str, Any]] = None,
    ) -> bool:
        """存储嵌入向量.

        Args:
            text: 输入文本
            embedding: 嵌入向量 (np.ndarray)
            model_name: 嵌入模型名称（用于追踪）
            extra_meta: 额外元数据

        Returns:
            是否存储成功
        """
        key = self._make_key(text)
        mapping: dict[str, Any] = {
            "embedding": embedding.astype(np.float32).tobytes(),
            "text_hash": compute_hash(text),
        }
        if model_name:
            mapping["model"] = model_name
        if extra_meta:
            import json
            mapping["meta"] = json.dumps(extra_meta, ensure_ascii=False)

        try:
            pipe = self._client.pipeline()
            pipe.hset(key, mapping=mapping)
            if self._ttl > 0:
                pipe.expire(key, self._ttl)
            pipe.execute()
            return True
        except Exception as e:
            logger.error("Failed to set embedding for text: %s", e)
            return False

    def exists(self, text: str) -> bool:
        """检查文本是否已缓存嵌入向量.

        Args:
            text: 输入文本

        Returns:
            是否存在缓存
        """
        key = self._make_key(text)
        try:
            return bool(self._client.exists(key))
        except Exception:
            return False

    def delete(self, text: str) -> bool:
        """删除指定文本的缓存.

        Args:
            text: 输入文本

        Returns:
            是否删除成功
        """
        key = self._make_key(text)
        try:
            return bool(self._client.delete(key))
        except Exception:
            return False

    def clear(self) -> int:
        """清空所有嵌入缓存（按前缀匹配 SCAN + DELETE）.

        Returns:
            删除的 Key 数量
        """
        pattern = f"{self._prefix}*"
        keys: list[bytes] = []
        try:
            cursor = 0
            while True:
                cursor, batch = self._client.scan(cursor, match=pattern, count=1000)
                keys.extend(batch)
                if cursor == 0:
                    break
        except Exception as e:
            logger.error("Failed to scan keys for clear: %s", e)
            return 0

        if keys:
            try:
                return self._client.delete(*keys)
            except Exception as e:
                logger.error("Failed to delete keys: %s", e)
        return 0

    def count(self) -> int:
        """统计缓存的条目数（按前缀 SCAN 计数）.

        Returns:
            条目数量
        """
        pattern = f"{self._prefix}*"
        count = 0
        try:
            cursor = 0
            while True:
                cursor, batch = self._client.scan(cursor, match=pattern, count=1000)
                count += len(batch)
                if cursor == 0:
                    break
        except Exception as e:
            logger.error("Failed to count keys: %s", e)
        return count

    def reset_stats(self) -> None:
        """重置统计信息."""
        self._stats = CacheStats()

    def __repr__(self) -> str:
        return f"<EmbeddingsCache prefix='{self._prefix}' ttl={self._ttl}s>"
