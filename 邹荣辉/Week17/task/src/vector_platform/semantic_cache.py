"""Semantic Cache — 语义缓存.

两级命中策略：
  1. 精确匹配：SHA-256 哈希 → Redis HASH (O(1))
  2. 语义匹配：嵌入向量 KNN 搜索 → 相似度阈值匹配

用于缓存 LLM 请求-响应对，显著降低模型调用成本和响应延迟。

Usage:
    from redis import Redis
    from vector_platform.semantic_cache import SemanticCache

    redis_client = Redis(host="localhost", port=6379, decode_responses=False)
    cache = SemanticCache(
        client=redis_client,
        vectorizer=my_vectorizer,  # 任何有 .encode(text) -> np.ndarray 的对象
        distance_threshold=0.15,
        ttl=3600,
    )

    # 检查缓存
    cached = cache.lookup("什么是向量数据库？")
    if cached:
        return cached["response"]

    # 调用 LLM
    response = llm.chat("什么是向量数据库？")

    # 存储缓存
    cache.store("什么是向量数据库？", response)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Optional

import numpy as np
from redis import Redis

from vector_platform.types import CacheStats, compute_hash

logger = logging.getLogger(__name__)


class SemanticCache:
    """语义缓存 — 两级命中策略的 LLM 请求缓存.

    第一级（精确匹配）: SHA-256(request) → 直接返回缓存响应
    第二级（语义匹配）: embedding(request) → KNN搜索 → 相似度高则返回

    Usage:
        cache = SemanticCache(redis_client, vectorizer, distance_threshold=0.15)

        # 查询缓存
        cached = cache.lookup("什么是向量数据库？")
        if cached:
            return cached["response"]

        # 调用 LLM 后存储
        response = llm.chat("什么是向量数据库？")
        cache.store("什么是向量数据库？", response)
    """

    def __init__(
        self,
        client: Redis,
        vectorizer,  # duck typing: .encode(text) -> np.ndarray
        name: str = "llm-semantic-cache",
        distance_threshold: float = 0.15,
        ttl: int = 3600,
        prefix: str = "semcache:",
        enable_stats: bool = True,
        max_semantic_scan: int = 500,
    ):
        """初始化语义缓存.

        Args:
            client: Redis 客户端
            vectorizer: 向量器 — 需要实现 .encode(text) -> np.ndarray
            name: 缓存命名空间
            distance_threshold: 语义相似距离阈值（低于此值视为命中，范围 0~1）
            ttl: 缓存过期时间（秒），0 表示永不过期
            prefix: Key 前缀
            enable_stats: 是否启用统计
            max_semantic_scan: 语义匹配时最多扫描的条目数（LRU 近似）
        """
        self._client = client
        self._vectorizer = vectorizer
        self._name = name
        self._distance_threshold = distance_threshold
        self._ttl = ttl
        self._prefix = prefix
        self._enable_stats = enable_stats
        self._max_semantic_scan = max_semantic_scan
        self._stats = CacheStats()

    # ─── 属性 ────────────────────────────────────────────────────

    @property
    def stats(self) -> CacheStats:
        """只读统计信息."""
        return self._stats

    @property
    def distance_threshold(self) -> float:
        return self._distance_threshold

    @property
    def max_semantic_scan(self) -> int:
        return self._max_semantic_scan

    # ─── Key 生成 ─────────────────────────────────────────────────

    def _exact_key(self, text: str) -> str:
        """精确匹配 Key."""
        return f"{self._prefix}{self._name}:exact:{compute_hash(text)}"

    def _semantic_key(self, entry_id: str) -> str:
        """语义缓存条目 Key."""
        return f"{self._prefix}{self._name}:entry:{entry_id}"

    def _embedding_key(self) -> str:
        """语义嵌入集合 Key (ZSET)."""
        return f"{self._prefix}{self._name}:embeddings"

    # ─── 查询 ────────────────────────────────────────────────────

    def lookup(self, prompt: str, **context: Any) -> Optional[dict[str, Any]]:
        """查询缓存 — 先精确匹配，再语义匹配.

        Args:
            prompt: 用户输入
            **context: 可选的上下文元数据（保留用于未来扩展）

        Returns:
            命中时返回 {"response": ..., "score": ..., "match_type": ..., "latency_ms": ...}
            未命中返回 None
        """
        start = time.perf_counter()

        # ─── 第一级：精确匹配 ──────────────────────────
        exact_key = self._exact_key(prompt)
        try:
            cached = self._client.hgetall(exact_key)
        except Exception:
            cached = {}

        if cached:
            elapsed_ms = (time.perf_counter() - start) * 1000
            if self._enable_stats:
                self._stats.hits_exact += 1
                self._stats.total_latency_ms += elapsed_ms

            response = cached.get(b"response", cached.get("response", ""))
            if isinstance(response, bytes):
                response = response.decode()

            return {
                "response": response,
                "score": 1.0,
                "match_type": "exact",
                "latency_ms": elapsed_ms,
            }

        # ─── 第二级：语义匹配 ──────────────────────────
        query_vec = self._vectorizer.encode(prompt)
        query_blob = query_vec.astype(np.float32).tobytes()

        # 获取最近的 N 个缓存条目（LRU 近似 — ZSET 按向量模长排序）
        try:
            embedding_key = self._embedding_key()
            members = self._client.zrevrange(
                embedding_key, 0, self._max_semantic_scan - 1
            )
        except Exception:
            members = []

        best_score = 0.0
        best_entry_id: Optional[str] = None

        for member in members:
            if isinstance(member, bytes):
                member = member.decode()

            entry_key = self._semantic_key(member)
            try:
                entry_vec_blob = self._client.hget(entry_key, "embedding")
            except Exception:
                continue

            if entry_vec_blob is None:
                continue

            entry_vec = np.frombuffer(entry_vec_blob, dtype=np.float32)
            similarity = self._cosine_similarity(query_vec, entry_vec)
            if similarity > best_score:
                best_score = similarity
                best_entry_id = member

        # 距离阈值检查
        distance = 1.0 - best_score
        if best_entry_id is not None and distance <= self._distance_threshold:
            entry_key = self._semantic_key(best_entry_id)
            try:
                entry_data = self._client.hgetall(entry_key)
            except Exception:
                entry_data = {}

            response = entry_data.get(b"response", entry_data.get("response", ""))
            if isinstance(response, bytes):
                response = response.decode()

            elapsed_ms = (time.perf_counter() - start) * 1000
            if self._enable_stats:
                self._stats.hits_semantic += 1
                self._stats.total_latency_ms += elapsed_ms

            return {
                "response": response,
                "score": float(best_score),
                "match_type": "semantic",
                "distance": float(distance),
                "latency_ms": elapsed_ms,
            }

        # ─── 未命中 ────────────────────────────────────
        elapsed_ms = (time.perf_counter() - start) * 1000
        if self._enable_stats:
            self._stats.misses += 1
            self._stats.total_latency_ms += elapsed_ms

        return None

    # ─── 存储 ────────────────────────────────────────────────────

    def store(
        self,
        prompt: str,
        response: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """存储请求-响应对到语义缓存.

        Args:
            prompt: 用户输入
            response: LLM 响应
            metadata: 附加元数据

        Returns:
            是否存储成功
        """
        entry_id = str(uuid.uuid4())[:12]

        # 精确匹配数据
        exact_key = self._exact_key(prompt)
        exact_data: dict[str, Any] = {"response": response}
        if metadata:
            exact_data["metadata"] = json.dumps(metadata, ensure_ascii=False)

        # 语义嵌入数据
        embedding = self._vectorizer.encode(prompt)
        entry_key = self._semantic_key(entry_id)
        entry_data: dict[str, Any] = {
            "prompt": prompt,
            "response": response,
            "embedding": embedding.astype(np.float32).tobytes(),
        }
        if metadata:
            entry_data["metadata"] = json.dumps(metadata, ensure_ascii=False)

        # 计算嵌入的标量分数用于 ZSET 排序（向量模长）
        score = float(np.linalg.norm(embedding))

        try:
            pipe = self._client.pipeline()

            # 精确匹配 Key
            pipe.hset(exact_key, mapping=exact_data)
            if self._ttl > 0:
                pipe.expire(exact_key, self._ttl)

            # 语义缓存 Key
            pipe.hset(entry_key, mapping=entry_data)
            if self._ttl > 0:
                pipe.expire(entry_key, self._ttl)

            # 加入嵌入集合
            pipe.zadd(self._embedding_key(), {entry_id: score})
            if self._ttl > 0:
                pipe.expire(self._embedding_key(), self._ttl)

            pipe.execute()
            return True
        except Exception as e:
            logger.error("Failed to store semantic cache entry: %s", e)
            return False

    # ─── 管理 ────────────────────────────────────────────────────

    def clear(self) -> int:
        """清空此命名空间下的所有缓存.

        Returns:
            删除的 Key 数量
        """
        pattern = f"{self._prefix}{self._name}:*"
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

    def reset_stats(self) -> None:
        """重置统计信息."""
        self._stats = CacheStats()

    # ─── 内部方法 ─────────────────────────────────────────────────

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def __repr__(self) -> str:
        return (
            f"<SemanticCache name='{self._name}' "
            f"threshold={self._distance_threshold} ttl={self._ttl}s "
            f"max_scan={self._max_semantic_scan}>"
        )
