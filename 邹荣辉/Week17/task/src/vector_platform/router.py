"""Semantic Router — 语义路由.

基于向量相似度匹配用户意图到预定义路由：
  1. 将用户输入嵌入到向量空间
  2. 匹配最近的路由定义
  3. 距离低于阈值时路由到对应处理器
  4. 超过阈值时回退到默认路由

支持可选 Redis 持久化，路由定义可跨进程共享。

Usage:
    from vector_platform.router import SemanticRouter, Route, RouteMatch

    router = SemanticRouter(vectorizer=my_vectorizer, distance_threshold=0.3)

    # 注册路由
    router.add_route("weather", "Weather related queries", examples=["今天天气怎么样？"])
    router.add_route("math", "Math calculation requests", examples=["1+1等于多少？"])
    router.set_default("general")

    # 匹配意图
    match = router.match("今天会下雨吗？")
    print(match.route.name)  # "weather"
    print(match.score)       # 0.92
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from redis import Redis

logger = logging.getLogger(__name__)


# ─── 类型定义 ────────────────────────────────────────────────────


@dataclass
class Route:
    """单条路由定义."""

    name: str
    description: str
    embedding: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RouteMatch:
    """路由匹配结果."""

    route: Route
    score: float
    matched: bool = True
    fallback: bool = False


# ─── SemanticRouter ──────────────────────────────────────────────


class SemanticRouter:
    """语义路由 — 基于向量相似度匹配用户意图到处理器.

    支持可选 Redis 持久化（传入 client 参数时启用）。

    Usage:
        router = SemanticRouter(vectorizer, distance_threshold=0.3)

        # 注册路由
        router.add_route("weather", "Weather related queries")
        router.add_route("math", "Math calculation requests")
        router.set_default("general")

        # 匹配
        match = router.match("What's the weather today?")
        print(match.route.name)  # "weather"
    """

    # 默认路由持久化命名空间
    DEFAULT_NAMESPACE = "default"

    def __init__(
        self,
        vectorizer,  # duck typing: .encode(text) -> np.ndarray
        client: Optional[Redis] = None,
        distance_threshold: float = 0.3,
        default_route: str = "general",
        namespace: str = DEFAULT_NAMESPACE,
    ):
        """初始化语义路由.

        Args:
            vectorizer: 向量器 — 需要实现 .encode(text) -> np.ndarray
            client: Redis 客户端（可选，用于持久化路由定义）
            distance_threshold: 路由匹配距离阈值（低于此值视为匹配，范围 0~1）
            default_route: 默认路由名称（未匹配时回退到此路由）
            namespace: Redis 持久化命名空间（仅 client 不为 None 时生效）
        """
        self._vectorizer = vectorizer
        self._client = client
        self._distance_threshold = distance_threshold
        self._default_route_name = default_route
        self._namespace = namespace
        self._routes: dict[str, Route] = {}

    # ─── 属性 ────────────────────────────────────────────────────

    @property
    def routes(self) -> dict[str, Route]:
        """返回路由字典的副本."""
        return dict(self._routes)

    @property
    def default_route(self) -> str:
        return self._default_route_name

    @property
    def distance_threshold(self) -> float:
        return self._distance_threshold

    # ─── Redis Key ────────────────────────────────────────────────

    def _routes_key(self) -> str:
        return f"router:{self._namespace}:routes"

    # ─── 路由管理 ────────────────────────────────────────────────

    def add_route(
        self,
        name: str,
        description: str,
        examples: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Route:
        """添加一条路由.

        Args:
            name: 路由名称（唯一标识）
            description: 路由描述（用于生成嵌入向量）
            examples: 示例输入（可选，用于增强嵌入质量）
            metadata: 附加元数据

        Returns:
            创建的 Route 对象
        """
        # 拼接描述和示例生成嵌入向量
        text_to_embed = description
        if examples:
            text_to_embed += " " + " ".join(examples)

        embedding = self._vectorizer.encode(text_to_embed)

        route = Route(
            name=name,
            description=description,
            embedding=embedding,
            metadata=metadata or {},
        )
        self._routes[name] = route
        logger.info("Added route: %s (desc=%r)", name, description[:50])

        # 可选持久化到 Redis
        if self._client is not None:
            self._save_route_to_redis(route)

        return route

    def remove_route(self, name: str) -> bool:
        """移除一条路由.

        Args:
            name: 路由名称

        Returns:
            是否成功移除
        """
        if name in self._routes:
            del self._routes[name]
            # 从 Redis 中移除
            if self._client is not None:
                try:
                    self._client.hdel(self._routes_key(), name)
                except Exception as e:
                    logger.warning("Failed to remove route from Redis: %s", e)
            logger.info("Removed route: %s", name)
            return True
        return False

    def set_default(self, route_name: str) -> None:
        """设置默认路由名称.

        Args:
            route_name: 回退路由名称
        """
        self._default_route_name = route_name

    # ─── 路由匹配 ────────────────────────────────────────────────

    def match(self, query: str) -> RouteMatch:
        """匹配最合适的路由.

        将 query 编码为向量，与所有已注册路由的嵌入向量计算余弦相似度，
        选择相似度最高且距离低于阈值的路由。若没有符合条件的路由，回退到默认路由。

        Args:
            query: 用户输入文本

        Returns:
            RouteMatch 结果
        """
        if not self._routes:
            return RouteMatch(
                route=Route(name=self._default_route_name, description="default"),
                score=0.0,
                matched=False,
                fallback=True,
            )

        query_vec = self._vectorizer.encode(query)

        best_score = -1.0
        best_route: Optional[Route] = None

        for route in self._routes.values():
            if route.embedding is not None:
                score = float(self._cosine_similarity(query_vec, route.embedding))
                if score > best_score:
                    best_score = score
                    best_route = route

        distance = 1.0 - best_score

        if best_route is not None and distance <= self._distance_threshold:
            return RouteMatch(route=best_route, score=best_score)
        else:
            # 回退到默认路由
            fallback_route = self._routes.get(
                self._default_route_name,
                Route(name=self._default_route_name, description="fallback"),
            )
            return RouteMatch(
                route=fallback_route,
                score=best_score if best_score > 0 else 0.0,
                matched=False,
                fallback=True,
            )

    def match_name(self, query: str) -> str:
        """匹配路由并直接返回路由名称.

        Args:
            query: 用户输入文本

        Returns:
            路由名称字符串
        """
        return self.match(query).route.name

    def find_similar_routes(
        self, description: str, top_k: int = 3
    ) -> list[tuple[str, float]]:
        """查找与描述最相似的路由（用于路由发现/推荐）.

        Args:
            description: 路由描述文本
            top_k: 返回最相似的 K 个路由

        Returns:
            [(路由名称, 相似度分数), ...] 按分数降序排列
        """
        query_vec = self._vectorizer.encode(description)
        scores: list[tuple[str, float]] = []

        for route in self._routes.values():
            if route.embedding is not None:
                score = float(self._cosine_similarity(query_vec, route.embedding))
                scores.append((route.name, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    # ─── Redis 持久化 ─────────────────────────────────────────────

    def load_routes(self) -> int:
        """从 Redis 恢复已持久化的路由定义.

        需要初始化时传入了 Redis client。

        Returns:
            恢复的路由数量
        """
        if self._client is None:
            logger.warning("No Redis client configured, cannot load routes")
            return 0

        try:
            raw = self._client.hgetall(self._routes_key())
        except Exception as e:
            logger.error("Failed to load routes from Redis: %s", e)
            return 0

        loaded = 0
        for name_bytes, data_bytes in raw.items():
            try:
                name = name_bytes.decode() if isinstance(name_bytes, bytes) else name_bytes
                data_str = data_bytes.decode() if isinstance(data_bytes, bytes) else data_bytes
                data = json.loads(data_str)

                # 恢复嵌入向量
                embedding_blob = bytes(data["embedding_blob"])
                embedding = np.frombuffer(embedding_blob, dtype=np.float32)

                route = Route(
                    name=data["name"],
                    description=data.get("description", ""),
                    embedding=embedding,
                    metadata=data.get("metadata", {}),
                )
                self._routes[name] = route
                loaded += 1
            except Exception as e:
                logger.warning("Failed to parse route entry: %s", e)

        logger.info("Loaded %d routes from Redis", loaded)
        return loaded

    def _save_route_to_redis(self, route: Route) -> None:
        """将单条路由持久化到 Redis."""
        if self._client is None:
            return

        if route.embedding is None:
            return

        data = {
            "name": route.name,
            "description": route.description,
            "embedding_blob": list(route.embedding.astype(np.float32).tobytes()),
            "metadata": route.metadata,
        }

        try:
            self._client.hset(
                self._routes_key(),
                route.name,
                json.dumps(data, ensure_ascii=False),
            )
        except Exception as e:
            logger.warning("Failed to persist route '%s' to Redis: %s", route.name, e)

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
            f"<SemanticRouter routes={list(self._routes.keys())} "
            f"default='{self._default_route_name}' threshold={self._distance_threshold}>"
        )
