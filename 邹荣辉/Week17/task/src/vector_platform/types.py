"""平台共享类型定义.

本文件仅存放跨多个模块共享的基础类型，各模块独有类型在各自文件内定义：
  - milvus_index.py: DistanceMetric, IndexAlgorithm, StorageType, DataType,
                     FusionMethod, VectorEntry, QueryResult, Pydantic 字段模型
  - semantic_cache.py: CacheStatus
  - router.py: Route, RouteMatch
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


# ═══════════════════════════════════════════════════════════════════
# 缓存统计
# ═══════════════════════════════════════════════════════════════════


@dataclass
class CacheStats:
    """缓存统计信息 — SemanticCache 和 EmbeddingsCache 共用."""

    hits_exact: int = 0
    hits_semantic: int = 0
    misses: int = 0
    total_latency_ms: float = 0.0

    @property
    def total_requests(self) -> int:
        return self.hits_exact + self.hits_semantic + self.misses

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.hits_exact + self.hits_semantic) / self.total_requests

    @property
    def avg_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests


# ═══════════════════════════════════════════════════════════════════
# 对话消息
# ═══════════════════════════════════════════════════════════════════


@dataclass
class Message:
    """单条对话消息 — memory.py 使用."""

    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: float = 0.0
    token_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", 0.0),
            token_count=data.get("token_count", 0),
        )


# ═══════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════


def compute_hash(text: str) -> str:
    """计算文本的SHA-256哈希值 — 用于缓存键生成."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
