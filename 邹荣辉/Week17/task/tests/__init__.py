"""
向量检索与智能缓存服务平台 — Python SDK.

五大核心模块:
    - milvus_index: 索引 Schema 定义、索引生命周期管理、向量/全文/混合检索
    - semantic_cache: 语义缓存（LLM 请求-响应缓存）
    - embedding_cache: 嵌入缓存（避免重复 Embedding 计算）
    - memory: 对话历史管理（时间检索 + 语义检索）
    - router: 语义路由（意图识别）
"""

__version__ = "0.2.0"

# ─── File 1: Schema & Data Retrieval ─────────────────────────
from vector_platform.milvus_index import (
    # 枚举
    DataType,
    DistanceMetric,
    FusionMethod,
    IndexAlgorithm,
    StorageType,
    # 数据类
    QueryResult,
    VectorEntry,
    # 字段 Pydantic 模型
    GeoField,
    NumericField,
    TagField,
    TextField,
    VectorField,
    # Schema
    IndexSchema,
    # 索引管理
    SearchIndex,
    # 查询引擎
    FilterQuery,
    HybridQuery,
    QueryBuilder,
    TextQuery,
    VectorQuery,
    # 融合函数
    linear_weighted_fusion,
    reciprocal_rank_fusion,
)

# ─── File 2: Semantic Caching ────────────────────────────────
from vector_platform.semantic_cache import SemanticCache

# ─── File 3: Embedding Caching ───────────────────────────────
from vector_platform.embedding_cache import EmbeddingsCache

# ─── File 4: Conversation Memory ─────────────────────────────
from vector_platform.memory import SemanticMessageHistory

# ─── File 5: Semantic Router ─────────────────────────────────
from vector_platform.router import Route, RouteMatch, SemanticRouter

# ─── Shared Types ────────────────────────────────────────────
from vector_platform.types import CacheStats, Message, compute_hash

__all__ = [
    # milvus_index
    "IndexSchema",
    "SearchIndex",
    "VectorQuery",
    "TextQuery",
    "FilterQuery",
    "HybridQuery",
    "QueryBuilder",
    "QueryResult",
    "VectorEntry",
    "DistanceMetric",
    "IndexAlgorithm",
    "StorageType",
    "DataType",
    "FusionMethod",
    "VectorField",
    "TagField",
    "TextField",
    "NumericField",
    "GeoField",
    "reciprocal_rank_fusion",
    "linear_weighted_fusion",
    # semantic_cache
    "SemanticCache",
    # embedding_cache
    "EmbeddingsCache",
    # memory
    "SemanticMessageHistory",
    # router
    "SemanticRouter",
    "Route",
    "RouteMatch",
    # types
    "CacheStats",
    "Message",
    "compute_hash",
]
