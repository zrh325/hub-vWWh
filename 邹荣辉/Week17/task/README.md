# Vector Platform — 向量检索与智能缓存服务平台

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Version](https://img.shields.io/badge/version-0.2.0-green.svg)](./pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](./LICENSE)

## 项目背景

随着公司各业务线对AI能力（大语言模型、RAG应用等）的深入探索，多个团队在独立开发中重复建设向量数据库接入、语义缓存、对话记忆管理等模块，导致技术栈碎片化、资源利用率低。**Vector Platform** 旨在构建一个公司内部统一、生产就绪的 Python SDK，充分利用已部署的高性能 Redis 集群，封装 AI 原生数据模式与操作，赋能各业务团队快速、低成本地构建高质量 AI 应用。

> 参考项目: [RedisVL](https://github.com/redis/redis-vl-python)

## 核心模块

本 SDK 提供五大核心模块，覆盖向量检索 → 智能缓存 → 对话记忆 → 语义路由的完整链路：

| 模块           | 文件                   | 功能                                         |
| ------------ | -------------------- | ------------------------------------------ |
| **统一向量数据管理** | `milvus_index.py`    | 索引 Schema 定义、索引生命周期管理、KNN 向量检索             |
| **混合查询引擎**   | `milvus_index.py`    | 向量相似性 + 元数据过滤 + 全文搜索的组合查询（RRF / 线性加权融合）    |
| **语义缓存**     | `semantic_cache.py`  | 两级命中策略（精确匹配 + 语义匹配），大幅降低 LLM 调用成本与延迟       |
| **嵌入缓存**     | `embedding_cache.py` | 缓存文本→向量转换结果，避免重复 Embedding 计算              |
| **对话记忆**     | `memory.py`          | 按 session\_id 分区存储，支持时间窗口检索 + 语义检索 + 跨会话搜索 |
| **语义路由**     | `router.py`          | 基于向量相似度的意图识别与路由匹配，支持 Redis 持久化             |

## 技术栈

| 组件                         | 用途                      |
| -------------------------- | ----------------------- |
| **Python 3.10+**           | SDK 主语言                 |
| **Redis** + **RediSearch** | 向量存储、全文搜索、缓存、会话管理       |
| **NumPy**                  | 向量计算（余弦相似度、融合排序等）       |
| **Pydantic v2**            | Schema 定义与数据验证          |
| **YAML**                   | 索引 Schema 文件化定义         |
| **sentence-transformers**  | 文本嵌入向量生成                |
| **structlog**              | 结构化日志                   |
| **FastAPI** (可选)           | HTTP 服务暴露               |
| **OpenTelemetry** (可选)     | 可观测性（Metrics / Tracing） |

## 项目结构

```
task/
├── src/vector_platform/          # SDK 源码
│   ├── __init__.py               # 公开 API 导出 (v0.2.0)
│   ├── milvus_index.py           # 索引 Schema + 向量/全文/混合查询引擎
│   ├── semantic_cache.py         # 语义缓存（二级命中：精确+语义）
│   ├── embedding_cache.py        # 嵌入缓存
│   ├── memory.py                 # 对话历史管理（支持语义检索）
│   ├── router.py                 # 语义路由（意图识别）
│   └── types.py                  # 共享类型（CacheStats, Message, compute_hash）
├── tests/                        # 测试套件
│   ├── conftest.py               # 共享 fixtures（mock Redis 等）
│   ├── test_schema.py            # Schema 相关测试
│   ├── test_cache.py             # 缓存模块测试
│   ├── test_memory.py            # 对话记忆测试
│   ├── test_query.py             # 查询引擎测试
│   ├── test_routing.py           # 语义路由测试
│   ├── test_types.py             # 共享类型测试
│   └── test_service.py           # 服务层测试
├── examples/
│   ├── quickstart.py             # 端到端快速上手示例
│   └── schema_example.yaml       # 索引 Schema YAML 示例
├── pyproject.toml                # 项目配置与依赖声明
├── README.md                     # 本文件
└── CLAUDE.md                     # Claude Code 辅助文件
```

## 快速开始

### 安装

```bash
# 基础安装
pip install -e .

# 含 FastAPI 服务
pip install -e ".[fastapi]"

# 含 OpenAI 集成
pip install -e ".[openai]"

# 含可观测性
pip install -e ".[prometheus]"

# 全部可选依赖
pip install -e ".[all]"
```

### 基本使用

```python
from redis import Redis
import numpy as np
from vector_platform import (
    IndexSchema, SearchIndex, VectorQuery, HybridQuery, QueryBuilder,
    SemanticCache, EmbeddingsCache,
    SemanticMessageHistory, SemanticRouter, Route, RouteMatch,
    Message, CacheStats,
)

# 1. 连接 Redis
client = Redis(host="localhost", port=6379, decode_responses=False)

# 2. 定义索引 Schema（YAML 或编程方式）
schema = IndexSchema.from_yaml("examples/schema_example.yaml")
# 或
schema = IndexSchema(
    index={"name": "docs", "prefix": "doc", "storage_type": "hash"},
    fields=[
        {"name": "content", "type": "text"},
        {"name": "embedding", "type": "vector", "dims": 384, "distance_metric": "cosine"},
        {"name": "category", "type": "tag"},
    ],
)

# 3. 创建索引 & 灌入数据
index = SearchIndex(schema, client)
index.create(overwrite=True)
index.load([
    {"id": "1", "content": "机器学习入门指南", "embedding": vec1, "category": ["AI"]},
    {"id": "2", "content": "深度学习框架对比", "embedding": vec2, "category": ["AI", "DL"]},
])

# 4. 向量检索
vq = VectorQuery(schema, client)
results = vq.search(vector=query_vec, top_k=10, filter_expr="@category:{AI}")

# 5. 混合查询（向量 + 过滤 + 全文，RRF 融合）
results = (
    QueryBuilder(schema, client)
    .vector(query_vec)
    .filter("@category:{AI}")
    .text("机器学习")
    .top_k(20)
    .fusion("rrf")
    .execute()
)

# 6. 语义缓存 — 避免重复调用 LLM
cache = SemanticCache(client, vectorizer, distance_threshold=0.15)
result = cache.lookup("什么是向量数据库？")
if result is None:
    response = llm.chat("什么是向量数据库？")
    cache.store("什么是向量数据库？", response)

# 7. 对话记忆 — 按 session 管理 + 语义搜索
memory = SemanticMessageHistory(client, vectorizer, ttl=86400)
memory.create_session("session_123", model="gpt-4")
memory.add("session_123", Message(role="user", content="你好"))
memory.add("session_123", Message(role="assistant", content="你好！有什么可以帮助你的？"))
# 语义搜索历史消息
results = memory.search("session_123", "问候语", top_k=5)

# 8. 语义路由 — 意图识别
router = SemanticRouter(vectorizer, client, distance_threshold=0.3)
router.add_route("weather", "天气查询", examples=["今天天气怎么样？"])
router.add_route("math", "数学计算", examples=["1+1等于多少？"])
router.set_default("general")
match = router.match("今天会下雨吗？")
print(match.route.name)  # "weather"
```

## 缓存策略

### 语义缓存 — 两级命中

```
用户请求
  └─ 第1级: SHA-256 精确匹配 → Redis HASH (O(1))
      └─ 命中 → 直接返回 🎯
      └─ 未命中 →
          第2级: Embedding KNN 语义搜索 → 余弦相似度匹配
              └─ 距离 ≤ threshold → 返回语义相似结果 🔍
              └─ 超过阈值 → 调用 LLM 并缓存结果 💾
```

### 嵌入缓存

```
文本 → SHA-256(text) → Redis HASH 查找
  ├─ 命中 → 直接返回向量 (跳过 Embedding API 调用)
  └─ 未命中 → 调用 Embedding API → 存储缓存 → 返回向量
```

## 运行测试

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行全部测试
pytest

# 含覆盖率
pytest --cov=vector_platform --cov-report=term-missing

# 仅运行特定模块测试
pytest tests/test_cache.py -v
```

## 设计决策

- **Redis 为核心基础设施** — SDK 封装 Redis，不替代它。利用公司已有的 Redis 集群
- **Session 作用域** — 对话缓存按 `session_id` 分区存储，天然支持多租户
- **Duck Typing** — 向量器接口不做强制类型约束，任何实现了 `.encode(text) -> np.ndarray` 的对象均可接入
- **可选持久化** — 语义路由支持内存模式（无 Redis）和持久化模式（Redis），灵活适配不同场景
- **RediSearch 原生能力** — 向量检索、全文搜索、标签过滤均直接使用 RediSearch 模块，避免引入额外组件

