# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

构建一个公司内部统一的、生产就绪的**向量检索与智能缓存服务平台**。核心是利用公司已部署的高性能 Redis 集群，封装 AI 原生数据模式与操作，为上层业务提供 Python SDK 与配套服务。

参考项目: [RedisVL](https://github.com/redis/redis-vl-python)

当前版本: **v0.2.0** — 已实现完整的 Python SDK，含测试覆盖。

## Tech Stack

- **Python 3.10+** — SDK 主语言
- **Redis + RediSearch** — 向量存储、全文搜索、缓存、会话管理
- **NumPy** — 向量计算（余弦相似度、融合排序等）
- **Pydantic v2** — Schema 定义与验证
- **YAML** — 索引 Schema 文件化定义
- **sentence-transformers** — 文本嵌入（Duck Typing，可替换）
- **structlog** — 结构化日志
- **FastAPI** (optional) — HTTP 服务暴露
- **OpenTelemetry** (optional) — 可观测性
- **Redis 是已有基础设施** — SDK 封装它，不替代它

## Package Structure

```
src/vector_platform/
├── __init__.py           # 公开 API 导出，__version__ = "0.2.0"
├── types.py              # 共享类型: CacheStats, Message, compute_hash()
├── milvus_index.py       # 索引管理 + 查询引擎 (最大文件 ~1280行)
│   ├── 枚举: DistanceMetric, IndexAlgorithm, StorageType, DataType, FusionMethod
│   ├── 字段模型: VectorField, TagField, TextField, NumericField, GeoField
│   ├── IndexSchema       — Schema 定义 (YAML/编程), RediSearch FT.CREATE 生成
│   ├── SearchIndex       — 索引生命周期 (create/drop/exists/load/insert/delete)
│   ├── VectorQuery       — KNN 向量检索 (FT.SEARCH KNN)
│   ├── TextQuery         — 全文搜索 (BM25/TFIDF)
│   ├── FilterQuery       — 元数据过滤 (Tag/Numeric/Geo)
│   ├── HybridQuery       — 混合查询引擎 (向量+过滤+全文, RRF/线性加权)
│   ├── QueryBuilder      — 流式查询构建器 (链式API)
│   └── 融合函数: reciprocal_rank_fusion, linear_weighted_fusion
├── semantic_cache.py     # 语义缓存 — 二级命中 (精确SHA256 + 语义KNN)
├── embedding_cache.py    # 嵌入缓存 — 文本→向量映射 (Redis HASH)
├── memory.py             # 对话记忆 — session分区 + 时间检索 + 语义检索 + 跨会话搜索
└── router.py             # 语义路由 — 向量意图匹配 + Redis可选持久化
```

## Key Design Decisions

- **Duck Typing for Vectorizer** — 所有需要嵌入的模块接受任何有 `.encode(text) -> np.ndarray` 的对象，不强制类型
- **Cache storage is session-scoped** — 对话按 `session_id` 分区存储
- **Two-level cache hit** — SemanticCache 先精确匹配再语义匹配
- **Redis key patterns**:
  - `{prefix}:{name}:exact:{sha256}` — 精确缓存
  - `{prefix}:{name}:entry:{uuid}` — 语义缓存条目
  - `{prefix}:{name}:embeddings` — 嵌入向量 ZSET
  - `session:{id}:messages` / `session:{id}:meta` / `session:{id}:msg_emb:*` — 对话记忆
  - `router:{namespace}:routes` — 路由持久化
- **RediSearch as primary query engine** — 向量检索、全文搜索、标签过滤均走 RediSearch，不引入额外向量数据库
- **Optional Redis persistence** — SemanticRouter 支持纯内存模式和 Redis 持久化模式
- **`max_semantic_scan`** — 语义检索时最多扫描的条目数（LRU 近似），在性能和精度间权衡

## Testing

- 测试框架: **pytest** + **pytest-asyncio** (asyncio_mode = "auto")
- 使用 `unittest.mock.MagicMock` 模拟 Redis 客户端
- 测试文件:
  - `tests/test_schema.py` — Schema 定义、验证、YAML 序列化
  - `tests/test_cache.py` — SemanticCache + EmbeddingsCache
  - `tests/test_memory.py` — SemanticMessageHistory
  - `tests/test_query.py` — VectorQuery / TextQuery / FilterQuery / HybridQuery / QueryBuilder
  - `tests/test_routing.py` — SemanticRouter
  - `tests/test_types.py` — CacheStats, Message, compute_hash
  - `tests/test_service.py` — 服务层测试
- 公共 fixtures 在 `tests/conftest.py`（mock_redis, sample_schema_dict, sample_vector, sample_entries）
- 运行: `pytest` 或 `pytest --cov=vector_platform --cov-report=term-missing`

## Coding Conventions

- **Python 3.10+** — 使用 `from __future__ import annotations` 和 `|` 联合类型
- **字符编码**: 所有中文注释/文档用 UTF-8
- **命名**:
  - 文件: `snake_case`
  - 类: `PascalCase`
  - 函数/变量: `snake_case`
  - 私有成员: `_prefix`
- **Docstring**: 所有公开类和函数需要中英文双语 docstring
- **类型注解**: 所有公开方法需要类型注解
- **Pydantic v2** — 使用 `field_validator` / `model_validator`（非 v1 的 `validator` / `root_validator`）
- **格式化**: Ruff (line-length=100, select=E/F/I/N/W/UP/B/C4)
- **类型检查**: mypy (strict=false, warn_return_any=true)
- **日志**: 使用 `logging.getLogger(__name__)`，日志级别: info 用于关键操作, warning 用于非致命异常, error 用于失败

## Current State

SDK 已实现核心功能（v0.2.0），含完整测试套件。各模块状态：

- ✅ 向量索引管理 (IndexSchema + SearchIndex) — 完成
- ✅ 向量/全文/过滤/混合查询 — 完成
- ✅ 语义缓存 (二级命中) — 完成
- ✅ 嵌入缓存 — 完成
- ✅ 对话记忆 (时间+语义检索) — 完成
- ✅ 语义路由 — 完成
- ❌ FastAPI HTTP 服务层 — 未实现（pyproject.toml 已声明可选依赖）
- ❌ OpenTelemetry 集成 — 未实现（pyproject.toml 已声明可选依赖）
- ❌ OpenAI 集成适配器 — 未实现（pyproject.toml 已声明可选依赖）

## Common Commands

```bash
# 开发安装
pip install -e ".[dev]"

# 运行测试
pytest
pytest --cov=vector_platform --cov-report=term-missing

# 代码检查
ruff check src/
mypy src/

# 运行示例（需要本地 Redis）
python examples/quickstart.py
```
