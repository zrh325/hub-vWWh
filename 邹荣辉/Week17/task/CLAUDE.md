# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

构建一个公司内部统一的、生产就绪的**向量检索与智能缓存服务平台**。核心是利用公司已部署的高性能Redis集群，封装AI原生数据模式与操作，为上层业务提供Python SDK与配套服务。

参考项目: [RedisVL](https://github.com/redis/redis-vl-python)

## Tech Stack

- **Python** — SDK and service language
- **Redis** — vector storage, caching, session management
- **Faiss** — vector similarity search (alongside Redis)
- **LLM integrations** — semantic caching for model calls (对话、Embedding、意图识别等)

## Architecture (Planned)

Four core modules:

1. **统一向量数据管理** — 标准化索引定义、数据灌入、向量检索接口，支持多种向量化算法和相似度度量
2. **混合查询引擎** — 向量相似性搜索 + 业务元数据过滤 + 关键词全文搜索的组合查询
3. **智能缓存体系**:
   - **语义缓存** — 基于语义相似的LLM请求-结果缓存，降低模型调用成本与延迟
   - **嵌入缓存** — 缓存文本→向量转换结果，避免重复嵌入计算
4. **LLM应用支持组件** — 对话历史管理（按session_id分离存储）、语义路由等通用模式

## Key Design Decisions (from README)

- Cache storage is session-scoped: conversations are partitioned by `session_id`
- Both LLM call results AND historical conversations are cached
- Redis is the existing infrastructure — the SDK wraps it, doesn't replace it
- Reference architecture is RedisVL's design patterns (AI-native data schemas on Redis)

## Current State

This repository is at the specification stage. The README.md is the authoritative source for project requirements and design intent. No code has been written yet.
