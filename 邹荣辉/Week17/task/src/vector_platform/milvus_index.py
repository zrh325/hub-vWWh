"""Milvus-like Schema & Search — 向量索引管理与混合检索.

统一的索引 Schema 定义、索引生命周期管理和多模式数据检索：
  - 向量检索 (KNN) — RediSearch FT.SEARCH KNN
  - 全文检索 — RediSearch BM25/TFIDF
  - 元数据过滤 — Tag/Numeric/Geo 过滤
  - 混合查询 — 向量 + 全文 + 过滤，支持 RRF / 线性加权融合
  - 流式 QueryBuilder API

Usage:
    from redis import Redis
    from vector_platform.milvus_index import (
        IndexSchema, SearchIndex, VectorQuery, HybridQuery, QueryBuilder,
    )

    redis_client = Redis(host="localhost", port=6379, decode_responses=False)

    # 1. 定义 Schema
    schema = IndexSchema.from_yaml("vector_index.yaml")

    # 2. 创建索引
    index = SearchIndex(schema, redis_client)
    index.create(overwrite=True)

    # 3. 灌入数据
    index.load([{"id": "1", "embedding": vec, "content": "..."}])

    # 4. 向量检索
    vq = VectorQuery(schema, redis_client)
    results = vq.search(vector=query_vec, top_k=10)

    # 5. 混合查询
    hq = HybridQuery(schema, redis_client)
    results = hq.search(vector=query_vec, text_query="关键词", top_k=20)
"""

from __future__ import annotations

import json
import logging
import time
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from redis import Redis

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 枚举类型
# ═══════════════════════════════════════════════════════════════════


class DistanceMetric(str, Enum):
    """向量相似度度量方式."""
    L2 = "L2"
    IP = "IP"
    COSINE = "COSINE"

    @classmethod
    def _missing_(cls, value: object) -> "DistanceMetric":
        if isinstance(value, str):
            for member in cls:
                if member.value.lower() == value.lower():
                    return member
        raise ValueError(f"'{value}' is not a valid {cls.__name__}")


class IndexAlgorithm(str, Enum):
    """向量索引算法."""
    FLAT = "FLAT"
    HNSW = "HNSW"

    @classmethod
    def _missing_(cls, value: object) -> "IndexAlgorithm":
        if isinstance(value, str):
            for member in cls:
                if member.value.lower() == value.lower():
                    return member
        raise ValueError(f"'{value}' is not a valid {cls.__name__}")


class StorageType(str, Enum):
    """数据存储格式."""
    HASH = "hash"
    JSON = "json"


class DataType(str, Enum):
    """向量值数据类型."""
    FLOAT32 = "float32"
    FLOAT64 = "float64"

    @classmethod
    def _missing_(cls, value: object) -> "DataType":
        if isinstance(value, str):
            for member in cls:
                if member.value.lower() == value.lower():
                    return member
        raise ValueError(f"'{value}' is not a valid {cls.__name__}")


class FusionMethod(str, Enum):
    """混合查询结果融合方式."""
    LINEAR = "linear"
    RRF = "rrf"


# ═══════════════════════════════════════════════════════════════════
# 数据类
# ═══════════════════════════════════════════════════════════════════


@dataclass
class VectorEntry:
    """一条向量记录."""
    id: str
    vector: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    fields: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


@dataclass
class QueryResult:
    """查询结果."""
    entries: list[VectorEntry] = field(default_factory=list)
    total: int = 0
    query_time_ms: float = 0.0
    fusion_method: Optional[str] = None

    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self) -> bool:
        return len(self.entries) > 0


# ═══════════════════════════════════════════════════════════════════
# 字段定义 (Pydantic 模型)
# ═══════════════════════════════════════════════════════════════════


class VectorField(BaseModel):
    """向量字段定义."""
    name: str
    dims: int = Field(gt=0, description="向量维度")
    algorithm: IndexAlgorithm = IndexAlgorithm.HNSW
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    datatype: DataType = DataType.FLOAT32
    m: int = Field(default=16, ge=4, le=512, description="HNSW M参数")
    ef_construction: int = Field(default=200, ge=4, le=4096, description="HNSW构建搜索深度")
    ef_runtime: int = Field(default=10, ge=4, le=4096, description="HNSW查询搜索深度")
    count: int = Field(default=6, ge=1, description="初始向量数量")


class TagField(BaseModel):
    """标签字段定义."""
    name: str
    separator: str = Field(default=",", description="多值分隔符")


class TextField(BaseModel):
    """文本字段定义."""
    name: str
    weight: float = Field(default=1.0, ge=0.0, description="搜索权重")
    nostem: bool = Field(default=False, description="是否禁用词干分析")


class NumericField(BaseModel):
    """数值字段定义."""
    name: str
    sortable: bool = Field(default=False, description="是否支持排序")


class GeoField(BaseModel):
    """地理位置字段定义."""
    name: str


# ═══════════════════════════════════════════════════════════════════
# IndexSchema
# ═══════════════════════════════════════════════════════════════════


class IndexSchema(BaseModel):
    """索引Schema定义 — 整个索引的元数据描述.

    Usage:
        # YAML 方式
        schema = IndexSchema.from_yaml("vector_index.yaml")

        # 编程方式
        schema = IndexSchema(
            index={"name": "docs", "prefix": "doc", "storage_type": "json"},
            fields=[
                {"name": "content", "type": "text"},
                {"name": "embedding", "type": "vector", "dims": 1536, "distance_metric": "cosine"},
                {"name": "category", "type": "tag"},
            ],
        )
    """

    index: dict[str, Any] = Field(description="索引元信息 (name, prefix, storage_type)")
    fields: list[dict[str, Any]] = Field(default_factory=list, description="字段定义列表")

    @field_validator("index")
    @classmethod
    def validate_index_meta(cls, v: dict[str, Any]) -> dict[str, Any]:
        required = ["name", "prefix"]
        for key in required:
            if key not in v:
                raise ValueError(f"index must contain '{key}' field")
        v.setdefault("storage_type", "hash")
        if v["storage_type"] not in {"hash", "json"}:
            raise ValueError(f"storage_type must be 'hash' or 'json', got '{v['storage_type']}'")
        return v

    @model_validator(mode="after")
    def validate_fields(self) -> "IndexSchema":
        """验证字段定义的有效性."""
        for f in self.fields:
            if "name" not in f or "type" not in f:
                raise ValueError(f"Each field must have 'name' and 'type': {f}")
            ftype = f["type"]
            if ftype == "vector":
                if "dims" not in f:
                    raise ValueError(f"Vector field '{f['name']}' requires 'dims'")
                VectorField(**f)
            elif ftype == "tag":
                TagField(**f)
            elif ftype == "text":
                TextField(**f)
            elif ftype == "numeric":
                NumericField(**f)
            elif ftype == "geo":
                GeoField(**f)
            else:
                raise ValueError(
                    f"Unknown field type '{ftype}' for field '{f['name']}'. "
                    f"Supported: vector, tag, text, numeric, geo"
                )
        return self

    # ─── 便捷属性 ──────────────────────────────────────────

    @property
    def name(self) -> str:
        return self.index["name"]

    @property
    def prefix(self) -> str:
        return self.index["prefix"]

    @property
    def storage_type(self) -> StorageType:
        return StorageType(self.index.get("storage_type", "hash"))

    @property
    def vector_fields(self) -> list[dict[str, Any]]:
        return [f for f in self.fields if f["type"] == "vector"]

    @property
    def text_fields(self) -> list[dict[str, Any]]:
        return [f for f in self.fields if f["type"] == "text"]

    @property
    def tag_fields(self) -> list[dict[str, Any]]:
        return [f for f in self.fields if f["type"] == "tag"]

    @property
    def numeric_fields(self) -> list[dict[str, Any]]:
        return [f for f in self.fields if f["type"] == "numeric"]

    @property
    def field_names(self) -> list[str]:
        return [f["name"] for f in self.fields]

    # ─── 序列化 ────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "IndexSchema":
        """从YAML文件加载Schema定义."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Schema YAML file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict) or "index" not in data:
            raise ValueError(
                f"Invalid schema YAML: must contain top-level 'index' key. Got: {type(data)}"
            )
        return cls(**data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """将Schema导出为YAML文件."""
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(), f, default_flow_style=False, allow_unicode=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IndexSchema":
        """从字典创建Schema."""
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """导出为字典."""
        return self.model_dump()

    # ─── RediSearch 命令生成 ────────────────────────────────

    def to_ft_create(self) -> str:
        """生成 RediSearch FT.CREATE 命令."""
        prefix = self.prefix
        index_name = self.name
        parts = [f"FT.CREATE {index_name} ON {self.storage_type.value.upper()} PREFIX 1 {prefix}: SCHEMA"]
        for f in self.fields:
            parts.append(self._build_field_cmd(f))
        return " ".join(parts)

    def _build_field_cmd(self, field: dict[str, Any]) -> str:
        """为单个字段构建 RediSearch 命令片段."""
        name = field["name"]
        ftype = field.get("type", "text")
        if ftype == "tag":
            sep = field.get("separator", ",")
            return f"{name} TAG SEPARATOR {sep}"
        elif ftype == "text":
            weight = field.get("weight", 1.0)
            opts = f" WEIGHT {weight}"
            if field.get("nostem"):
                opts += " NOSTEM"
            return f"{name} TEXT{opts}"
        elif ftype == "numeric":
            sortable = " SORTABLE" if field.get("sortable") else ""
            return f"{name} NUMERIC{sortable}"
        elif ftype == "geo":
            return f"{name} GEO"
        elif ftype == "vector":
            return self._build_vector_cmd(name, field)
        return f"{name} TEXT"

    def _build_vector_cmd(self, name: str, field: dict[str, Any]) -> str:
        """构建向量字段的 RediSearch 子命令."""
        vf = VectorField(**field)
        return (
            f"{name} VECTOR {vf.algorithm.value} "
            f"{field.get('count', 6)} "
            f"TYPE {vf.datatype.value} "
            f"DIM {vf.dims} "
            f"DISTANCE_METRIC {vf.distance_metric.value} "
            f"M {vf.m} "
            f"EF_CONSTRUCTION {vf.ef_construction} "
            f"EF_RUNTIME {vf.ef_runtime}"
        )

    # ─── 辅助方法 ──────────────────────────────────────────

    def key_for(self, entry_id: str) -> str:
        """生成指定ID的Redis Key."""
        return f"{self.prefix}:{entry_id}"

    def get_vector_field(self) -> Optional[dict[str, Any]]:
        """获取第一个向量字段."""
        vec_fields = self.vector_fields
        return vec_fields[0] if vec_fields else None

    def __repr__(self) -> str:
        return f"<IndexSchema name='{self.name}' prefix='{self.prefix}' fields={len(self.fields)}>"


# ═══════════════════════════════════════════════════════════════════
# SearchIndex — 索引生命周期管理
# ═══════════════════════════════════════════════════════════════════


class SearchIndex:
    """Redis向量搜索索引 — 管理索引的完整生命周期.

    Usage:
        schema = IndexSchema.from_yaml("vector_index.yaml")
        index = SearchIndex(schema, redis_client)
        index.create(overwrite=True)
        index.load([{"id": "1", "embedding": vec, "content": "..."}])
    """

    def __init__(self, schema: IndexSchema, client: Redis):
        self._schema = schema
        self._client = client

    @property
    def schema(self) -> IndexSchema:
        return self._schema

    @property
    def name(self) -> str:
        return self._schema.name

    @property
    def prefix(self) -> str:
        return self._schema.prefix

    @property
    def client(self) -> Redis:
        return self._client

    # ─── 索引生命周期 ─────────────────────────────────────

    def create(self, overwrite: bool = False) -> bool:
        """创建索引."""
        if self.exists():
            if overwrite:
                logger.info("Dropping existing index '%s' before recreation", self.name)
                self.drop()
            else:
                logger.warning("Index '%s' already exists, skipping creation", self.name)
                return False
        try:
            cmd = self._schema.to_ft_create()
            logger.info("Creating index: %s", cmd)
            self._client.execute_command(*cmd.split())
            logger.info("Index '%s' created successfully", self.name)
            return True
        except Exception as e:
            logger.error("Failed to create index '%s': %s", self.name, e)
            raise

    def drop(self) -> bool:
        """删除索引."""
        try:
            self._client.execute_command(f"FT.DROPINDEX {self.name}")
            logger.info("Index '%s' dropped", self.name)
            return True
        except Exception as e:
            logger.warning("Failed to drop index '%s': %s", self.name, e)
            return False

    def exists(self) -> bool:
        """检查索引是否存在."""
        try:
            self._client.execute_command(f"FT.INFO {self.name}")
            return True
        except Exception:
            return False

    def info(self) -> dict[str, Any]:
        """获取索引详细信息."""
        try:
            raw = self._client.execute_command(f"FT.INFO {self.name}")
            return self._parse_info(raw)
        except Exception as e:
            logger.error("Failed to get info for index '%s': %s", self.name, e)
            raise

    @staticmethod
    def _parse_info(raw: list) -> dict[str, Any]:
        info: dict[str, Any] = {}
        it = iter(raw)
        for key in it:
            key_str = key.decode() if isinstance(key, bytes) else key
            value = next(it)
            info[key_str] = value
        return info

    # ─── 数据操作 ─────────────────────────────────────────

    def load(
        self,
        entries: list[dict[str, Any]],
        vector_field: Optional[str] = None,
        id_field: str = "id",
        chunk_size: int = 1000,
    ) -> int:
        """批量灌入数据."""
        if not entries:
            return 0

        if vector_field is None:
            vec_field = self._schema.get_vector_field()
            if vec_field is None:
                raise ValueError("No vector field defined in schema and vector_field not specified")
            vector_field = vec_field["name"]

        storage_type = self._schema.storage_type.value
        loaded = 0

        for i in range(0, len(entries), chunk_size):
            chunk = entries[i : i + chunk_size]
            pipe = self._client.pipeline(transaction=False)
            for entry in chunk:
                key = self._schema.key_for(entry[id_field])
                data = self._serialize_entry(entry, vector_field, storage_type)
                pipe.hset(key, mapping=data)
            pipe.execute()
            loaded += len(chunk)

        logger.info("Loaded %d entries into index '%s'", loaded, self.name)
        return loaded

    def _serialize_entry(
        self, entry: dict[str, Any], vector_field: str, storage_type: str
    ) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for f in self._schema.fields:
            fname = f["name"]
            if fname not in entry:
                continue
            value = entry[fname]
            if f["type"] == "vector":
                data[fname] = self._serialize_vector(value)
            elif f["type"] == "tag" and isinstance(value, list):
                data[fname] = f.get("separator", ",").join(str(v) for v in value)
            elif isinstance(value, dict) and storage_type == "json":
                data[fname] = json.dumps(value)
            else:
                data[fname] = str(value)
        return data

    @staticmethod
    def _serialize_vector(vec: Any) -> bytes:
        if isinstance(vec, np.ndarray):
            return vec.astype(np.float32).tobytes()
        elif isinstance(vec, list):
            return np.array(vec, dtype=np.float32).tobytes()
        elif isinstance(vec, bytes):
            return vec
        else:
            raise TypeError(f"Unsupported vector type: {type(vec)}")

    # ─── 条目操作 ─────────────────────────────────────────

    def insert(self, entry_id: str, fields: dict[str, Any], vector_field: Optional[str] = None) -> bool:
        """插入/更新单条记录."""
        key = self._schema.key_for(entry_id)
        if vector_field is None:
            vec_field = self._schema.get_vector_field()
            vector_field = vec_field["name"] if vec_field else "embedding"
        data = self._serialize_entry(
            {**fields, "id": entry_id}, vector_field, self._schema.storage_type.value
        )
        self._client.hset(key, mapping=data)
        return True

    def delete(self, entry_id: str) -> bool:
        """删除单条记录."""
        key = self._schema.key_for(entry_id)
        return bool(self._client.delete(key))

    def get(self, entry_id: str) -> Optional[dict[str, Any]]:
        """获取单条记录."""
        key = self._schema.key_for(entry_id)
        data = self._client.hgetall(key)
        return data if data else None

    def count(self) -> int:
        """估算索引中的文档数量."""
        try:
            info = self.info()
            return int(info.get("num_docs", 0))
        except Exception:
            return 0

    def __repr__(self) -> str:
        return f"<SearchIndex name='{self.name}' prefix='{self.prefix}'>"


# ═══════════════════════════════════════════════════════════════════
# 内部辅助：提取 ID
# ═══════════════════════════════════════════════════════════════════


def _extract_id(key: str, prefix: str) -> str:
    """从 Redis key 提取 ID."""
    needle = prefix + ":"
    return key[len(needle):] if key.startswith(needle) else key


def _to_vector_blob(vec: Union[list[float], np.ndarray]) -> bytes:
    """将向量转换为 Redis 向量 blob 格式."""
    return np.asarray(vec, dtype=np.float32).tobytes()


# ═══════════════════════════════════════════════════════════════════
# FilterQuery — 元数据过滤查询
# ═══════════════════════════════════════════════════════════════════


class FilterQuery:
    """元数据过滤查询 — 基于 RediSearch 的标签/数值/文本/地理位置过滤.

    Usage:
        query = FilterQuery(schema, redis_client)
        results = query.filter(filter_expr="@category:{news} @year:[2020 +inf]")
    """

    def __init__(self, schema: IndexSchema, client: Redis):
        self._schema = schema
        self._client = client

    def filter(
        self,
        filter_expr: str,
        return_fields: Optional[list[str]] = None,
        sort_by: Optional[str] = None,
        sort_asc: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> QueryResult:
        """执行元数据过滤查询."""
        cmd = [
            "FT.SEARCH", self._schema.name,
            filter_expr,
            "LIMIT", str(offset), str(limit),
        ]
        if sort_by:
            cmd.extend(["SORTBY", sort_by, "ASC" if sort_asc else "DESC"])
        if return_fields:
            cmd.extend(["RETURN", str(len(return_fields)), *return_fields])

        start_time = time.perf_counter()
        try:
            raw_result = self._client.execute_command(*cmd)
        except Exception as e:
            logger.error("FilterQuery failed: %s", e)
            raise

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        entries = self._parse_result(raw_result)
        return QueryResult(entries=entries, total=len(entries), query_time_ms=elapsed_ms)

    def by_tag(
        self, field: str, values: Union[str, list[str]],
        return_fields: Optional[list[str]] = None, limit: int = 100,
    ) -> list[VectorEntry]:
        """按标签字段过滤."""
        if isinstance(values, str):
            values = [values]
        expr = "|".join(f"@{field}:{{{v}}}" for v in values)
        return self.filter(filter_expr=expr, return_fields=return_fields, limit=limit).entries

    def by_range(
        self, field: str,
        min_val: Optional[float] = None, max_val: Optional[float] = None,
        inclusive_min: bool = True, inclusive_max: bool = True,
        return_fields: Optional[list[str]] = None, limit: int = 100,
    ) -> list[VectorEntry]:
        """按数值字段范围过滤."""
        min_bracket = "[" if inclusive_min else "("
        max_bracket = "]" if inclusive_max else ")"
        min_str = str(min_val) if min_val is not None else "-inf"
        max_str = str(max_val) if max_val is not None else "+inf"
        expr = f"@{field}:{min_bracket}{min_str} {max_str}{max_bracket}"
        return self.filter(filter_expr=expr, return_fields=return_fields, limit=limit).entries

    def by_geo(
        self, field: str,
        longitude: float, latitude: float, radius: float, unit: str = "km",
        return_fields: Optional[list[str]] = None, limit: int = 100,
    ) -> list[VectorEntry]:
        """按地理位置半径过滤."""
        expr = f"@{field}:[{longitude} {latitude} {radius} {unit}]"
        return self.filter(filter_expr=expr, return_fields=return_fields, limit=limit).entries

    def count(self, filter_expr: str = "*") -> int:
        """统计符合条件的文档数."""
        try:
            result = self._client.execute_command(
                "FT.SEARCH", self._schema.name, filter_expr,
                "LIMIT", "0", "0",
            )
            return int(result[0]) if result else 0
        except Exception:
            return 0

    def _parse_result(self, raw_result: list) -> list[VectorEntry]:
        entries = []
        if not raw_result or len(raw_result) < 2:
            return entries
        i = 1
        while i < len(raw_result):
            key = raw_result[i]
            if isinstance(key, bytes):
                key = key.decode()
            i += 1
            fields_raw = raw_result[i] if i < len(raw_result) else []
            i += 1
            fields: dict[str, Any] = {}
            j = 0
            while j < len(fields_raw):
                fname = fields_raw[j]
                if isinstance(fname, bytes):
                    fname = fname.decode()
                fval = fields_raw[j + 1] if j + 1 < len(fields_raw) else ""
                if isinstance(fval, bytes):
                    fval = fval.decode()
                fields[fname] = fval
                j += 2
            entries.append(VectorEntry(id=_extract_id(key, self._schema.prefix), fields=fields))
        return entries

    def __repr__(self) -> str:
        return f"<FilterQuery index='{self._schema.name}'>"


# ═══════════════════════════════════════════════════════════════════
# TextQuery — 全文搜索
# ═══════════════════════════════════════════════════════════════════


class TextQuery:
    """全文搜索查询 — 基于 RediSearch BM25/TFIDF.

    Usage:
        query = TextQuery(schema, redis_client)
        results = query.search("机器学习", fields=["title", "content"])
    """

    def __init__(self, schema: IndexSchema, client: Redis):
        self._schema = schema
        self._client = client

    def search(
        self,
        query: str,
        fields: Optional[list[str]] = None,
        filter_expr: Optional[str] = None,
        language: Optional[str] = None,
        scorer: str = "BM25",
        expander: Optional[str] = None,
        highlight: bool = False,
        summarize: bool = False,
        return_fields: Optional[list[str]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> QueryResult:
        """执行全文搜索.

        Args:
            query: 搜索查询文本
            fields: 限定搜索的字段列表
            language: 文本语言（中文用 "chinese"）
            scorer: 评分函数
            limit: 返回数量上限
            offset: 偏移量
        """
        cmd = ["FT.SEARCH", self._schema.name]

        if fields:
            escaped = f'"{query}"' if " " in query else query
            field_query = "|".join(f"@{f}:{escaped}" for f in fields)
            cmd.append(f"({field_query})")
        else:
            cmd.append(query)

        if language:
            cmd.extend(["LANGUAGE", language])
        if scorer != "BM25":
            cmd.extend(["SCORER", scorer])
        if expander:
            cmd.extend(["EXPANDER", expander])
        if highlight:
            cmd.append("HIGHLIGHT")
        if summarize:
            cmd.append("SUMMARIZE")

        cmd.extend(["LIMIT", str(offset), str(limit)])
        if return_fields:
            cmd.extend(["RETURN", str(len(return_fields)), *return_fields])

        start_time = time.perf_counter()
        try:
            raw_result = self._client.execute_command(*cmd)
        except Exception as e:
            logger.error("TextQuery failed: %s", e)
            raise

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        entries = self._parse_result(raw_result)
        return QueryResult(entries=entries, total=len(entries), query_time_ms=elapsed_ms)

    def search_simple(
        self, query: str, fields: Optional[list[str]] = None, limit: int = 10,
    ) -> list[VectorEntry]:
        """简单搜索 — 直接返回条目列表."""
        return self.search(query=query, fields=fields, limit=limit).entries

    def _parse_result(self, raw_result: list) -> list[VectorEntry]:
        entries = []
        if not raw_result or len(raw_result) < 2:
            return entries
        i = 1
        while i < len(raw_result):
            key = raw_result[i]
            if isinstance(key, bytes):
                key = key.decode()
            i += 1
            fields_raw = raw_result[i] if i < len(raw_result) else []
            i += 1
            fields: dict[str, Any] = {}
            j = 0
            while j < len(fields_raw):
                fname = fields_raw[j]
                if isinstance(fname, bytes):
                    fname = fname.decode()
                fval = fields_raw[j + 1] if j + 1 < len(fields_raw) else ""
                if isinstance(fval, bytes):
                    fval = fval.decode()
                fields[fname] = fval
                j += 2
            entries.append(VectorEntry(id=_extract_id(key, self._schema.prefix), fields=fields))
        return entries

    def __repr__(self) -> str:
        return f"<TextQuery index='{self._schema.name}'>"


# ═══════════════════════════════════════════════════════════════════
# VectorQuery — KNN向量检索
# ═══════════════════════════════════════════════════════════════════


class VectorQuery:
    """向量相似度查询 — 基于 RediSearch KNN.

    Usage:
        query = VectorQuery(schema, redis_client)
        results = query.search(vector=[0.1, 0.2, ...], top_k=10, filter_expr="@category:{news}")
    """

    def __init__(self, schema: IndexSchema, client: Redis):
        self._schema = schema
        self._client = client

    def search(
        self,
        vector: Union[list[float], np.ndarray],
        vector_field: Optional[str] = None,
        top_k: int = 10,
        filter_expr: Optional[str] = None,
        return_fields: Optional[list[str]] = None,
        ef_runtime: Optional[int] = None,
        dialect: int = 2,
    ) -> QueryResult:
        """执行KNN向量搜索.

        Args:
            vector: 查询向量
            vector_field: 向量字段名（不指定则取Schema第一个向量字段）
            top_k: 返回结果数
            filter_expr: 可选过滤表达式，如 "@category:{news}"
            return_fields: 要返回的字段列表
            ef_runtime: HNSW查询搜索深度（覆盖Schema默认值）
            dialect: RediSearch方言版本
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        if vector_field is None:
            vec_field = self._schema.get_vector_field()
            if vec_field is None:
                raise ValueError("No vector field defined in schema")
            vector_field = vec_field["name"]

        vec_blob = _to_vector_blob(vector)
        ef = ef_runtime or self._get_ef_runtime()

        filter_prefix = filter_expr if filter_expr else "*"
        knn_query = (
            f"{filter_prefix}=>[KNN {top_k} @{vector_field} $BLOB "
            f"EF_RUNTIME {ef}]"
        )

        cmd = [
            "FT.SEARCH", self._schema.name,
            knn_query,
            "PARAMS", "2", "BLOB", vec_blob,
            "DIALECT", str(dialect),
        ]
        if return_fields:
            cmd.extend(["RETURN", str(len(return_fields)), *return_fields])
        else:
            all_fields = self._schema.field_names
            cmd.extend(["RETURN", str(len(all_fields)), *all_fields])

        start_time = time.perf_counter()
        try:
            raw_result = self._client.execute_command(*cmd)
        except Exception as e:
            logger.error("VectorQuery failed: %s", e)
            raise

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        entries = self._parse_search_result(raw_result, vector_field)
        return QueryResult(entries=entries, total=len(entries), query_time_ms=elapsed_ms)

    def search_simple(
        self,
        vector: Union[list[float], np.ndarray],
        vector_field: Optional[str] = None,
        top_k: int = 10,
        filter_expr: Optional[str] = None,
        return_fields: Optional[list[str]] = None,
    ) -> list[VectorEntry]:
        """简化搜索 — 直接返回条目列表."""
        return self.search(
            vector=vector, vector_field=vector_field, top_k=top_k,
            filter_expr=filter_expr, return_fields=return_fields,
        ).entries

    def _get_ef_runtime(self) -> int:
        vec_field = self._schema.get_vector_field()
        return vec_field.get("ef_runtime", 10) if vec_field else 10

    def _parse_search_result(self, raw_result: list, vector_field: str) -> list[VectorEntry]:
        entries = []
        if not raw_result or len(raw_result) < 2:
            return entries

        i = 1
        while i < len(raw_result):
            key = raw_result[i]
            if isinstance(key, bytes):
                key = key.decode()
            i += 1

            fields_raw = raw_result[i] if i < len(raw_result) else []
            i += 1

            fields: dict[str, Any] = {}
            score = 0.0
            j = 0
            while j < len(fields_raw):
                fname = fields_raw[j]
                if isinstance(fname, bytes):
                    fname = fname.decode()
                fval = fields_raw[j + 1] if j + 1 < len(fields_raw) else ""

                if fname == "__" + vector_field + "_score":
                    score = float(fval)
                elif fname == vector_field and isinstance(fval, bytes):
                    fields[fname] = np.frombuffer(fval, dtype=np.float32)
                else:
                    if isinstance(fval, bytes):
                        fval = fval.decode()
                    fields[fname] = fval
                j += 2

            entries.append(VectorEntry(
                id=_extract_id(key, self._schema.prefix),
                vector=fields.get(vector_field, np.array([])),
                fields=fields,
                score=float(score),
            ))
        return entries

    def __repr__(self) -> str:
        return f"<VectorQuery index='{self._schema.name}'>"


# ═══════════════════════════════════════════════════════════════════
# 融合函数
# ═══════════════════════════════════════════════════════════════════


def reciprocal_rank_fusion(
    result_sets: list[list[VectorEntry]],
    k: int = 60,
) -> list[VectorEntry]:
    """Reciprocal Rank Fusion (RRF) — 对多组结果进行融合排序.

    Args:
        result_sets: 多组查询结果
        k: RRF平滑参数（默认60）

    Returns:
        融合排序后的结果列表
    """
    scores: dict[str, tuple[float, VectorEntry]] = {}
    for results in result_sets:
        for rank, entry in enumerate(results, start=1):
            rrf_score = 1.0 / (k + rank)
            if entry.id in scores:
                prev_score, prev_entry = scores[entry.id]
                scores[entry.id] = (prev_score + rrf_score, prev_entry)
            else:
                scores[entry.id] = (rrf_score, entry)
    sorted_entries = sorted(scores.values(), key=lambda x: x[0], reverse=True)
    return [entry for _score, entry in sorted_entries]


def linear_weighted_fusion(
    result_sets: list[list[VectorEntry]],
    weights: list[float],
) -> list[VectorEntry]:
    """线性加权融合 — 归一化分数后加权求和.

    Args:
        result_sets: 多组查询结果
        weights: 每组对应的权重
    """
    if len(weights) != len(result_sets):
        raise ValueError(
            f"Number of weights ({len(weights)}) must match result_sets ({len(result_sets)})"
        )
    scores: dict[str, tuple[float, VectorEntry]] = {}
    for results, weight in zip(result_sets, weights):
        if not results:
            continue
        actual_scores = [e.score if e.score is not None else 0.0 for e in results]
        max_score = max(actual_scores) if actual_scores else 1.0
        if max_score <= 0.0:
            max_score = 1.0
        for entry in results:
            raw_score = entry.score if entry.score is not None else 0.0
            normalized = raw_score / max_score
            weighted = normalized * weight
            if entry.id in scores:
                prev_score, prev_entry = scores[entry.id]
                scores[entry.id] = (prev_score + weighted, prev_entry)
            else:
                scores[entry.id] = (weighted, entry)
    sorted_entries = sorted(scores.values(), key=lambda x: x[0], reverse=True)
    return [entry for _score, entry in sorted_entries]


# ═══════════════════════════════════════════════════════════════════
# HybridQuery — 混合查询引擎
# ═══════════════════════════════════════════════════════════════════


class HybridQuery:
    """混合查询引擎 — 组合向量搜索 + 过滤 + 全文搜索.

    Usage:
        query = HybridQuery(schema, redis_client)

        # 向量 + 标签过滤
        results = query.search(vector=my_vector, filter_expr="@category:{news}", top_k=10)

        # 向量 + 全文 + RRF融合
        results = query.search(
            vector=my_vector, text_query="机器学习",
            top_k=20, fusion=FusionMethod.RRF,
        )
    """

    def __init__(self, schema: IndexSchema, client: Redis):
        self._schema = schema
        self._client = client
        self._vector_query = VectorQuery(schema, client)
        self._filter_query = FilterQuery(schema, client)
        self._text_query = TextQuery(schema, client)

    def search(
        self,
        *,
        vector: Optional[Union[list[float], np.ndarray]] = None,
        text_query: Optional[str] = None,
        filter_expr: Optional[str] = None,
        vector_field: Optional[str] = None,
        top_k: int = 10,
        return_fields: Optional[list[str]] = None,
        fusion: FusionMethod = FusionMethod.LINEAR,
        rrf_k: int = 60,
        vector_weight: float = 0.6,
        text_weight: float = 0.3,
        filter_weight: float = 0.1,
    ) -> QueryResult:
        """执行混合查询.

        Args:
            vector: 查询向量（可选）
            text_query: 全文搜索文本（可选）
            filter_expr: 元数据过滤表达式（可选）
            vector_field: 向量字段名
            top_k: 返回数量
            return_fields: 返回字段
            fusion: 融合方式 (LINEAR 或 RRF)
            rrf_k: RRF平滑参数
            vector_weight/text_weight/filter_weight: 线性融合权重
        """
        if vector is None and text_query is None and filter_expr is None:
            raise ValueError(
                "At least one of vector, text_query, or filter_expr is required"
            )

        start_time = time.perf_counter()
        result_sets: list[list[VectorEntry]] = []
        weights: list[float] = []

        # 向量搜索
        if vector is not None:
            vec_results = self._vector_query.search(
                vector=vector, vector_field=vector_field,
                top_k=top_k * 2, filter_expr=filter_expr,
                return_fields=return_fields,
            )
            result_sets.append(vec_results.entries)
            weights.append(vector_weight)

        # 全文搜索
        if text_query is not None:
            text_results = self._text_query.search(
                query=text_query, filter_expr=filter_expr,
                limit=top_k * 2, return_fields=return_fields,
            )
            result_sets.append(text_results.entries)
            weights.append(text_weight)

        # 纯过滤
        if filter_expr is not None and vector is None and text_query is None:
            filter_results = self._filter_query.filter(
                filter_expr=filter_expr,
                return_fields=return_fields,
                limit=top_k * 2,
            )
            result_sets.append(filter_results.entries)
            weights = [1.0]

        # 融合
        if len(result_sets) == 1:
            fused = result_sets[0]
        elif fusion == FusionMethod.RRF:
            fused = reciprocal_rank_fusion(result_sets, k=rrf_k)
        else:
            actual_weights = weights[: len(result_sets)]
            total = sum(actual_weights)
            if total > 0:
                actual_weights = [w / total for w in actual_weights]
            fused = linear_weighted_fusion(result_sets, actual_weights)

        fused = fused[:top_k]
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return QueryResult(
            entries=fused, total=len(fused),
            query_time_ms=elapsed_ms, fusion_method=fusion.value,
        )

    def search_simple(
        self,
        *,
        vector: Optional[Union[list[float], np.ndarray]] = None,
        text_query: Optional[str] = None,
        filter_expr: Optional[str] = None,
        top_k: int = 10,
    ) -> list[VectorEntry]:
        """简化混合查询 — 直接返回条目列表."""
        return self.search(
            vector=vector, text_query=text_query,
            filter_expr=filter_expr, top_k=top_k,
        ).entries

    def __repr__(self) -> str:
        return f"<HybridQuery index='{self._schema.name}'>"


# ═══════════════════════════════════════════════════════════════════
# QueryBuilder — 流式查询构建器
# ═══════════════════════════════════════════════════════════════════


class QueryBuilder:
    """流式查询构建器 — 链式API构建和执行查询.

    Usage:
        results = (
            QueryBuilder(schema, client)
            .vector([0.1, 0.2, ...])
            .filter("@category:{news}")
            .text("机器学习")
            .top_k(20)
            .fusion("rrf")
            .execute()
        )
    """

    def __init__(self, schema: IndexSchema, client: Redis):
        self._schema = schema
        self._client = client
        self._hybrid = HybridQuery(schema, client)
        self._vector: Optional[Union[list[float], np.ndarray]] = None
        self._text: Optional[str] = None
        self._filter_expr: Optional[str] = None
        self._vector_field: Optional[str] = None
        self._top_k: int = 10
        self._return_fields: Optional[list[str]] = None
        self._fusion: FusionMethod = FusionMethod.LINEAR
        self._rrf_k: int = 60
        self._weights: dict[str, float] = {"vector": 0.6, "text": 0.3, "filter": 0.1}

    # ─── 查询条件 ─────────────────────────────────────────

    def vector(self, vec: Union[list[float], np.ndarray], field: Optional[str] = None) -> "QueryBuilder":
        """设置查询向量."""
        self._vector = vec
        self._vector_field = field
        return self

    def text(self, query: str) -> "QueryBuilder":
        """设置全文搜索文本."""
        self._text = query
        return self

    def filter(self, expr: str) -> "QueryBuilder":
        """设置过滤表达式."""
        self._filter_expr = expr
        return self

    def tag(self, field: str, *values: str) -> "QueryBuilder":
        """按标签过滤（链式便利方法）."""
        if len(values) == 1:
            self._filter_expr = f"@{field}:{{{values[0]}}}"
        else:
            self._filter_expr = " | ".join(f"@{field}:{{{v}}}" for v in values)
        return self

    # ─── 查询参数 ─────────────────────────────────────────

    def top_k(self, k: int) -> "QueryBuilder":
        """设置返回结果数."""
        self._top_k = k
        return self

    def return_fields(self, *fields: str) -> "QueryBuilder":
        """设置返回字段."""
        self._return_fields = list(fields)
        return self

    def fusion(self, method: Union[str, FusionMethod], rrf_k: int = 60) -> "QueryBuilder":
        """设置融合方式."""
        self._fusion = FusionMethod(method) if isinstance(method, str) else method
        self._rrf_k = rrf_k
        return self

    def weights(
        self,
        vector: Optional[float] = None,
        text: Optional[float] = None,
        filter: Optional[float] = None,
    ) -> "QueryBuilder":
        """设置各通道权重."""
        if vector is not None:
            self._weights["vector"] = vector
        if text is not None:
            self._weights["text"] = text
        if filter is not None:
            self._weights["filter"] = filter
        return self

    # ─── 执行 ─────────────────────────────────────────────

    def execute(self) -> QueryResult:
        """执行查询并返回完整结果."""
        return self._hybrid.search(
            vector=self._vector, text_query=self._text,
            filter_expr=self._filter_expr, vector_field=self._vector_field,
            top_k=self._top_k, return_fields=self._return_fields,
            fusion=self._fusion, rrf_k=self._rrf_k,
            vector_weight=self._weights["vector"],
            text_weight=self._weights["text"],
            filter_weight=self._weights["filter"],
        )

    def execute_simple(self) -> list[VectorEntry]:
        """执行查询并直接返回条目列表."""
        return self.execute().entries

    def __repr__(self) -> str:
        parts = []
        if self._vector is not None:
            parts.append("vector")
        if self._text is not None:
            parts.append(f"text='{self._text[:30]}'")
        if self._filter_expr is not None:
            parts.append(f"filter='{self._filter_expr}'")
        return f"<QueryBuilder {' + '.join(parts) or 'empty'}>"
