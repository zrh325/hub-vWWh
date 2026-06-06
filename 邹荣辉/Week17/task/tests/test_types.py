"""types & milvus_index 类型单元测试。"""

import numpy as np
import pytest

from vector_platform.milvus_index import (
    DataType,
    DistanceMetric,
    FusionMethod,
    IndexAlgorithm,
    QueryResult,
    StorageType,
    VectorEntry,
)
from vector_platform.types import CacheStats, Message, compute_hash


class TestEnums:
    def test_distance_metric(self):
        assert DistanceMetric.L2.value == "L2"
        assert DistanceMetric.IP.value == "IP"
        assert DistanceMetric.COSINE.value == "COSINE"

    def test_index_algorithm(self):
        assert IndexAlgorithm.FLAT.value == "FLAT"
        assert IndexAlgorithm.HNSW.value == "HNSW"

    def test_storage_type(self):
        assert StorageType.HASH.value == "hash"
        assert StorageType.JSON.value == "json"

    def test_fusion_method(self):
        assert FusionMethod.LINEAR.value == "linear"
        assert FusionMethod.RRF.value == "rrf"


class TestVectorEntry:
    def test_create_entry(self):
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        entry = VectorEntry(id="doc_1", vector=vec, fields={"text": "hello"}, score=0.85)
        assert entry.id == "doc_1"
        assert entry.score == 0.85
        assert entry.fields["text"] == "hello"
        assert np.array_equal(entry.vector, vec)

    def test_default_values(self):
        entry = VectorEntry(id="doc_2", vector=np.array([0.0]))
        assert entry.fields == {}
        assert entry.score == 0.0


class TestQueryResult:
    def test_empty_result(self):
        result = QueryResult()
        assert len(result) == 0
        assert not result

    def test_with_entries(self):
        entries = [
            VectorEntry(id="a", vector=np.array([1.0]), score=0.9),
            VectorEntry(id="b", vector=np.array([2.0]), score=0.8),
        ]
        result = QueryResult(entries=entries, total=2, query_time_ms=5.0)
        assert len(result) == 2
        assert result
        assert result.total == 2
        assert result.query_time_ms == 5.0


class TestCacheStats:
    def test_empty_stats(self):
        stats = CacheStats()
        assert stats.total_requests == 0
        assert stats.hit_rate == 0.0
        assert stats.avg_latency_ms == 0.0

    def test_all_hits(self):
        stats = CacheStats(hits_exact=5, hits_semantic=3, misses=0, total_latency_ms=20.0)
        assert stats.total_requests == 8
        assert stats.hit_rate == 1.0
        assert stats.avg_latency_ms == 2.5

    def test_mixed(self):
        stats = CacheStats(hits_exact=3, hits_semantic=2, misses=5, total_latency_ms=100.0)
        assert stats.total_requests == 10
        assert stats.hit_rate == 0.5
        assert stats.avg_latency_ms == 10.0

    def test_zero_requests(self):
        stats = CacheStats(hits_exact=0, hits_semantic=0, misses=0)
        assert stats.hit_rate == 0.0
        assert stats.avg_latency_ms == 0.0


class TestMessage:
    def test_create_message(self):
        msg = Message(role="user", content="你好", timestamp=1717500000.0, token_count=2)
        assert msg.role == "user"
        assert msg.content == "你好"
        assert msg.token_count == 2

    def test_to_dict(self):
        msg = Message(role="assistant", content="你好！有什么可以帮助的？", timestamp=1717500001.0, token_count=8)
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert d["content"] == "你好！有什么可以帮助的？"
        assert d["token_count"] == 8

    def test_from_dict(self):
        data = {"role": "system", "content": "你是一个助手", "timestamp": 1717500000.0, "token_count": 5}
        msg = Message.from_dict(data)
        assert msg.role == "system"
        assert msg.content == "你是一个助手"
        assert msg.token_count == 5

    def test_from_dict_defaults(self):
        data = {"role": "user", "content": "hello"}
        msg = Message.from_dict(data)
        assert msg.timestamp == 0.0
        assert msg.token_count == 0


class TestComputeHash:
    def test_same_text_same_hash(self):
        h1 = compute_hash("hello world")
        h2 = compute_hash("hello world")
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_different_text_different_hash(self):
        h1 = compute_hash("hello world")
        h2 = compute_hash("hello World")
        assert h1 != h2

    def test_hash_length(self):
        h = compute_hash("test")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)
