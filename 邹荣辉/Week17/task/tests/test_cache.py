"""Cache 模块单元测试 — EmbeddingsCache + SemanticCache。"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from vector_platform.embedding_cache import EmbeddingsCache
from vector_platform.semantic_cache import SemanticCache


class TestEmbeddingsCache:
    @pytest.fixture
    def cache(self):
        client = MagicMock()
        return EmbeddingsCache(client, ttl=3600, prefix="test:emb:")

    def test_set_and_get(self, cache):
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        cache._client.hget.return_value = vec.astype(np.float32).tobytes()

        result = cache.get("hello world")
        assert result is not None
        assert np.array_equal(result, vec)

    def test_get_miss(self, cache):
        cache._client.hget.return_value = None
        result = cache.get("unknown text")
        assert result is None

    def test_exists_true(self, cache):
        cache._client.exists.return_value = 1
        assert cache.exists("hello") is True

    def test_exists_false(self, cache):
        cache._client.exists.return_value = 0
        assert cache.exists("hello") is False

    def test_delete(self, cache):
        cache._client.delete.return_value = 1
        assert cache.delete("hello") is True

    def test_make_key(self, cache):
        key = cache._make_key("test")
        assert key.startswith("test:emb:")
        assert len(key) == len("test:emb:") + 64  # prefix + SHA-256

    def test_stats_tracked(self, cache):
        vec = np.array([0.0], dtype=np.float32)
        cache._client.hget.return_value = vec.tobytes()
        cache.get("text")
        assert cache.stats.hits_exact == 1

        cache._client.hget.return_value = None
        cache.get("other")
        assert cache.stats.misses == 1

    def test_reset_stats(self, cache):
        vec = np.array([0.0], dtype=np.float32)
        cache._client.hget.return_value = vec.tobytes()
        cache.get("text")
        cache.reset_stats()
        assert cache.stats.total_requests == 0

    def test_clear(self, cache):
        cache._client.scan.side_effect = [
            (0, [b"test:emb:key1", b"test:emb:key2"]),
        ]
        cache._client.delete.return_value = 2
        assert cache.clear() == 2

    def test_repr(self, cache):
        r = repr(cache)
        assert "test:emb" in r
        assert "3600" in r

    def test_count(self, cache):
        cache._client.scan.side_effect = [(0, [b"k1", b"k2", b"k3"])]
        assert cache.count() == 3


class TestSemanticCache:
    @pytest.fixture
    def mock_vectorizer(self):
        """模拟向量器，返回固定向量。"""
        vec = MagicMock()
        vec.encode.return_value = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        vec.dims = 4
        vec.name = "MockVectorizer"
        return vec

    @pytest.fixture
    def cache(self, mock_vectorizer):
        client = MagicMock()
        return SemanticCache(
            client,
            mock_vectorizer,
            name="test-cache",
            distance_threshold=0.2,
            ttl=3600,
        )

    def test_exact_hit(self, cache):
        """精确匹配命中。"""
        cache._client.hgetall.return_value = {
            b"response": b"cached response",
        }

        result = cache.lookup("什么是机器学习？")
        assert result is not None
        assert result["response"] == "cached response"
        assert result["match_type"] == "exact"
        assert result["score"] == 1.0

    def test_exact_miss_semantic_hit(self, cache):
        """精确匹配未命中，语义匹配命中。"""
        # hgetall side_effect: 第1次 = 精确未命中, 第2次 = 语义条目数据
        cache._client.hgetall.side_effect = [
            {},  # exact match: miss
            {b"response": b"semantic similar response"},  # semantic entry data
        ]
        cache._client.zrevrange.return_value = [b"entry-001"]
        cached_vec = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        cache._client.hget.return_value = cached_vec.tobytes()

        result = cache.lookup("test prompt")
        assert result is not None
        assert result["match_type"] == "semantic"
        assert result["response"] == "semantic similar response"
        assert result["score"] == 1.0  # cosine similarity = 1.0

    def test_full_miss(self, cache):
        """完全未命中。"""
        cache._client.hgetall.return_value = {}
        cache._client.zrevrange.return_value = []  # 无嵌入条目

        result = cache.lookup("从未见过的问题")
        assert result is None

    def test_semantic_below_threshold(self, cache):
        """语义相似度低于阈值视为未命中。"""
        cache._client.hgetall.side_effect = [
            {},  # exact match: miss
            {},  # semantic entry data: empty (won't be used since distance > threshold)
        ]
        cache._client.zrevrange.return_value = [b"entry-001"]
        different_vec = np.array([-0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        cache._client.hget.return_value = different_vec.tobytes()

        result = cache.lookup("completely different question")
        assert result is None  # cosine similarity is negative, distance > threshold

    def test_store(self, cache):
        assert cache.store("问题", "回答") is True
        # 验证 pipeline 被调用
        cache._client.pipeline.return_value.execute.assert_called_once()

    def test_distance_threshold(self, cache):
        assert cache.distance_threshold == 0.2

    def test_cosine_similarity(self, cache):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0], dtype=np.float32)
        assert cache._cosine_similarity(a, b) == pytest.approx(1.0)

        c = np.array([0.0, 1.0], dtype=np.float32)
        assert cache._cosine_similarity(a, c) == pytest.approx(0.0)

    def test_clear(self, cache):
        cache._client.scan.side_effect = [(0, [b"semcache:test-cache:key1"])]
        cache._client.delete.return_value = 1
        assert cache.clear() == 1

    def test_reset_stats(self, cache):
        cache._client.hgetall.return_value = {}
        cache._client.zrevrange.return_value = []
        cache.lookup("test")
        assert cache.stats.misses == 1
        cache.reset_stats()
        assert cache.stats.total_requests == 0

    def test_exact_key_format(self, cache):
        key = cache._exact_key("test")
        assert "test-cache" in key
        assert key.startswith("semcache:")

    def test_stats_tracking(self, cache):
        # 精确命中
        cache._client.hgetall.return_value = {b"response": b"ok"}
        cache.lookup("test1")
        assert cache.stats.hits_exact == 1

        # 未命中
        cache._client.hgetall.return_value = {}
        cache._client.zrevrange.return_value = []
        cache.lookup("test2")
        assert cache.stats.misses == 1
