"""Query 模块单元测试 — VectorQuery, FilterQuery, TextQuery, HybridQuery, QueryBuilder。"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from vector_platform.milvus_index import (
    FilterQuery,
    HybridQuery,
    IndexSchema,
    QueryBuilder,
    TextQuery,
    VectorQuery,
    linear_weighted_fusion,
    reciprocal_rank_fusion,
)
from vector_platform.milvus_index import FusionMethod, VectorEntry


@pytest.fixture
def schema():
    return IndexSchema(
        index={"name": "docs", "prefix": "doc", "storage_type": "hash"},
        fields=[
            {"name": "content", "type": "text"},
            {"name": "embedding", "type": "vector", "dims": 4, "distance_metric": "cosine"},
            {"name": "category", "type": "tag"},
            {"name": "year", "type": "numeric", "sortable": True},
        ],
    )


@pytest.fixture
def query_vec():
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _make_ft_result(total, entries_data):
    """构造 FT.SEARCH 返回格式。"""
    result = [total]
    for entry in entries_data:
        result.append(f"doc:{entry['id']}".encode())
        fields = []
        for k, v in entry.items():
            if k == "score":
                k = "__embedding_score"
            fields.append(k.encode())
            fields.append(str(v).encode() if not isinstance(v, bytes) else v)
        result.append(fields)
    return result


class TestVectorQuery:
    def test_search_returns_results(self, schema):
        mock_client = MagicMock()
        mock_client.execute_command.return_value = _make_ft_result(1, [
            {"id": "1", "content": "hello world", "category": "AI", "score": "0.95"},
        ])

        query = VectorQuery(schema, mock_client)
        result = query.search(vector=[1.0, 0.0, 0.0, 0.0], top_k=5)

        assert len(result) == 1
        assert result.entries[0].id == "1"
        assert result.entries[0].fields["content"] == "hello world"

    def test_search_empty(self, schema):
        mock_client = MagicMock()
        mock_client.execute_command.return_value = [0]

        query = VectorQuery(schema, mock_client)
        result = query.search(vector=[0.0, 0.0, 0.0, 0.0])

        assert len(result) == 0
        assert not result

    def test_search_with_filter(self, schema):
        mock_client = MagicMock()
        mock_client.execute_command.return_value = _make_ft_result(0, [])

        query = VectorQuery(schema, mock_client)
        result = query.search(
            vector=[1.0, 0.0, 0.0, 0.0],
            filter_expr="@category:{AI}",
        )
        assert len(result) == 0

    def test_no_vector_field_raises(self):
        schema_no_vec = IndexSchema(
            index={"name": "docs", "prefix": "doc"},
            fields=[{"name": "title", "type": "text"}],
        )
        query = VectorQuery(schema_no_vec, MagicMock())
        with pytest.raises(ValueError, match="No vector field"):
            query.search(vector=[1.0, 2.0])

    def test_repr(self, schema):
        query = VectorQuery(schema, MagicMock())
        assert "docs" in repr(query)


class TestFilterQuery:
    def test_filter_by_tag(self, schema):
        mock_client = MagicMock()
        mock_client.execute_command.return_value = _make_ft_result(2, [
            {"id": "1", "content": "doc1", "category": "AI"},
            {"id": "2", "content": "doc2", "category": "AI"},
        ])

        query = FilterQuery(schema, mock_client)
        results = query.by_tag("category", "AI")

        assert len(results) == 2
        assert results[0].fields["category"] == "AI"

    def test_filter_by_range(self, schema):
        mock_client = MagicMock()
        mock_client.execute_command.return_value = _make_ft_result(1, [
            {"id": "3", "year": "2023", "content": "recent"},
        ])

        query = FilterQuery(schema, mock_client)
        results = query.by_range("year", min_val=2020, max_val=2025)

        assert len(results) == 1

    def test_count(self, schema):
        mock_client = MagicMock()
        mock_client.execute_command.return_value = [42]

        query = FilterQuery(schema, mock_client)
        assert query.count() == 42


class TestTextQuery:
    def test_search(self, schema):
        mock_client = MagicMock()
        mock_client.execute_command.return_value = _make_ft_result(1, [
            {"id": "1", "content": "机器学习入门教程"},
        ])

        query = TextQuery(schema, mock_client)
        result = query.search("机器学习")

        assert len(result) == 1
        assert result.entries[0].fields["content"] == "机器学习入门教程"

    def test_search_simple(self, schema):
        mock_client = MagicMock()
        mock_client.execute_command.return_value = _make_ft_result(1, [
            {"id": "42", "content": "test"},
        ])

        query = TextQuery(schema, mock_client)
        entries = query.search_simple("test", limit=10)
        assert len(entries) == 1


class TestHybridFusion:
    """RRF 和 LinearWeighted 融合算法测试。"""

    def test_reciprocal_rank_fusion(self):
        e1 = VectorEntry(id="a", vector=np.array([1.0]), score=0.9)
        e2 = VectorEntry(id="b", vector=np.array([2.0]), score=0.7)
        e3 = VectorEntry(id="a", vector=np.array([1.0]), score=0.6)  # duplicate
        e4 = VectorEntry(id="c", vector=np.array([3.0]), score=0.5)

        results = reciprocal_rank_fusion([
            [e1, e2],  # rank 1: a, rank 2: b
            [e3, e4],  # rank 1: a, rank 2: c
        ])

        # a appears in both lists at rank 1 -> highest RRF score
        assert results[0].id == "a"
        assert len(results) == 3  # unique entries

    def test_linear_weighted_fusion(self):
        e1 = VectorEntry(id="a", vector=np.array([1.0]), score=1.0)
        e2 = VectorEntry(id="b", vector=np.array([2.0]), score=0.5)
        e3 = VectorEntry(id="c", vector=np.array([3.0]), score=0.8)

        results = linear_weighted_fusion(
            [[e1, e2], [e3]],
            [0.7, 0.3],
        )
        assert len(results) == 3

    def test_linear_weighted_mismatch(self):
        with pytest.raises(ValueError):
            linear_weighted_fusion([[VectorEntry(id="a", vector=np.array([1.0]))]], [0.5, 0.5])


class TestQueryBuilder:
    def test_chain_api(self, schema):
        mock_client = MagicMock()
        mock_client.execute_command.return_value = _make_ft_result(2, [
            {"id": "1", "content": "hello", "category": "AI", "score": "0.9"},
            {"id": "2", "content": "world", "category": "ML", "score": "0.8"},
        ])

        result = (
            QueryBuilder(schema, mock_client)
            .vector([1.0, 0.0, 0.0, 0.0])
            .top_k(5)
            .execute()
        )
        assert len(result) == 2

    def test_builder_repr(self, schema):
        builder = (
            QueryBuilder(schema, MagicMock())
            .vector([1.0, 2.0, 3.0, 4.0])
            .text("test")
            .tag("category", "AI")
        )
        r = repr(builder)
        assert "vector" in r
        assert "text" in r
        assert "filter" in r
