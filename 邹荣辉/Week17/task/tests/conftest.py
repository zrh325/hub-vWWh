"""测试共享 fixtures。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def mock_redis():
    """模拟 Redis 客户端。"""
    mock = MagicMock()
    mock.ping.return_value = True
    mock.execute_command.return_value = [1]
    mock.pipeline.return_value = MagicMock()
    return mock


@pytest.fixture
def sample_schema_dict():
    """示例索引Schema字典。"""
    return {
        "index": {
            "name": "test_index",
            "prefix": "test",
            "storage_type": "hash",
        },
        "fields": [
            {
                "name": "content",
                "type": "text",
                "weight": 1.0,
            },
            {
                "name": "embedding",
                "type": "vector",
                "dims": 384,
                "algorithm": "hnsw",
                "distance_metric": "cosine",
                "datatype": "float32",
            },
            {
                "name": "category",
                "type": "tag",
            },
            {
                "name": "year",
                "type": "numeric",
                "sortable": True,
            },
        ],
    }


@pytest.fixture
def sample_vector():
    """示例向量（384维）。"""
    return np.random.randn(384).astype(np.float32).tolist()


@pytest.fixture
def sample_entries():
    """示例数据条目。"""
    return [
        {
            "id": f"doc_{i}",
            "content": f"文档内容 {i}: 这是一段关于机器学习和人工智能的文本",
            "embedding": np.random.randn(384).astype(np.float32),
            "category": ["AI", "ML"] if i % 2 == 0 else ["Data"],
            "year": 2020 + i,
        }
        for i in range(5)
    ]
