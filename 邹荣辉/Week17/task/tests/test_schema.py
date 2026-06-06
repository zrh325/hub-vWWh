"""Schema 模块单元测试。"""

import tempfile
from pathlib import Path

import pytest
import yaml

from vector_platform.milvus_index import IndexSchema, SearchIndex, StorageType


class TestIndexSchema:
    def test_create_minimal(self):
        schema = IndexSchema(
            index={"name": "docs", "prefix": "doc"},
        )
        assert schema.name == "docs"
        assert schema.prefix == "doc"
        assert schema.storage_type == StorageType.HASH
        assert schema.fields == []

    def test_create_with_fields(self, sample_schema_dict):
        schema = IndexSchema(**sample_schema_dict)
        assert schema.name == "test_index"
        assert len(schema.fields) == 4
        assert len(schema.vector_fields) == 1
        assert len(schema.text_fields) == 1
        assert len(schema.tag_fields) == 1
        assert len(schema.numeric_fields) == 1

    def test_invalid_index_missing_name(self):
        with pytest.raises(ValueError, match="must contain 'name'"):
            IndexSchema(index={"prefix": "doc"})

    def test_invalid_index_missing_prefix(self):
        with pytest.raises(ValueError, match="must contain 'prefix'"):
            IndexSchema(index={"name": "docs"})

    def test_invalid_storage_type(self):
        with pytest.raises(ValueError, match="storage_type must be"):
            IndexSchema(index={"name": "docs", "prefix": "doc", "storage_type": "set"})

    def test_invalid_field_type(self):
        with pytest.raises(ValueError, match="Unknown field type"):
            IndexSchema(
                index={"name": "docs", "prefix": "doc"},
                fields=[{"name": "bad", "type": "unknown"}],
            )

    def test_vector_field_requires_dims(self):
        with pytest.raises(ValueError):
            IndexSchema(
                index={"name": "docs", "prefix": "doc"},
                fields=[{"name": "vec", "type": "vector"}],
            )

    def test_field_names(self, sample_schema_dict):
        schema = IndexSchema(**sample_schema_dict)
        assert "content" in schema.field_names
        assert "embedding" in schema.field_names
        assert "category" in schema.field_names

    def test_get_vector_field(self, sample_schema_dict):
        schema = IndexSchema(**sample_schema_dict)
        vec = schema.get_vector_field()
        assert vec is not None
        assert vec["name"] == "embedding"
        assert vec["dims"] == 384

    def test_no_vector_field(self):
        schema = IndexSchema(
            index={"name": "docs", "prefix": "doc"},
            fields=[{"name": "title", "type": "text"}],
        )
        assert schema.get_vector_field() is None
        assert schema.vector_fields == []

    def test_key_for_entry(self):
        schema = IndexSchema(index={"name": "docs", "prefix": "doc"})
        assert schema.key_for("123") == "doc:123"

    def test_to_ft_create(self, sample_schema_dict):
        schema = IndexSchema(**sample_schema_dict)
        cmd = schema.to_ft_create()
        assert "FT.CREATE" in cmd
        assert schema.name in cmd
        assert "PREFIX 1 test:" in cmd
        assert "VECTOR HNSW" in cmd
        assert "DISTANCE_METRIC COSINE" in cmd

    def test_from_dict(self, sample_schema_dict):
        schema = IndexSchema.from_dict(sample_schema_dict)
        assert schema.name == "test_index"

    def test_to_dict(self, sample_schema_dict):
        schema = IndexSchema(**sample_schema_dict)
        d = schema.to_dict()
        assert d["index"]["name"] == "test_index"
        assert len(d["fields"]) == 4

    def test_yaml_roundtrip(self, sample_schema_dict):
        schema = IndexSchema(**sample_schema_dict)
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            temp_path = f.name
            schema.to_yaml(temp_path)

        try:
            loaded = IndexSchema.from_yaml(temp_path)
            assert loaded.name == schema.name
            assert loaded.prefix == schema.prefix
            assert len(loaded.fields) == len(schema.fields)
        finally:
            Path(temp_path).unlink()

    def test_from_yaml_not_found(self):
        with pytest.raises(FileNotFoundError):
            IndexSchema.from_yaml("nonexistent.yaml")

    def test_repr(self, sample_schema_dict):
        schema = IndexSchema(**sample_schema_dict)
        r = repr(schema)
        assert "test_index" in r
        assert "test" in r


class TestSearchIndex:
    def test_create_index(self, mock_redis, sample_schema_dict):
        mock_redis.execute_command.side_effect = [
            Exception("not found"),  # exists() -> False
            1,  # create success
        ]
        schema = IndexSchema(**sample_schema_dict)
        index = SearchIndex(schema, mock_redis)
        assert index.create() is True
        assert mock_redis.execute_command.call_count >= 2  # exists check + FT.CREATE

    def test_create_skip_existing(self, mock_redis, sample_schema_dict):
        mock_redis.execute_command.return_value = 1  # exists -> True
        schema = IndexSchema(**sample_schema_dict)
        index = SearchIndex(schema, mock_redis)
        assert index.create(overwrite=False) is False

    def test_create_overwrite(self, mock_redis, sample_schema_dict):
        mock_redis.execute_command.side_effect = [
            1,  # exists() -> True
            1,  # drop success
            1,  # create success
        ]
        schema = IndexSchema(**sample_schema_dict)
        index = SearchIndex(schema, mock_redis)
        assert index.create(overwrite=True) is True

    def test_drop(self, mock_redis, sample_schema_dict):
        mock_redis.execute_command.return_value = 1
        schema = IndexSchema(**sample_schema_dict)
        index = SearchIndex(schema, mock_redis)
        assert index.drop() is True

    def test_exists_true(self, mock_redis, sample_schema_dict):
        mock_redis.execute_command.return_value = [
            b"index_name", sample_schema_dict["index"]["name"],
            b"num_docs", b"42",
        ]
        schema = IndexSchema(**sample_schema_dict)
        index = SearchIndex(schema, mock_redis)
        assert index.exists() is True

    def test_exists_false(self, mock_redis, sample_schema_dict):
        mock_redis.execute_command.side_effect = Exception("Unknown Index name")
        schema = IndexSchema(**sample_schema_dict)
        index = SearchIndex(schema, mock_redis)
        assert index.exists() is False

    def test_properties(self, mock_redis, sample_schema_dict):
        schema = IndexSchema(**sample_schema_dict)
        index = SearchIndex(schema, mock_redis)
        assert index.name == "test_index"
        assert index.prefix == "test"
        assert index.schema is schema

    def test_repr(self, mock_redis, sample_schema_dict):
        schema = IndexSchema(**sample_schema_dict)
        index = SearchIndex(schema, mock_redis)
        r = repr(index)
        assert "test_index" in r
