"""Memory 模块单元测试 — SemanticMessageHistory。"""

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from vector_platform.memory import SemanticMessageHistory
from vector_platform.types import Message


class TestSemanticMessageHistory:
    @pytest.fixture
    def mock_vectorizer(self):
        vec = MagicMock()
        vec.encode.return_value = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return vec

    @pytest.fixture
    def memory(self, mock_vectorizer):
        client = MagicMock()
        return SemanticMessageHistory(client, mock_vectorizer, ttl=3600, max_window=50)

    def test_create_session(self, memory):
        assert memory.create_session("session_1", model="gpt-4o", user_id="user_1")

    def test_delete_session(self, memory):
        memory._client.delete.return_value = 3
        assert memory.delete_session("session_1") == 3

    def test_session_exists(self, memory):
        memory._client.exists.return_value = 1
        assert memory.session_exists("session_1") is True

        memory._client.exists.return_value = 0
        assert memory.session_exists("session_2") is False

    def test_add_message(self, memory):
        memory._client.zcard.return_value = 5  # 低于摘要阈值
        msg = Message(role="user", content="hello", token_count=1)
        assert memory.add("session_1", msg) is True

    def test_add_many(self, memory):
        memory._client.zcard.return_value = 3
        messages = [
            Message(role="user", content="q1"),
            Message(role="assistant", content="a1"),
        ]
        assert memory.add_many("session_1", messages) is True

    def test_get_history(self, memory):
        msg1 = Message(role="user", content="question", timestamp=100.0)
        msg2 = Message(role="assistant", content="answer", timestamp=101.0)
        memory._client.zrange.return_value = [
            json.dumps(msg1.to_dict(), ensure_ascii=False).encode(),
            json.dumps(msg2.to_dict(), ensure_ascii=False).encode(),
        ]

        history = memory.get_history("session_1")
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[1].role == "assistant"

    def test_get_history_window(self, memory):
        msg = Message(role="user", content="q", timestamp=100.0)
        memory._client.zrevrange.return_value = [
            json.dumps(msg.to_dict(), ensure_ascii=False).encode(),
        ]

        history = memory.get_history("session_1", window=5)
        assert len(history) == 1

    def test_last_n(self, memory):
        msg = Message(role="user", content="recent", timestamp=100.0)
        memory._client.zrevrange.return_value = [
            json.dumps(msg.to_dict(), ensure_ascii=False).encode(),
        ]

        history = memory.last_n("session_1", n=3)
        assert len(history) == 1

    def test_last_user_message(self, memory):
        msg = Message(role="user", content="last question", timestamp=100.0)
        memory._client.zrevrange.return_value = [
            json.dumps(msg.to_dict(), ensure_ascii=False).encode(),
        ]

        result = memory.last_user_message("session_1")
        assert result is not None
        assert result.role == "user"
        assert result.content == "last question"

    def test_last_user_message_none(self, memory):
        memory._client.zrevrange.return_value = []
        result = memory.last_user_message("session_1")
        assert result is None

    def test_count(self, memory):
        memory._client.zcard.return_value = 42
        assert memory.count("session_1") == 42

    def test_summary_operations(self, memory):
        memory._client.get.return_value = b"Session summary text"
        summary = memory.get_summary("session_1")
        assert summary == "Session summary text"

    def test_set_summary(self, memory):
        assert memory.set_summary("session_1", "New summary") is True

    def test_should_summarize(self, memory):
        memory._client.zcard.return_value = 60  # above default trigger of 50
        assert memory.should_summarize("session_1") is True

        memory._client.zcard.return_value = 30
        assert memory.should_summarize("session_1") is False

    def test_session_meta(self, memory):
        memory._client.hgetall.return_value = {
            b"created_at": b"100.0",
            b"model": b"gpt-4o",
        }
        meta = memory.session_meta("session_1")
        assert meta["model"] == "gpt-4o"

    def test_key_formats(self, memory):
        assert "session:s1:messages" in memory._messages_key("s1")
        assert "session:s1:meta" in memory._meta_key("s1")
        assert "session:s1:summary" in memory._summary_key("s1")
