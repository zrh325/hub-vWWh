"""SemanticMessageHistory — 对话历史管理（支持语义检索）.

基于 Redis 按 session_id 分离存储对话历史，支持：
  - 时间窗口检索（ZSET 按 timestamp 排序）
  - 语义检索（嵌入向量相似度搜索）🆕
  - 跨会话语义搜索 🆕
  - 自动摘要触发

Usage:
    from redis import Redis
    from vector_platform.memory import SemanticMessageHistory
    from vector_platform.types import Message

    redis_client = Redis(host="localhost", port=6379, decode_responses=False)
    memory = SemanticMessageHistory(
        client=redis_client,
        vectorizer=my_vectorizer,  # 任何有 .encode(text) -> np.ndarray 的对象
        ttl=86400,
    )

    # 创建会话
    memory.create_session("session_123", model="gpt-4", user_id="user_1")

    # 添加消息
    memory.add("session_123", Message(role="user", content="你好"))
    memory.add("session_123", Message(role="assistant", content="你好！有什么可以帮助你的？"))

    # 时间窗口检索
    history = memory.get_history("session_123", window=10)

    # 语义检索 🆕
    results = memory.search("session_123", "问候语", top_k=5)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Optional

import numpy as np
from redis import Redis

from vector_platform.types import Message

logger = logging.getLogger(__name__)


class SemanticMessageHistory:
    """对话记忆管理器 — 按 session_id 分区存储对话，支持语义搜索.

    Key 模式:
        session:{session_id}:messages       — ZSET (score=timestamp, member=Message JSON)
        session:{session_id}:meta            — HASH (created_at, last_active, model, user_id, ...)
        session:{session_id}:summary         — STRING (摘要文本)
        session:{session_id}:msg_embeddings  — ZSET (score=timestamp, member=content_hash) 🆕
        session:{session_id}:msg_emb:{hash}  — HASH (role, content, embedding, timestamp) 🆕

    Usage:
        memory = SemanticMessageHistory(redis_client, vectorizer, ttl=86400)

        # 添加消息
        memory.add("session_123", Message(role="user", content="你好"))

        # 时间检索
        history = memory.get_history("session_123", window=10)

        # 语义检索
        results = memory.search("session_123", "打招呼", top_k=5)
    """

    def __init__(
        self,
        client: Redis,
        vectorizer,  # duck typing: .encode(text) -> np.ndarray
        ttl: int = 86400,  # 默认24小时
        max_window: int = 100,
        summary_trigger: int = 50,
        max_semantic_scan: int = 200,
    ):
        """初始化对话记忆.

        Args:
            client: Redis 客户端
            vectorizer: 向量器 — 需要实现 .encode(text) -> np.ndarray
            ttl: 会话过期时间（秒），0 表示永不过期
            max_window: 默认最大窗口大小
            summary_trigger: 触发摘要的消息数阈值
            max_semantic_scan: 语义搜索时最多扫描的消息数（从最新开始）
        """
        self._client = client
        self._vectorizer = vectorizer
        self._ttl = ttl
        self._max_window = max_window
        self._summary_trigger = summary_trigger
        self._max_semantic_scan = max_semantic_scan

    # ─── 属性 ────────────────────────────────────────────────────

    @property
    def ttl(self) -> int:
        return self._ttl

    @property
    def max_window(self) -> int:
        return self._max_window

    @property
    def max_semantic_scan(self) -> int:
        return self._max_semantic_scan

    # ─── Key 生成 ─────────────────────────────────────────────────

    @staticmethod
    def _messages_key(session_id: str) -> str:
        return f"session:{session_id}:messages"

    @staticmethod
    def _meta_key(session_id: str) -> str:
        return f"session:{session_id}:meta"

    @staticmethod
    def _summary_key(session_id: str) -> str:
        return f"session:{session_id}:summary"

    @staticmethod
    def _embeddings_key(session_id: str) -> str:
        """消息嵌入 ZSET Key."""
        return f"session:{session_id}:msg_embeddings"

    @staticmethod
    def _emb_entry_key(session_id: str, content_hash: str) -> str:
        """单条消息嵌入 HASH Key."""
        return f"session:{session_id}:msg_emb:{content_hash}"

    @staticmethod
    def _hash_text(text: str) -> str:
        """计算文本的 SHA-256 哈希（用于嵌入 Key 中的 content_hash）."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _session_pattern() -> str:
        return "session:*:meta"

    # ─── 会话管理 ────────────────────────────────────────────────

    def create_session(
        self,
        session_id: str,
        model: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """创建新会话.

        Args:
            session_id: 会话ID
            model: 使用的模型名称
            user_id: 用户ID
            metadata: 自定义元数据

        Returns:
            是否创建成功
        """
        now = time.time()
        meta: dict[str, Any] = {
            "created_at": str(now),
            "last_active": str(now),
        }
        if model:
            meta["model"] = model
        if user_id:
            meta["user_id"] = user_id
        if metadata:
            meta["metadata"] = json.dumps(metadata, ensure_ascii=False)

        meta_key = self._meta_key(session_id)
        try:
            pipe = self._client.pipeline()
            pipe.hset(meta_key, mapping=meta)
            if self._ttl > 0:
                pipe.expire(meta_key, self._ttl)
            pipe.execute()
            return True
        except Exception as e:
            logger.error("Failed to create session '%s': %s", session_id, e)
            return False

    def delete_session(self, session_id: str) -> int:
        """删除会话和所有关联数据.

        Args:
            session_id: 会话ID

        Returns:
            删除的 Key 数量
        """
        keys = [
            self._messages_key(session_id),
            self._meta_key(session_id),
            self._summary_key(session_id),
            self._embeddings_key(session_id),
        ]
        # 也尝试删除消息嵌入 HASH（SCAN 匹配）
        pattern = f"session:{session_id}:msg_emb:*"
        try:
            cursor = 0
            while True:
                cursor, batch = self._client.scan(cursor, match=pattern, count=1000)
                keys.extend(batch)
                if cursor == 0:
                    break
        except Exception:
            pass

        try:
            return self._client.delete(*keys)
        except Exception as e:
            logger.error("Failed to delete session '%s': %s", session_id, e)
            return 0

    def session_exists(self, session_id: str) -> bool:
        """检查会话是否存在.

        Args:
            session_id: 会话ID

        Returns:
            是否存在
        """
        try:
            return bool(self._client.exists(self._meta_key(session_id)))
        except Exception:
            return False

    # ─── 消息操作 ────────────────────────────────────────────────

    def add(self, session_id: str, message: Message) -> bool:
        """添加一条消息（同时存储时间线 + 语义嵌入）.

        Args:
            session_id: 会话ID
            message: Message 对象

        Returns:
            是否添加成功
        """
        msg_key = self._messages_key(session_id)
        meta_key = self._meta_key(session_id)
        emb_zset_key = self._embeddings_key(session_id)
        now = time.time()

        if message.timestamp <= 0:
            message.timestamp = now

        payload = json.dumps(message.to_dict(), ensure_ascii=False)
        content_hash = self._hash_text(message.content)

        try:
            pipe = self._client.pipeline()

            # 时间线存储
            pipe.zadd(msg_key, {payload: message.timestamp})
            pipe.hset(meta_key, "last_active", str(now))

            # 语义嵌入存储（只对 user 和 assistant 消息存储 embedding）
            if message.role in ("user", "assistant") and message.content.strip():
                embedding = self._vectorizer.encode(message.content)
                emb_entry_key = self._emb_entry_key(session_id, content_hash)
                emb_data = {
                    "role": message.role,
                    "content": message.content,
                    "embedding": embedding.astype(np.float32).tobytes(),
                    "timestamp": str(message.timestamp),
                }
                pipe.hset(emb_entry_key, mapping=emb_data)
                # ZSET 映射 timestamp → content_hash（用于按时间排序扫描）
                pipe.zadd(emb_zset_key, {content_hash: message.timestamp})
                if self._ttl > 0:
                    pipe.expire(emb_entry_key, self._ttl)
                    pipe.expire(emb_zset_key, self._ttl)

            # TTL
            if self._ttl > 0:
                pipe.expire(msg_key, self._ttl)
                pipe.expire(meta_key, self._ttl)

            pipe.execute()
        except Exception as e:
            logger.error("Failed to add message to session '%s': %s", session_id, e)
            return False

        # 检查是否到达摘要触发阈值
        try:
            count = self._client.zcard(msg_key)
            if count >= self._summary_trigger:
                logger.info(
                    "Session '%s' has %d messages (trigger=%d), consider summarizing",
                    session_id, count, self._summary_trigger,
                )
        except Exception:
            pass

        return True

    def add_many(self, session_id: str, messages: list[Message]) -> bool:
        """批量添加消息.

        Args:
            session_id: 会话ID
            messages: 消息列表

        Returns:
            是否全部添加成功
        """
        success = True
        for msg in messages:
            if not self.add(session_id, msg):
                success = False
        return success

    # ─── 时间窗口检索 ─────────────────────────────────────────────

    def get_history(
        self,
        session_id: str,
        window: Optional[int] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        include_system: bool = True,
    ) -> list[Message]:
        """获取对话历史（时间维度）.

        Args:
            session_id: 会话ID
            window: 最近N条消息（None=全部）
            start_time: 起始时间戳
            end_time: 结束时间戳
            include_system: 是否包含system消息

        Returns:
            消息列表（按时间升序）
        """
        msg_key = self._messages_key(session_id)

        try:
            if window is not None:
                raw = self._client.zrevrange(msg_key, 0, window - 1)
            elif start_time is not None or end_time is not None:
                min_score = str(start_time) if start_time is not None else "-inf"
                max_score = str(end_time) if end_time is not None else "+inf"
                raw = self._client.zrangebyscore(msg_key, min_score, max_score)
            else:
                raw = self._client.zrange(msg_key, 0, -1)
        except Exception as e:
            logger.error("Failed to get history for session '%s': %s", session_id, e)
            return []

        messages = self._parse_messages(raw, include_system)

        # window 模式返回的是倒序，需要反转
        if window is not None:
            messages.reverse()

        return messages

    def last_n(self, session_id: str, n: int = 10) -> list[Message]:
        """获取最后 N 条消息.

        Args:
            session_id: 会话ID
            n: 消息数量

        Returns:
            消息列表（按时间升序）
        """
        return self.get_history(session_id, window=n)

    def last_user_message(self, session_id: str) -> Optional[Message]:
        """获取最后一条用户消息.

        Args:
            session_id: 会话ID

        Returns:
            最后一条用户消息或 None
        """
        msg_key = self._messages_key(session_id)
        try:
            raw = self._client.zrevrange(msg_key, 0, -1)
        except Exception:
            return None

        for item in reversed(raw):
            if isinstance(item, bytes):
                item = item.decode()
            try:
                msg = Message.from_dict(json.loads(item))
                if msg.role == "user":
                    return msg
            except Exception:
                pass
        return None

    # ─── 语义检索 🆕 ──────────────────────────────────────────────

    def search(
        self,
        session_id: str,
        query: str,
        top_k: int = 5,
        threshold: float = 0.3,
    ) -> list[tuple[Message, float]]:
        """在当前会话中进行语义搜索.

        将查询文本编码为向量，与最近消息的嵌入向量计算余弦相似度，
        返回相似度最高且超过阈值的 top_k 条消息。

        Args:
            session_id: 会话ID
            query: 查询文本
            top_k: 返回最相似的 K 条消息
            threshold: 相似度阈值（低于此值的消息被过滤，范围 0~1）

        Returns:
            [(Message, similarity_score), ...] 按相似度降序排列
        """
        query_vec = self._vectorizer.encode(query)

        # 获取最近 N 条消息的 content_hash
        emb_zset_key = self._embeddings_key(session_id)
        try:
            members = self._client.zrevrange(
                emb_zset_key, 0, self._max_semantic_scan - 1
            )
        except Exception:
            return []

        if not members:
            return []

        # 加载嵌入并计算相似度
        scored: list[tuple[Message, float]] = []
        for member in members:
            content_hash = member.decode() if isinstance(member, bytes) else member
            emb_key = self._emb_entry_key(session_id, content_hash)

            try:
                entry = self._client.hgetall(emb_key)
            except Exception:
                continue

            if not entry:
                continue

            emb_blob = entry.get(b"embedding", entry.get("embedding"))
            if emb_blob is None:
                continue

            try:
                msg_vec = np.frombuffer(emb_blob, dtype=np.float32)
            except Exception:
                continue

            similarity = self._cosine_similarity(query_vec, msg_vec)
            if similarity >= threshold:
                role = entry.get(b"role", entry.get("role", "user"))
                content = entry.get(b"content", entry.get("content", ""))
                ts = entry.get(b"timestamp", entry.get("timestamp", "0"))
                if isinstance(role, bytes):
                    role = role.decode()
                if isinstance(content, bytes):
                    content = content.decode()
                if isinstance(ts, bytes):
                    ts = ts.decode()

                msg = Message(
                    role=role,
                    content=content,
                    timestamp=float(ts),
                )
                scored.append((msg, float(similarity)))

        # 按相似度降序排列，取 top_k
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def search_across_sessions(
        self,
        query: str,
        session_ids: Optional[list[str]] = None,
        top_k: int = 10,
        threshold: float = 0.3,
    ) -> list[tuple[str, Message, float]]:
        """跨多个会话进行语义搜索.

        Args:
            query: 查询文本
            session_ids: 要搜索的会话 ID 列表（None 表示搜索所有会话）
            top_k: 返回最相似的 K 条消息
            threshold: 相似度阈值

        Returns:
            [(session_id, Message, similarity_score), ...] 按相似度降序排列
        """
        # 如果没有指定会话，扫描所有会话
        if session_ids is None:
            session_ids = self.list_sessions()

        all_scored: list[tuple[str, Message, float]] = []

        for sid in session_ids:
            results = self.search(sid, query, top_k=top_k * 2, threshold=threshold)
            for msg, score in results:
                all_scored.append((sid, msg, score))

        # 全局排序，取 top_k
        all_scored.sort(key=lambda x: x[2], reverse=True)
        return all_scored[:top_k]

    # ─── 摘要 ────────────────────────────────────────────────────

    def get_summary(self, session_id: str) -> Optional[str]:
        """获取会话摘要.

        Args:
            session_id: 会话ID

        Returns:
            摘要文本或 None
        """
        try:
            val = self._client.get(self._summary_key(session_id))
        except Exception:
            return None

        if isinstance(val, bytes):
            return val.decode()
        return val

    def set_summary(self, session_id: str, summary: str) -> bool:
        """设置会话摘要.

        Args:
            session_id: 会话ID
            summary: 摘要文本

        Returns:
            是否设置成功
        """
        key = self._summary_key(session_id)
        try:
            if self._ttl > 0:
                self._client.setex(key, self._ttl, summary)
            else:
                self._client.set(key, summary)
            return True
        except Exception as e:
            logger.error("Failed to set summary for session '%s': %s", session_id, e)
            return False

    def should_summarize(self, session_id: str) -> bool:
        """检查是否应该触发摘要（消息数超过阈值）.

        Args:
            session_id: 会话ID

        Returns:
            是否应触发摘要
        """
        try:
            count = self._client.zcard(self._messages_key(session_id))
        except Exception:
            return False
        return count >= self._summary_trigger

    # ─── 信息 ────────────────────────────────────────────────────

    def count(self, session_id: str) -> int:
        """获取会话消息数.

        Args:
            session_id: 会话ID

        Returns:
            消息数量
        """
        try:
            return self._client.zcard(self._messages_key(session_id))
        except Exception:
            return 0

    def token_count(self, session_id: str) -> int:
        """估算会话总 token 数.

        Args:
            session_id: 会话ID

        Returns:
            估算 token 数
        """
        messages = self.get_history(session_id)
        return sum(m.token_count for m in messages)

    def session_meta(self, session_id: str) -> dict[str, Any]:
        """获取会话元数据.

        Args:
            session_id: 会话ID

        Returns:
            元数据字典
        """
        try:
            raw = self._client.hgetall(self._meta_key(session_id))
        except Exception:
            return {}

        meta: dict[str, Any] = {}
        for k, v in raw.items():
            if isinstance(k, bytes):
                k = k.decode()
            if isinstance(v, bytes):
                v = v.decode()
            meta[k] = v
        return meta

    def list_sessions(self) -> list[str]:
        """列出所有会话 ID（通过 SCAN 匹配 session meta keys）.

        Returns:
            会话 ID 列表
        """
        pattern = self._session_pattern()
        session_ids: list[str] = []
        try:
            cursor = 0
            while True:
                cursor, batch = self._client.scan(cursor, match=pattern, count=1000)
                for key in batch:
                    key_str = key.decode() if isinstance(key, bytes) else key
                    # key 格式: session:{id}:meta → 提取 id
                    parts = key_str.split(":")
                    if len(parts) >= 3 and parts[0] == "session" and parts[-1] == "meta":
                        session_id = ":".join(parts[1:-1])
                        session_ids.append(session_id)
                if cursor == 0:
                    break
        except Exception as e:
            logger.error("Failed to list sessions: %s", e)
        return session_ids

    # ─── 内部方法 ─────────────────────────────────────────────────

    @staticmethod
    def _parse_messages(raw: list, include_system: bool) -> list[Message]:
        """将原始 ZSET 数据解析为 Message 列表."""
        messages = []
        for item in raw:
            if isinstance(item, bytes):
                item = item.decode()
            try:
                msg = Message.from_dict(json.loads(item))
                if not include_system and msg.role == "system":
                    continue
                messages.append(msg)
            except Exception as e:
                logger.warning("Failed to parse message: %s", e)
        return messages

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def __repr__(self) -> str:
        return (
            f"<SemanticMessageHistory ttl={self._ttl}s "
            f"max_window={self._max_window} max_semantic_scan={self._max_semantic_scan}>"
        )
