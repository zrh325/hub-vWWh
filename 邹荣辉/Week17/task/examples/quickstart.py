"""Quickstart Example — Vector Platform SDK.

演示核心功能的端到端使用流程。
"""
import numpy as np

# ─── 1. 连接 Redis ─────────────────────────────────────────────

from vector_platform import create_connection

conn = create_connection("redis://localhost:6379/0")
print(f"Redis connected: {conn.ping()}")

# ─── 2. 定义索引Schema ─────────────────────────────────────────

from vector_platform.schema import IndexSchema

schema = IndexSchema(
    index={
        "name": "demo_index",
        "prefix": "demo",
        "storage_type": "hash",
    },
    fields=[
        {"name": "content", "type": "text"},
        {
            "name": "embedding",
            "type": "vector",
            "dims": 384,
            "algorithm": "hnsw",
            "distance_metric": "cosine",
            "datatype": "float32",
        },
        {"name": "category", "type": "tag"},
        {"name": "year", "type": "numeric"},
    ],
)
print(f"Schema: {schema.name} ({len(schema.fields)} fields)")

# ─── 3. 向量器 ─────────────────────────────────────────────────

from vector_platform.vectorizer import VectorizerFactory

factory = VectorizerFactory()
# 使用本地 sentence-transformers
# vec = factory.create_sentence_transformer("all-MiniLM-L6-v2")
# 或使用 OpenAI
# vec = factory.create_openai("text-embedding-3-small")

# 演示用假向量器
from vector_platform.vectorizer.base import VectorizerProvider


class DemoVectorizer(VectorizerProvider):
    def encode(self, text: str) -> np.ndarray:
        return np.random.randn(384).astype(np.float32)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        return np.random.randn(len(texts), 384).astype(np.float32)

    @property
    def dims(self) -> int:
        return 384


factory.register("demo", DemoVectorizer())
vectorizer = factory.get("demo")

# ─── 4. 创建索引 ───────────────────────────────────────────────

from vector_platform.schema import SearchIndex

index = SearchIndex(schema, conn.sync)
try:
    index.create(overwrite=True)
    print(f"Index '{index.name}' created")
except Exception:
    print(f"Index '{index.name}' exists, skipping creation")

# ─── 5. 灌入数据 ───────────────────────────────────────────────

documents = [
    {
        "id": f"doc_{i}",
        "content": f"This is document {i} about machine learning and AI",
        "embedding": vectorizer.encode(f"Document {i} about AI"),
        "category": ["AI", "ML"] if i % 2 == 0 else ["Data"],
        "year": 2020 + i % 5,
    }
    for i in range(10)
]
loaded = index.load(documents)
print(f"Loaded {loaded} documents")

# ─── 6. 向量搜索 ───────────────────────────────────────────────

from vector_platform.query import VectorQuery, FilterQuery, HybridQuery

vq = VectorQuery(schema, conn.sync)
query_vec = vectorizer.encode("machine learning frameworks")
results = vq.search(vector=query_vec, top_k=3)
print(f"\nVector search: found {results.total} results in {results.query_time_ms:.2f}ms")
for entry in results.entries:
    print(f"  {entry.id}: score={entry.score:.4f}")

# ─── 7. 标签过滤 ───────────────────────────────────────────────

fq = FilterQuery(schema, conn.sync)
ai_docs = fq.by_tag("category", "AI", limit=5)
print(f"\nTag filter: found {len(ai_docs)} AI documents")

# ─── 8. 混合查询 ───────────────────────────────────────────────

hq = HybridQuery(schema, conn.sync)
mixed = hq.search(
    vector=query_vec,
    filter_expr="@category:{AI}",
    top_k=5,
)
print(f"\nHybrid search: {mixed.total} results ({mixed.fusion_method})")

# ─── 9. 嵌入缓存 ───────────────────────────────────────────────

from vector_platform.cache import EmbeddingsCache

emb_cache = EmbeddingsCache(conn.sync, ttl=3600)
emb_cache.set("what is AI?", vectorizer.encode("what is AI?"))
cached_vec = emb_cache.get("what is AI?")
print(f"\nEmbedding cache: {'hit' if cached_vec is not None else 'miss'}")
print(f"Cache stats: hit_rate={emb_cache.stats.hit_rate:.1%}")

# ─── 10. 语义缓存 ──────────────────────────────────────────────

from vector_platform.cache import SemanticCache

sem_cache = SemanticCache(conn.sync, vectorizer, name="demo")
sem_cache.store("Hello, what is machine learning?", "Machine learning is a subset of AI...")

result = sem_cache.lookup("Hello, what is machine learning?")
if result:
    print(f"Semantic cache hit: type={result['match_type']}, score={result['score']:.4f}")

# ─── 11. 对话记忆 ──────────────────────────────────────────────

from vector_platform.memory import ConversationMemory
from vector_platform.types import Message

memory = ConversationMemory(conn.sync, ttl=86400)
memory.create_session("user_123", model="gpt-4o")
memory.add("user_123", Message(role="system", content="You are a helpful assistant"))
memory.add("user_123", Message(role="user", content="Tell me about AI"))

history = memory.last_n("user_123", n=3)
print(f"\nConversation history: {len(history)} messages")
for msg in history:
    print(f"  [{msg.role}]: {msg.content[:50]}...")

# ─── 12. 语义路由 ──────────────────────────────────────────────

from vector_platform.routing import SemanticRouter

router = SemanticRouter(vectorizer)
router.add_route("qa", "question answering about facts and knowledge")
router.add_route("code", "programming and code generation tasks")
router.add_route("chat", "casual conversation and chit-chat")
router.set_default("chat")

matched = router.match("How do I write a Python function?")
print(f"\nSemantic routing: '{matched.route.name}' (score={matched.score:.4f})")

print("\n✓ Vector Platform Quickstart complete!")
