"""Routing 模块单元测试 — SemanticRouter。"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from vector_platform.router import Route, RouteMatch, SemanticRouter


class TestRoute:
    def test_create_route(self):
        r = Route(name="weather", description="weather queries")
        assert r.name == "weather"
        assert r.description == "weather queries"
        assert r.embedding is None

    def test_route_with_embedding(self):
        vec = np.array([1.0, 0.0], dtype=np.float32)
        r = Route(name="math", description="math queries", embedding=vec)
        assert np.array_equal(r.embedding, vec)


class TestRouteMatch:
    def test_matched(self):
        r = Route(name="test", description="test")
        m = RouteMatch(route=r, score=0.95)
        assert m.matched is True
        assert m.fallback is False

    def test_fallback(self):
        r = Route(name="default", description="default")
        m = RouteMatch(route=r, score=0.0, matched=False, fallback=True)
        assert m.matched is False
        assert m.fallback is True


class TestSemanticRouter:
    @pytest.fixture
    def mock_vectorizer(self):
        vec = MagicMock()
        # 不同文本返回不同向量以便区分路由
        def encode(text):
            if "weather" in text.lower():
                return np.array([1.0, 0.0, 0.0], dtype=np.float32)
            elif "math" in text.lower() or "calculation" in text.lower():
                return np.array([0.0, 1.0, 0.0], dtype=np.float32)
            elif "goodbye" in text.lower():
                return np.array([-0.5, -0.5, -0.5], dtype=np.float32)
            else:
                return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        vec.encode.side_effect = encode
        return vec

    @pytest.fixture
    def router(self, mock_vectorizer):
        r = SemanticRouter(mock_vectorizer, distance_threshold=0.3)
        r.add_route("weather", "weather related queries", examples=["what is the weather?"])
        r.add_route("math", "math calculation requests", examples=["calculate 2+2"])
        r.set_default("general")
        r.add_route("general", "general questions", examples=["how are you?"])
        return r

    def test_match_weather(self, router):
        result = router.match("what is the weather today?")
        assert result.route.name == "weather"
        assert result.matched is True

    def test_match_math(self, router):
        result = router.match("please do a math calculation")
        assert result.route.name == "math"
        assert result.matched is True

    def test_match_fallback(self, router):
        """不匹配任何路由时回退到默认。"""
        result = router.match("goodbye everyone")
        assert result.fallback is True
        assert result.route.name == "general"

    def test_match_name(self, router):
        name = router.match_name("weather forecast")
        assert name == "weather"

    def test_no_routes(self, mock_vectorizer):
        router = SemanticRouter(mock_vectorizer)
        result = router.match("anything")
        assert result.fallback is True
        assert result.route.name == "general"

    def test_add_route_with_examples(self, mock_vectorizer):
        router = SemanticRouter(mock_vectorizer)
        route = router.add_route(
            "support",
            "customer support",
            examples=["I need help", "my order is missing"],
        )
        assert route.name == "support"
        assert route.embedding is not None

    def test_remove_route(self, router):
        assert router.remove_route("weather") is True
        assert router.remove_route("nonexistent") is False

    def test_find_similar_routes(self, router):
        """查找相似路由。"""
        similar = router.find_similar_routes("weather forecast", top_k=2)
        assert len(similar) == 2
        assert similar[0][0] == "weather"  # weather is most similar

    def test_routes_property(self, router):
        routes = router.routes
        assert "weather" in routes
        assert "math" in routes
        assert isinstance(routes["weather"], Route)

    def test_set_default(self, router):
        router.set_default("weather")
        result = router.match("goodbye everyone")
        assert result.route.name == "weather"
