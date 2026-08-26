"""Per-call backend override + search options + response metadata.

Covers the card-Etap-3 contract:
- ``backend`` applies to one call only, never mutates configuration
- unknown / capability-mismatched names error by name (no silent switch)
- options are forwarded only to providers that declare them
  (``supported_search_options``); the rest are dropped with a warning
- every response carries ``meta`` (requested_backend, served_by,
  result_count, elapsed_s)
- Tavily maps mode/topic/time_range/domain filters to native params
- keyless ring failover advances past a broken (non-throttle) vendor
- SSE parsing survives Latin-1-mojibake C1 control chars (Polish text)
"""

import asyncio
import json

import pytest

import tools.web_tools as web_tools
from agent import web_search_registry as reg
from agent.web_search_provider import WebSearchProvider
from plugins.web import keyless_mcp


class _FakeSearchProvider(WebSearchProvider):
    def __init__(self, name, options=()):
        self._name = name
        self._options = frozenset(options)
        self.calls = []

    @property
    def name(self):
        return self._name

    def is_available(self):
        return True

    def supports_search(self):
        return True

    def supports_extract(self):
        return False

    def supported_search_options(self):
        return self._options

    def search(self, query, limit=5, **kwargs):
        self.calls.append({"query": query, "limit": limit, **kwargs})
        return {
            "success": True,
            "data": {"web": [
                {"title": "t", "url": f"https://{self._name}.example",
                 "description": "", "position": 1},
            ]},
        }


class _FakeExtractProvider(WebSearchProvider):
    def __init__(self, name):
        self._name = name
        self.calls = []

    @property
    def name(self):
        return self._name

    def is_available(self):
        return True

    def supports_search(self):
        return False

    def supports_extract(self):
        return True

    def extract(self, urls, **kwargs):
        self.calls.append(list(urls))
        return [
            {"url": u, "title": "t", "content": "body", "raw_content": "body",
             "metadata": {"sourceURL": u}}
            for u in urls
        ]


@pytest.fixture
def fakes(monkeypatch):
    a = _FakeSearchProvider("fake-a", {"mode", "topic"})
    b = _FakeSearchProvider("fake-b")
    x = _FakeExtractProvider("fake-x")
    for p in (a, b, x):
        reg.register_provider(p)
    monkeypatch.setattr(
        web_tools, "_load_web_config",
        lambda: {"search_backend": "fake-b", "extract_backend": "fake-x",
                 "keyless_rescue": False},
    )
    yield a, b, x
    with reg._lock:
        for name in ("fake-a", "fake-b", "fake-x"):
            reg._providers.pop(name, None)


class TestPerCallSearchBackend:
    def test_override_wins_for_one_call_only(self, fakes):
        a, b, _ = fakes
        result = json.loads(web_tools.web_search_tool("q", backend="fake-a"))
        assert result["success"] is True
        assert result["meta"]["requested_backend"] == "fake-a"
        assert result["meta"]["served_by"] == "fake-a"
        assert len(a.calls) == 1 and not b.calls
        # Next call without backend: configured provider again — nothing
        # was persisted by the override.
        result2 = json.loads(web_tools.web_search_tool("q2"))
        assert result2["meta"]["served_by"] == "fake-b"
        assert result2["meta"]["requested_backend"] is None
        assert len(a.calls) == 1 and len(b.calls) == 1

    def test_unknown_backend_errors_by_name(self, fakes):
        result = json.loads(web_tools.web_search_tool("q", backend="nope"))
        assert result["success"] is False
        assert "'nope'" in result["error"]
        assert "Registered backends" in result["error"]

    def test_extract_only_backend_rejected(self, fakes):
        result = json.loads(web_tools.web_search_tool("q", backend="fake-x"))
        assert result["success"] is False
        assert "does not support search" in result["error"]

    def test_options_filtered_by_declared_support(self, fakes):
        a, _, _ = fakes
        result = json.loads(web_tools.web_search_tool(
            "q", backend="fake-a", mode="deep", topic="news", time_range="week",
        ))
        call = a.calls[0]
        assert call["mode"] == "deep" and call["topic"] == "news"
        assert "time_range" not in call
        assert any("time_range" in w for w in result["meta"]["warnings"])

    def test_meta_fields_present(self, fakes):
        result = json.loads(web_tools.web_search_tool("q"))
        meta = result["meta"]
        assert meta["result_count"] == 1
        assert isinstance(meta["elapsed_s"], (int, float))
        assert meta["served_by"] == "fake-b"


class TestPerCallExtractBackend:
    def _extract(self, coro):
        return json.loads(asyncio.run(coro))

    @pytest.fixture(autouse=True)
    def _safe_urls(self, monkeypatch):
        async def _ok(url):
            return True
        monkeypatch.setattr(web_tools, "async_is_safe_url", _ok)

    def test_override_dispatches_to_named_provider(self, fakes):
        _, _, x = fakes
        result = self._extract(web_tools.web_extract_tool(
            ["https://example.com/a"], backend="fake-x",
        ))
        assert result["results"][0]["content"] == "body"
        assert result["meta"]["requested_backend"] == "fake-x"
        assert result["meta"]["served_by"] == "fake-x"
        assert x.calls == [["https://example.com/a"]]

    def test_unknown_backend_errors_by_name(self, fakes):
        result = self._extract(web_tools.web_extract_tool(
            ["https://example.com/a"], backend="nope",
        ))
        assert result["success"] is False
        assert "'nope'" in result["error"]

    def test_search_only_backend_rejected(self, fakes):
        result = self._extract(web_tools.web_extract_tool(
            ["https://example.com/a"], backend="fake-a",
        ))
        assert result["success"] is False
        assert "search-only" in result["error"]


class TestTavilyOptionMapping:
    @pytest.fixture
    def keyed(self, monkeypatch):
        monkeypatch.setattr(
            "agent.web_search_provider.get_provider_env",
            lambda name: "tvly-x" if name == "TAVILY_API_KEY" else "",
        )
        captured = {}

        def _capture(endpoint, payload):
            captured["endpoint"] = endpoint
            captured["payload"] = payload
            return {"results": []}

        monkeypatch.setattr(
            "plugins.web.tavily.provider._tavily_request", _capture
        )
        return captured

    def test_native_param_mapping(self, keyed):
        from plugins.web.tavily.provider import TavilyWebSearchProvider

        result = TavilyWebSearchProvider().search(
            "q", 5, mode="deep", topic="news", time_range="week",
            include_domains=["gov.pl"], exclude_domains=["spam.example"],
        )
        payload = keyed["payload"]
        assert payload["search_depth"] == "advanced"
        assert payload["topic"] == "news"
        assert payload["time_range"] == "week"
        assert payload["include_domains"] == ["gov.pl"]
        assert payload["exclude_domains"] == ["spam.example"]
        assert result["success"] is True

    def test_invalid_values_dropped_with_warning(self, keyed):
        from plugins.web.tavily.provider import TavilyWebSearchProvider

        result = TavilyWebSearchProvider().search(
            "q", 5, mode="hyper", topic="gossip",
        )
        payload = keyed["payload"]
        assert "search_depth" not in payload and "topic" not in payload
        assert len(result["warnings"]) == 2


class TestRingFailover:
    def test_search_walks_past_broken_vendor(self, monkeypatch):
        monkeypatch.setattr(
            keyless_mcp, "_ring_order", lambda name: ["exa", "parallel"]
        )
        monkeypatch.setitem(
            keyless_mcp._KEYLESS_SEARCHERS, "exa",
            lambda q, n: {"success": False,
                          "error": "Unrecognized MCP response shape"},
        )
        monkeypatch.setitem(
            keyless_mcp._KEYLESS_SEARCHERS, "parallel",
            lambda q, n: {"success": True,
                          "data": {"web": [{"url": "https://p.example"}]}},
        )
        result = keyless_mcp.search_with_failover("exa", "q", 5)
        assert result["success"] is True
        assert result["data"]["served_by"] == "parallel"

    def test_extract_walks_past_whole_batch_failure(self, monkeypatch):
        monkeypatch.setattr(
            keyless_mcp, "_ring_order", lambda name: ["exa", "parallel"]
        )
        monkeypatch.setitem(
            keyless_mcp._KEYLESS_EXTRACTORS, "exa",
            lambda urls: [
                {"url": u, "title": "", "content": "",
                 "error": "Unrecognized MCP response shape"} for u in urls
            ],
        )
        monkeypatch.setitem(
            keyless_mcp._KEYLESS_EXTRACTORS, "parallel",
            lambda urls: [
                {"url": u, "title": "t", "content": "body",
                 "raw_content": "body"} for u in urls
            ],
        )
        results = keyless_mcp.extract_with_failover("exa", ["https://a.example"])
        assert results[0]["content"] == "body"


class TestMcpBodyParsing:
    def test_sse_line_with_c1_control_chars_parses(self):
        # "ą" (C4 85) mis-decoded as Latin-1 puts \x85 (NEL) inside the JSON
        # string; splitlines() would cut the SSE line there. split("\n")
        # must keep the frame whole.
        text_with_nel = "Urz\xc4\x85d"
        body = (
            "event: message\n"
            "data: " + json.dumps(
                {"result": {"content": [{"type": "text", "text": text_with_nel}]}}
            ) + "\n"
        )
        assert keyless_mcp._parse_mcp_body(body) == text_with_nel
