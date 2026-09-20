"""Per-call web backend and native search-option contracts."""

from __future__ import annotations

import asyncio
import json

import tools.web_tools as web_tools
from agent import web_search_registry
from plugins.web import keyless_mcp
from plugins.web.exa.provider import ExaWebSearchProvider
from tools.registry import registry


def test_tool_schemas_expose_per_call_controls() -> None:
    search_properties = registry.get_schema("web_search")["parameters"]["properties"]
    extract_properties = registry.get_schema("web_extract")["parameters"]["properties"]

    assert {
        "backend",
        "mode",
        "topic",
        "time_range",
        "include_domains",
        "exclude_domains",
    } <= set(search_properties)
    assert "backend" in extract_properties


def test_registered_search_handler_forwards_per_call_controls(monkeypatch) -> None:
    provider = _SearchProvider()
    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(web_tools, "_get_search_backend", lambda: (_ for _ in ()).throw(AssertionError("config read")))
    monkeypatch.setattr(web_search_registry, "get_provider", lambda name: provider if name == provider.name else None)
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)
    monkeypatch.setattr(web_tools._debug, "log_call", lambda *args, **kwargs: None)
    monkeypatch.setattr(web_tools._debug, "save", lambda: None)

    result = json.loads(
        registry.dispatch(
            "web_search",
            {"query": "registry controls", "backend": provider.name, "mode": "deep"},
        )
    )

    assert result["success"] is True
    assert provider.calls[0][2] == {"mode": "deep"}


def test_registered_extract_handler_forwards_backend(monkeypatch) -> None:
    class _ExtractProvider:
        name = "registered-extract"
        display_name = "Registered Extract"

        def supports_extract(self) -> bool:
            return True

    provider = _ExtractProvider()
    seen = []

    async def _safe_url(url: str) -> bool:
        return True

    async def _extract(selected, urls, format):
        seen.append(selected)
        return [
            {
                "url": urls[0],
                "title": "Page",
                "content": "body",
                "raw_content": "body",
                "metadata": {"sourceURL": urls[0]},
            }
        ]

    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(web_tools, "_registered_web_provider", lambda name: provider if name == provider.name else None)
    monkeypatch.setattr(web_tools, "_get_extract_backend", lambda: (_ for _ in ()).throw(AssertionError("config read")))
    monkeypatch.setattr(web_tools, "async_is_safe_url", _safe_url)
    monkeypatch.setattr(web_tools, "_extract_safe_urls", _extract)
    monkeypatch.setattr(web_tools, "_truncate_results", lambda *args, **kwargs: None)
    monkeypatch.setattr(web_tools._debug, "log_call", lambda *args, **kwargs: None)
    monkeypatch.setattr(web_tools._debug, "save", lambda: None)

    result = json.loads(
        registry.dispatch(
            "web_extract",
            {"urls": ["https://example.com"], "backend": provider.name},
        )
    )

    assert seen == [provider]
    assert result["meta"]["served_by"] == provider.name


class _SearchProvider:
    name = "fake-search"
    display_name = "Fake Search"

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, dict]] = []

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def supported_search_options(self):
        return frozenset({"mode", "topic"})

    def search(self, query: str, limit: int = 5, **kwargs) -> dict:
        self.calls.append((query, limit, kwargs))
        return {
            "success": True,
            "data": {
                "web": [
                    {
                        "title": "Result",
                        "url": "https://example.com",
                        "description": "body",
                        "position": 1,
                    }
                ]
            },
        }


def test_per_call_search_backend_bypasses_config(monkeypatch) -> None:
    provider = _SearchProvider()
    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(web_tools, "_get_search_backend", lambda: (_ for _ in ()).throw(AssertionError("config read")))
    monkeypatch.setattr(web_search_registry, "get_provider", lambda name: provider if name == provider.name else None)
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)
    monkeypatch.setattr(web_tools._debug, "log_call", lambda *args, **kwargs: None)
    monkeypatch.setattr(web_tools._debug, "save", lambda: None)

    result = json.loads(web_tools.web_search_tool("wetlands", backend="fake-search"))

    assert result["success"] is True
    assert provider.calls and provider.calls[0][0] == "wetlands"


def test_search_options_are_capability_filtered_and_reported(monkeypatch) -> None:
    provider = _SearchProvider()
    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(web_search_registry, "get_provider", lambda name: provider if name == provider.name else None)
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)
    monkeypatch.setattr(web_tools._debug, "log_call", lambda *args, **kwargs: None)
    monkeypatch.setattr(web_tools._debug, "save", lambda: None)

    result = json.loads(
        web_tools.web_search_tool(
            "wetlands options",
            backend="fake-search",
            mode="deep",
            topic="news",
            time_range="week",
        )
    )

    assert provider.calls[0][2] == {"mode": "deep", "topic": "news"}
    assert result["meta"]["requested_backend"] == "fake-search"
    assert result["meta"]["served_by"] == "fake-search"
    assert result["meta"]["mode"] == "deep"
    assert result["meta"]["result_count"] == 1
    assert any("time_range" in warning for warning in result["meta"]["warnings"])


def test_per_call_extract_backend_dispatches_to_named_provider(monkeypatch) -> None:
    class _ExtractProvider:
        name = "fake-extract"
        display_name = "Fake Extract"

        def supports_search(self) -> bool:
            return False

        def supports_extract(self) -> bool:
            return True

    provider = _ExtractProvider()
    seen = []

    async def _safe_url(url: str) -> bool:
        return True

    async def _extract(selected, urls, format):
        seen.append((selected, urls, format))
        return [
            {
                "url": urls[0],
                "title": "Page",
                "content": "body",
                "raw_content": "body",
                "metadata": {"sourceURL": urls[0]},
            }
        ]

    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(web_tools, "_registered_web_provider", lambda name: provider if name == provider.name else None)
    monkeypatch.setattr(web_tools, "_get_extract_backend", lambda: (_ for _ in ()).throw(AssertionError("config read")))
    monkeypatch.setattr(web_tools, "async_is_safe_url", _safe_url)
    monkeypatch.setattr(web_tools, "_extract_safe_urls", _extract)
    monkeypatch.setattr(web_tools, "_truncate_results", lambda *args, **kwargs: None)
    monkeypatch.setattr(web_tools._debug, "log_call", lambda *args, **kwargs: None)
    monkeypatch.setattr(web_tools._debug, "save", lambda: None)

    result = json.loads(
        asyncio.run(web_tools.web_extract_tool(["https://example.com"], backend="fake-extract"))
    )

    assert seen == [(provider, ["https://example.com"], None)]
    assert result["results"][0]["content"] == "body"
    assert result["meta"]["requested_backend"] == "fake-extract"
    assert result["meta"]["served_by"] == "fake-extract"


def test_explicit_keyless_backend_starts_with_requested_vendor(monkeypatch) -> None:
    provider = ExaWebSearchProvider()
    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(web_search_registry, "get_provider", lambda name: provider if name == "exa" else None)
    monkeypatch.setattr("agent.web_search_provider.get_provider_env", lambda name: "")
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)
    monkeypatch.setattr(keyless_mcp, "provider_tier", lambda name: "auto")
    monkeypatch.setattr(keyless_mcp, "_web_config_selects", lambda name: False)
    monkeypatch.setattr(keyless_mcp, "_ring_cursor", 1)
    monkeypatch.setitem(
        keyless_mcp._KEYLESS_SEARCHERS,
        "exa",
        lambda query, limit: {
            "success": True,
            "data": {"web": [{"url": "https://exa.example", "title": "Exa", "description": "", "position": 1}]},
        },
    )
    monkeypatch.setitem(
        keyless_mcp._KEYLESS_SEARCHERS,
        "parallel",
        lambda query, limit: {
            "success": True,
            "data": {"web": [{"url": "https://parallel.example", "title": "Parallel", "description": "", "position": 1}]},
        },
    )
    monkeypatch.setattr(web_tools._debug, "log_call", lambda *args, **kwargs: None)
    monkeypatch.setattr(web_tools._debug, "save", lambda: None)

    result = json.loads(web_tools.web_search_tool("explicit exa vendor", backend="exa"))

    assert result["data"]["web"][0]["url"] == "https://exa.example"
    assert result["meta"]["served_by"] == "exa"


def test_explicit_keyless_extract_starts_with_requested_vendor(monkeypatch) -> None:
    provider = ExaWebSearchProvider()

    async def _safe_url(url: str) -> bool:
        return True

    def _document(url: str, source: str) -> dict:
        return {
            "url": url,
            "title": source,
            "content": source,
            "raw_content": source,
            "metadata": {"sourceURL": url},
        }

    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(web_tools, "_registered_web_provider", lambda name: provider if name == "exa" else None)
    monkeypatch.setattr("agent.web_search_provider.get_provider_env", lambda name: "")
    monkeypatch.setattr(web_tools, "async_is_safe_url", _safe_url)
    monkeypatch.setattr("tools.website_policy.check_website_access", lambda url: None)
    monkeypatch.setattr("tools.web_result_cache.extract_cache_get", lambda *args, **kwargs: None)
    monkeypatch.setattr("tools.web_result_cache.extract_cache_put", lambda *args, **kwargs: None)
    monkeypatch.setattr(keyless_mcp, "provider_tier", lambda name: "auto")
    monkeypatch.setattr(keyless_mcp, "_web_config_selects", lambda name: False)
    monkeypatch.setattr(keyless_mcp, "_ring_cursor", 1)
    monkeypatch.setitem(
        keyless_mcp._KEYLESS_EXTRACTORS,
        "exa",
        lambda urls: [_document(url, "exa") for url in urls],
    )
    monkeypatch.setitem(
        keyless_mcp._KEYLESS_EXTRACTORS,
        "parallel",
        lambda urls: [_document(url, "parallel") for url in urls],
    )
    monkeypatch.setattr(web_tools, "_truncate_results", lambda *args, **kwargs: None)
    monkeypatch.setattr(web_tools._debug, "log_call", lambda *args, **kwargs: None)
    monkeypatch.setattr(web_tools._debug, "save", lambda: None)

    result = json.loads(
        asyncio.run(
            web_tools.web_extract_tool(
                ["https://extract-vendor.example"],
                backend="exa",
            )
        )
    )

    assert result["results"][0]["content"] == "exa"
    assert result["meta"]["served_by"] == "exa"


def test_keyless_ring_fails_over_after_vendor_error(monkeypatch) -> None:
    monkeypatch.setattr(keyless_mcp, "_ring_order", lambda name: ["exa", "parallel"])
    monkeypatch.setitem(
        keyless_mcp._KEYLESS_SEARCHERS,
        "exa",
        lambda query, limit: {"success": False, "error": "Unrecognized MCP response shape"},
    )
    monkeypatch.setitem(
        keyless_mcp._KEYLESS_SEARCHERS,
        "parallel",
        lambda query, limit: {
            "success": True,
            "data": {"web": [{"url": "https://parallel.example"}]},
        },
    )

    result = keyless_mcp.search_with_failover("exa", "wetlands", 5)

    assert result["success"] is True
    assert result["data"]["served_by"] == "parallel"
