"""Native per-call search options for bundled web providers."""

from __future__ import annotations

from types import SimpleNamespace

from plugins.web.parallel.provider import ParallelWebSearchProvider
from plugins.web.tavily.provider import TavilyWebSearchProvider


def test_tavily_maps_native_search_options(monkeypatch) -> None:
    captured = {}

    monkeypatch.setattr(
        "agent.web_search_provider.get_provider_env",
        lambda name: "tvly-test" if name == "TAVILY_API_KEY" else "",
    )
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)

    def _request(endpoint, payload, **kwargs):
        captured["endpoint"] = endpoint
        captured["payload"] = payload
        return {"results": []}

    monkeypatch.setattr("plugins.web.tavily.provider._tavily_request", _request)

    result = TavilyWebSearchProvider().search(
        "wetlands",
        5,
        mode="deep",
        topic="news",
        time_range="week",
        include_domains=["gov.pl"],
        exclude_domains=["spam.example"],
    )

    assert result["success"] is True
    assert captured["endpoint"] == "search"
    assert captured["payload"]["search_depth"] == "advanced"
    assert captured["payload"]["topic"] == "news"
    assert captured["payload"]["time_range"] == "week"
    assert captured["payload"]["include_domains"] == ["gov.pl"]
    assert captured["payload"]["exclude_domains"] == ["spam.example"]


def test_parallel_maps_deep_to_agentic(monkeypatch) -> None:
    captured = {}

    monkeypatch.setattr(
        "agent.web_search_provider.get_provider_env",
        lambda name: "parallel-test" if name == "PARALLEL_API_KEY" else "",
    )
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)

    def _search(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(results=[])

    client = SimpleNamespace(beta=SimpleNamespace(search=_search))
    monkeypatch.setattr("plugins.web.parallel.provider._get_sync_client", lambda: client)

    result = ParallelWebSearchProvider().search("wetlands", 5, mode="deep")

    assert result["success"] is True
    assert captured["mode"] == "agentic"
