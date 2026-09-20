#!/usr/bin/env python3
"""Generic web_search / web_extract tools over pluggable backends.

Backend is selected during ``hermes tools`` (``web.backend`` in config.yaml; per
capability via ``web.search_backend`` / ``web.extract_backend``). Every vendor
implementation lives in ``plugins/web/<vendor>/provider.py`` and registers with
``agent.web_search_registry``; this module owns selection, safety gates,
caching, keyless rescue, and the truncate-and-store result pipeline.
Debug: ``WEB_TOOLS_DEBUG=true`` writes ``logs/web_tools_debug_<UUID>.json``.
"""

import json
import logging
import os
import time
from contextlib import contextmanager
from typing import List, Dict, Any, Optional
# Per-vendor client cache slots; plugins read/write these via tools.web_tools (tests reset them to None).
_firecrawl_client = _firecrawl_client_config = _parallel_client = _async_parallel_client = _exa_client = None

from plugins.web.firecrawl.provider import _is_tool_gateway_ready, check_firecrawl_api_key
from tools.debug_helpers import DebugSession
from tools.tool_backend_helpers import NOUS_MANAGED_PROVIDER, read_selection, selection_exists
from tools.url_safety import async_is_safe_url
from tools.web_tools_rescue import _rescue_eligible, _rescue_search
from tools.web_tools_truncate import _effective_char_limit, _trim_results, _truncate_results, convert_base64_images_to_links
from tools.web_tools_extract import (
    _extract_safe_urls, _merge_in_order, _no_provider_error, _resolve_extract_provider, _result_entry,
    _strict_selection_error, _validate_extract_urls,
)

logger = logging.getLogger(__name__)


# ─── Backend Selection ────────────────────────────────────────────────────────

def _env_value(name: str) -> str:
    """Resolve ``name`` via the config-aware env layer (``hermes config set`` values), then process env.

    Mirrors the SearXNG provider's ``_searxng_url()`` so that values set through Hermes' config/.env layer
    (``hermes config set``, ``hermes tools``) are honored here too — not just raw process-env exports.
    Without this, a config-only ``SEARXNG_URL`` (or any provider key) leaves the backend auto-detect cascade
    and ``check_web_api_key()`` blind to it. See #34290.
    """
    try:
        from hermes_cli.config import get_env_value
        val = get_env_value(name)
    except Exception:
        val = None
    return ((os.getenv(name, "") if val is None else val) or "").strip()


def _has_env(name: str) -> bool:
    return bool(_env_value(name))


def _load_web_config() -> dict:
    """Load the ``web:`` section from config.yaml; always a dict (a null section yields ``{}``)."""
    try:
        from hermes_cli.config import load_config
        return load_config().get("web") or {}
    except Exception:
        return {}


def _configured_backend(key: str = "backend") -> str:
    """Lower-cased, stripped ``web.<key>`` value ("" when unset/null)."""
    return (_load_web_config().get(key) or "").lower().strip()


def _registry_call(func_name: str, default, *args):
    """``agent.web_search_registry.<func_name>(*args)``, or *default* if it raised (registry never fatal)."""
    try:
        import agent.web_search_registry as registry_mod
        return getattr(registry_mod, func_name)(*args)
    except Exception as exc:  # noqa: BLE001 — registry optional; never fatal
        logger.debug("web provider registry %s%r failed: %s", func_name, args, exc)
        return default


def _registered_web_provider(backend: str):
    """Plugin-registered web provider by name, or ``None``."""
    return _registry_call("get_provider", None, backend) if backend else None


def _list_registered_web_providers():
    """All plugin-registered web providers (empty list on failure)."""
    return _registry_call("list_providers", [])


def _probe(provider, method: str, context: str = "") -> Optional[bool]:
    """``bool(provider.<method>())``, or ``None`` if it raised (a broken provider is unavailable; *context* is
    appended to the debug log line, e.g. " during readiness check")."""
    try:
        return bool(getattr(provider, method)())
    except Exception as exc:  # noqa: BLE001 — a broken provider is "unavailable"
        name = getattr(provider, "name", provider)
        logger.debug("web provider %r.%s() raised%s: %s", name, method, context, exc)
        return None


def _get_backend() -> str:
    """Shared web backend name. A stored ``web.backend`` is returned as-is — no availability probe, no
    fallback — so a broken selection surfaces the vendor's honest error rather than silently rerouting.
    The managed ``use_gateway`` selection also resolves to firecrawl with no ladder. Autodetect runs
    whenever no SHARED web selection was ever stored: per-capability keys (``web.search_backend``,
    ``web.extract_backend``) name only their own capability and never reroute the other (#113017)."""
    configured = _configured_backend()
    if configured:
        # "nous" (managed subscription) is serviced by firecrawl, routed through the managed Tool Gateway.
        return "firecrawl" if configured == NOUS_MANAGED_PROVIDER else configured
    if read_selection("web") is not None:
        # Shared selection exists (use_gateway) but no shared name: firecrawl, no ladder.
        return "firecrawl"

    # Never-configured install. Explicit user credentials beat the managed-gateway probe (a Nous OAuth
    # token's tier may not grant web access; the gateway then fails at runtime with no fallback).
    # Free tiers trail paid.
    backend_candidates = (
        ("tavily", _has_env("TAVILY_API_KEY")), ("perplexity", _has_env("PERPLEXITY_API_KEY")),
        ("exa", _has_env("EXA_API_KEY")),
        ("parallel", _has_env("PARALLEL_API_KEY")), ("keenable", _has_env("KEENABLE_API_KEY")),
        ("firecrawl", _has_env("FIRECRAWL_API_KEY") or _has_env("FIRECRAWL_API_URL")),
        ("firecrawl", _is_tool_gateway_ready()), ("searxng", _has_env("SEARXNG_URL")),
        ("brave-free", _has_env("BRAVE_SEARCH_API_KEY")), ("ddgs", _ddgs_package_importable()),
    )
    for backend, available in backend_candidates:
        if available:
            return backend

    # Plugin-contributed providers (built-ins are covered above); probe the held object directly.
    for provider in _list_registered_web_providers():
        if provider.name not in _LEGACY_WEB_BACKENDS and _probe(provider, "is_available"):
            return provider.name

    # Keyless free tier — strictly last so it never pre-empts a keyed backend. Discovery must run
    # first: reachable from contexts that haven't loaded plugins (subprocess runs, delegate children).
    try:
        _ensure_web_plugins_loaded()
        from agent.web_search_registry import _keyless_preference, _keyless_tier_enabled
        if _keyless_tier_enabled():
            for name in _keyless_preference():
                provider = _registered_web_provider(name)
                if provider is not None and _probe(provider, "is_keyless_available"):
                    return name
    except Exception as exc:  # noqa: BLE001 — registry optional; never fatal
        logger.debug("keyless fallback walk failed: %s", exc)

    return "firecrawl"  # default (backward compat)


def _get_search_backend() -> str:
    """Backend for web_search: ``web.search_backend`` (strict, no probe) > ``web.backend`` > autodetect."""
    return _configured_backend("search_backend") or _get_backend()


def _get_extract_backend() -> str:
    """Backend for web_extract: ``web.extract_backend`` (strict, no probe) > ``web.backend`` > autodetect."""
    return _configured_backend("extract_backend") or _get_backend()


def _ddgs_package_importable() -> bool:
    """ddgs is the only backend gated on package presence; single symbol so tests can patch it."""
    try:
        import ddgs  # noqa: F401
        return True
    except ImportError:
        return False


def _xai_available() -> bool:
    # Cheap probe only (env var OR auth.json OAuth): resolve_xai_http_credentials() may hit the network.
    try:
        from tools.xai_http import has_xai_credentials
        return has_xai_credentials()
    except Exception:
        return False


# Built-in backends -> cheap availability probes; any other name is a plugin provider resolved via the
# registry's ``is_available()``. Lambdas so test patches of module-level helpers (_ddgs_package_importable,
# check_firecrawl_api_key) are honored at call time. ``xai`` is probed via has_xai_credentials(), not a
# registered provider, though the registry's _LEGACY_PREFERENCE omits it — drop it if xai ever registers.
_BUILTIN_AVAILABILITY = {
    "exa": lambda: _has_env("EXA_API_KEY"),
    "parallel": lambda: _has_env("PARALLEL_API_KEY"),
    "keenable": lambda: _has_env("KEENABLE_API_KEY"),
    "firecrawl": lambda: check_firecrawl_api_key(),
    "tavily": lambda: _has_env("TAVILY_API_KEY")
    or any(_configured_backend(k) == "tavily" for k in ("backend", "search_backend", "extract_backend")),
    "perplexity": lambda: _has_env("PERPLEXITY_API_KEY"),
    "searxng": lambda: _has_env("SEARXNG_URL"),
    "brave-free": lambda: _has_env("BRAVE_SEARCH_API_KEY"),
    "ddgs": lambda: _ddgs_package_importable(),
    "xai": _xai_available,
}
_LEGACY_WEB_BACKENDS = frozenset(_BUILTIN_AVAILABILITY)


def _is_backend_available(backend: str) -> bool:
    """True when *backend* is usable — the single availability chokepoint. Non-legacy names delegate to the
    registered provider's ``is_available()`` (unregistered names fall through); built-ins use cheap probes.

    For plugin-registered backends (any name outside :data:`_LEGACY_WEB_BACKENDS`), availability is
    delegated to the provider's ``is_available()`` via the web_search_registry. This is the single
    chokepoint through which ``_get_backend``, ``_get_capability_backend``, and ``check_web_api_key`` all
    resolve availability — fixing custom-provider discovery for every caller at once (issues #28651, #31873,
    #32698). Built-in backends keep their cheap hardcoded probes below.
    """
    backend = (backend or "").lower().strip()
    provider = None if backend in _LEGACY_WEB_BACKENDS else _registered_web_provider(backend)
    if provider is not None:
        return _probe(provider, "is_available") or False
    probe = _BUILTIN_AVAILABILITY.get(backend)
    return probe() if probe else False


# ─── Firecrawl Client ──────────────────────────────────────────────────────── After PR #25182, the
# firecrawl client, lazy SDK proxy, dual-auth config resolution, response normalizers, and
# check_firecrawl_api_key() all live in plugins.web.firecrawl.provider.
def _web_requires_env() -> list[str]:
    """Tool-registry metadata env vars for the web backends. Gateway vars are always listed: gating them
    on ``managed_nous_tools_enabled()`` cost a synchronous portal HTTP refresh at every CLI startup.
    Contract: set var -> tool sees it; extras are harmless for the not-logged-in."""
    return [
        "EXA_API_KEY", "PARALLEL_API_KEY", "TAVILY_API_KEY", "PERPLEXITY_API_KEY", "KEENABLE_API_KEY", "FIRECRAWL_API_KEY",
        "FIRECRAWL_API_URL", "FIRECRAWL_GATEWAY_URL", "TOOL_GATEWAY_DOMAIN", "TOOL_GATEWAY_SCHEME",
        "TOOL_GATEWAY_USER_TOKEN",
    ]

_debug = DebugSession("web_tools", env_var="WEB_TOOLS_DEBUG")


# ─── Dispatch ─────────────────────────────────────────────────────────────────

# ─── Exa / Parallel inline helpers — moved into plugins ────────────────────── After PR #25182, the exa
# client + search/extract and parallel client + search/extract helpers all live in their respective plugins:
# - plugins/web/exa/provider.py - plugins/web/parallel/provider.py Both plugins register through
# agent.web_search_registry and the dispatchers in this file resolve them via get_active_*_provider().
def _ensure_web_plugins_loaded() -> None:
    """Idempotently run plugin discovery so the web registry is populated. Dispatch is reachable from contexts
    that never triggered discovery (subprocess agent runs, delegate children, scripts); without it a
    configured backend yields a misleading "No web ... provider" error.

    Every bundled web provider (brave-free, ddgs, searxng, exa, parallel, tavily, firecrawl, keenable)
    registers itself via ``plugins/web/<vendor>/__init__.py`` during plugin discovery. Tool dispatch can be
    reached from contexts that haven't already triggered discovery — subprocess agent runs, delegate
    children, standalone scripts, certain test paths — and without it the registry is empty and
    ``get_provider('firecrawl')`` returns ``None`` even when the user has ``web.extract_backend: firecrawl``
    configured and ``FIRECRAWL_API_KEY`` set. See #27580.
    """
    try:
        from hermes_cli.plugins import _ensure_plugins_discovered
        _ensure_plugins_discovered()
    except Exception as exc:  # noqa: BLE001
        # Warning, not debug: a broken plugin import is otherwise invisible.
        logger.warning("Web plugin discovery failed (non-fatal): %s", exc)


def _finish_debug(call_name: str, debug_call_data: dict, error_msg: Optional[str] = None) -> Optional[str]:
    """Log the call into the debug session; with *error_msg*, record it and return its ``tool_error`` envelope."""
    if error_msg is not None:
        logger.debug("%s", error_msg)
        debug_call_data["error"] = error_msg
    _debug.log_call(call_name, debug_call_data)
    _debug.save()
    return None if error_msg is None else tool_error(error_msg)


@contextmanager
def _explicit_vendor_scope(backend: str):
    """Pin one keyless-ring dispatch to an explicitly requested vendor."""
    token = context = None
    if backend:
        try:
            from plugins.web.keyless_mcp import _explicit_vendor as context
            token = context.set(backend)
        except Exception as exc:  # noqa: BLE001 — keyless ring is optional
            logger.debug("explicit vendor scope unavailable: %s", exc)
    try:
        yield
    finally:
        if context is not None and token is not None:
            context.reset(token)


def web_search_tool(
    query: str,
    limit: int = 5,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    topic: Optional[str] = None,
    time_range: Optional[str] = None,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
) -> str:
    """Search the web via the configured backend.

    ``backend`` overrides provider selection for this call only and never mutates
    persisted configuration.

    Returns a JSON string ``{"success": bool, "data": {"web": [{"title", "url", "description", "position"},
    ...]}}`` (metadata only — use web_extract_tool for page content) or ``{"success": false, "error": ...}``.
    """
    try:
        limit = min(max(int(limit), 1), 100)
    except (TypeError, ValueError):
        limit = 5
    requested_backend = (backend or "").lower().strip()
    requested_options = {
        key: value
        for key, value in (
            ("mode", mode),
            ("topic", topic),
            ("time_range", time_range),
            ("include_domains", include_domains),
            ("exclude_domains", exclude_domains),
        )
        if value
    }
    started = time.monotonic()
    warnings: List[str] = []
    debug_call_data = {
        "parameters": {
            "query": query,
            "limit": limit,
            "backend": requested_backend or None,
            "options": requested_options,
        },
        "error": None,
        "results_count": 0,
        "original_response_size": 0,
        "final_response_size": 0,
    }

    try:
        from tools.interrupt import is_interrupted
        if is_interrupted():
            return tool_error("Interrupted", success=False)
        # Sync only — every provider's search() is sync.
        _ensure_web_plugins_loaded()
        from agent.web_search_registry import get_active_search_provider, get_provider as _wsp_get_provider
        response_data = None
        if requested_backend:
            provider = _wsp_get_provider(requested_backend)
            if provider is None:
                names = ", ".join(sorted(p.name for p in _list_registered_web_providers())) or "none registered"
                response_data = {
                    "success": False,
                    "error": f"Unknown web search backend '{requested_backend}'. Registered backends: {names}.",
                }
            elif not provider.supports_search():
                response_data = {
                    "success": False,
                    "error": f"{provider.display_name} does not support search (extract-only backend).",
                }
        else:
            configured_backend = _get_search_backend()
            provider = _wsp_get_provider(configured_backend) if configured_backend else None
            if provider is None or not provider.supports_search():
                if provider is None and configured_backend and selection_exists("web"):
                    error_text = debug_call_data["error"] = _strict_selection_error("search", configured_backend)
                    _finish_debug("web_search_tool", debug_call_data)
                    return json.dumps({"success": False, "error": error_text}, indent=2, ensure_ascii=False)
                # Never-configured install: legacy availability-walked autodetect.
                provider = get_active_search_provider()

        if provider is None and response_data is None:
            fallback = "No web search provider configured. Run `hermes tools` to set one up."
            response_data = {"success": False, "error": _no_provider_error("search", fallback)}
        elif provider is not None and response_data is None:
            supported = frozenset(getattr(provider, "supported_search_options", lambda: ())())
            passed_options = {}
            for key, value in requested_options.items():
                if key in supported:
                    passed_options[key] = value
                else:
                    warnings.append(f"backend '{provider.name}' does not support option '{key}' (ignored)")
            logger.info("Web search via %s: '%s' (limit: %d)", provider.name, query, limit)
            response_data = _memoized_search(
                provider,
                query,
                limit,
                options=passed_options,
                explicit_backend=requested_backend,
            )

        response_data = dict(response_data)
        provider_warnings = response_data.pop("warnings", None)
        if provider_warnings:
            warnings.extend(str(item) for item in provider_warnings)
        data = response_data.get("data", {})
        meta = {
            "requested_backend": requested_backend or None,
            "served_by": data.get("served_by") or (provider.name if response_data.get("success") and provider else None),
            "mode": requested_options.get("mode"),
            "result_count": len(data.get("web", [])),
            "elapsed_s": round(time.monotonic() - started, 2),
        }
        if data.get("rescued_from"):
            meta["rescued_from"] = data["rescued_from"]
        if warnings:
            meta["warnings"] = warnings
        response_data["meta"] = meta

        debug_call_data["results_count"] = len(response_data.get("data", {}).get("web", []))
        result_json = json.dumps(response_data, indent=2, ensure_ascii=False)
        debug_call_data["final_response_size"] = len(result_json)
        _finish_debug("web_search_tool", debug_call_data)
        return result_json
    except Exception as e:
        return _finish_debug("web_search_tool", debug_call_data, f"Error searching web: {str(e)}")


def _memoized_search(
    provider,
    query: str,
    limit: int,
    options: Optional[Dict[str, Any]] = None,
    explicit_backend: str = "",
) -> dict:
    """TTL memo + single-flight around the paid vendor call (tools/web_result_cache.py); sits after every
    safety/config check. The provider is asked for the BUCKETED count so near-identical limits share an entry;
    the caller's count is sliced out. Only successful, non-rescued responses are cached — caching a rescue
    would make the one-shot ring fallback sticky for a whole TTL."""
    from tools.web_result_cache import bucket_limit, search_memo, slice_search_response

    options = options or {}
    memo_provider = provider.name
    if options:
        memo_provider += "|" + json.dumps(options, ensure_ascii=False, sort_keys=True)

    def _paid_search() -> tuple[dict, bool]:
        fetch_limit = bucket_limit(limit)
        with _explicit_vendor_scope(explicit_backend):
            try:
                resp = provider.search(query, fetch_limit, **options)
            except Exception as exc:  # noqa: BLE001 — candidate for rescue
                if not _rescue_eligible(provider):
                    raise
                return _rescue_search(provider.name, str(exc), query, fetch_limit), True
        if not resp.get("success") and _rescue_eligible(provider):
            return _rescue_search(provider.name, str(resp.get("error", "")), query, fetch_limit), True
        return resp, False

    response_data = search_memo.lookup(memo_provider, query, limit)
    if response_data is None:
        with search_memo.flight_lock(memo_provider, query, limit):
            # Re-check inside the lock: a concurrent identical call may have stored.
            response_data = search_memo.lookup(memo_provider, query, limit)
            if response_data is None:
                response_data, was_rescued = _paid_search()
                if not was_rescued:
                    search_memo.store(memo_provider, query, limit, response_data)
    return slice_search_response(response_data, limit)


async def web_extract_tool(
    urls: List[Any],
    format: str = None,
    char_limit: Optional[int] = None,
    backend: Optional[str] = None,
) -> str:
    """Extract clean page content (no LLM) from URLs via the configured backend.

    Pages over ``char_limit`` (default web.extract_char_limit or 15000) are head+tail truncated with a footer
    pointing at the stored full text; inline base64 images become ``[IMAGE: alt]``. URLs carrying secrets are
    refused before any fetch; private-network URLs are blocked per entry. Returns JSON ``{"results": [...]}``.
    """
    normalized_urls, normalized_indices, invalid_urls, blocked = _validate_extract_urls(urls)
    if blocked is not None:
        return blocked
    requested_backend = (backend or "").lower().strip()
    started = time.monotonic()
    debug_call_data = {
        "parameters": {
            "urls": normalized_urls,
            "format": format,
            "char_limit": char_limit,
            "backend": requested_backend or None,
        },
        "error": None,
        "pages_extracted": 0, "pages_truncated": 0, "original_response_size": 0, "final_response_size": 0,
        "truncation_metrics": [], "processing_applied": [],
    }

    try:
        logger.info("Extracting content from %d URL(s)", len(normalized_urls))
        # SSRF protection — filter private/internal URLs before any backend.
        safe_urls, safe_indices, ssrf_blocked = [], [], {}
        for index, url in zip(normalized_indices, normalized_urls):
            if await async_is_safe_url(url):
                safe_urls.append(url)
                safe_indices.append(index)
            else:
                ssrf_blocked[index] = _result_entry(
                    url, "Blocked: URL targets a private or internal network address"
                )

        results = []
        provider = None
        if safe_urls:
            _ensure_web_plugins_loaded()
            if requested_backend:
                provider = _registered_web_provider(requested_backend)
                if provider is None:
                    names = ", ".join(sorted(p.name for p in _list_registered_web_providers())) or "none registered"
                    return json.dumps(
                        {
                            "success": False,
                            "error": (
                                f"Unknown web extract backend '{requested_backend}'. "
                                f"Registered backends: {names}."
                            ),
                            "meta": {"requested_backend": requested_backend},
                        },
                        ensure_ascii=False,
                    )
                if not provider.supports_extract():
                    return json.dumps(
                        {
                            "success": False,
                            "error": (
                                f"{provider.display_name} is a search-only backend "
                                "and cannot extract URL content."
                            ),
                            "meta": {"requested_backend": requested_backend},
                        },
                        ensure_ascii=False,
                    )
                error_json = None
            else:
                provider, error_json = _resolve_extract_provider(_get_extract_backend())
            if error_json is not None:
                return error_json
            with _explicit_vendor_scope(requested_backend):
                results = await _extract_safe_urls(provider, safe_urls, format)
        # Reconstruct input order across invalid, blocked, and provider entries (providers preserve
        # the order of the safe URL list they receive).
        if invalid_urls or ssrf_blocked:
            fixed = {**ssrf_blocked, **invalid_urls}
            results = _merge_in_order(len(urls), fixed, safe_indices, safe_urls, results)

        logger.info("Extracted content from %d pages", len(results))
        debug_call_data["pages_extracted"] = len(results)
        debug_call_data["original_response_size"] = len(json.dumps({"results": results}))
        debug_call_data["processing_applied"].append("truncate_and_store")
        _truncate_results(results, _effective_char_limit(char_limit), debug_call_data)
        trimmed = _trim_results(results)
        result_json = (
            json.dumps(
                {
                    "results": trimmed,
                    "meta": {
                        "requested_backend": requested_backend or None,
                        "served_by": getattr(provider, "name", None),
                        "result_count": len(trimmed),
                        "elapsed_s": round(time.monotonic() - started, 2),
                    },
                },
                indent=2,
                ensure_ascii=False,
            )
            if trimmed
            else tool_error("Content was inaccessible or not found")
        )
        # Belt-and-suspenders sweep of the serialized JSON: a provider may tuck a base64 blob in metadata.
        cleaned_result = convert_base64_images_to_links(result_json)
        debug_call_data["final_response_size"] = len(cleaned_result)
        debug_call_data["processing_applied"].append("base64_image_conversion")
        _finish_debug("web_extract_tool", debug_call_data)
        return cleaned_result
    except Exception as e:
        return _finish_debug("web_extract_tool", debug_call_data, f"Error extracting content: {str(e)}")


def _provider_is_ready(provider) -> bool:
    """True when *provider* is keyed-available OR keyless-capable, without raising.

    ``get_active_*_provider()`` returns an explicitly configured backend even when ``is_available()`` is
    False (so dispatch can emit a precise error), so readiness gates (tool check_fn, ``hermes doctor``)
    must probe for real. Keyless mode (Exa/Parallel free tier) is a working state, not a misconfig.

    See #78412.
    """
    if provider is None:
        return False
    ready = _probe(provider, "is_available", " during readiness check")
    if ready is None:  # broken provider == not ready; don't try the keyless probe
        return False
    return bool(ready or _probe(provider, "is_keyless_available", " during readiness check"))


def check_web_api_key() -> bool:
    """``check_fn`` gate for web_search / web_extract: is any web backend available?

    A plugin-registered provider reporting ``is_available()`` must light the tools up even with no
    built-in credentials; resolution funnels through :func:`_is_backend_available`.

    See #28651, #31873.
    """
    # Boolean OR over configured + built-ins — probe order is irrelevant here.
    candidates = [c for c in (_configured_backend(),) if c] + list(_LEGACY_WEB_BACKENDS)
    if any(_is_backend_available(backend) for backend in candidates):
        return True
    # Plugin path. Discovery must run first: check_fn fires at tool-registration time, before any dispatch.
    try:
        _ensure_web_plugins_loaded()
        from agent.web_search_registry import get_active_search_provider, get_active_extract_provider
        return _provider_is_ready(get_active_search_provider()) or _provider_is_ready(
            get_active_extract_provider()
        )
    except Exception as exc:  # noqa: BLE001 — registry optional; never fatal
        logger.debug("web provider registry availability check failed: %s", exc)
        return False


# ─── Registry ─────────────────────────────────────────────────────────────────
from tools.registry import registry, tool_error

WEB_SEARCH_SCHEMA = {
    "name": "web_search",
    "description": "Search the web for information. Returns up to 5 results by default with titles, URLs, and descriptions. The query is passed through to the configured backend, so operators such as site:domain, filetype:pdf, intitle:word, -term, and \"exact phrase\" may work when the backend supports them.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up on the web. You may include backend-supported operators such as site:example.com, filetype:pdf, intitle:word, -term, or \"exact phrase\"."
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return. Defaults to 5.",
                "minimum": 1,
                "maximum": 100,
                "default": 5
            },
            "backend": {
                "type": "string",
                "description": "Per-call backend override. Applies only to this call and never changes configuration."
            },
            "mode": {
                "type": "string",
                "enum": ["fast", "deep"]
            },
            "topic": {
                "type": "string",
                "enum": ["general", "news", "finance"]
            },
            "time_range": {
                "type": "string",
                "enum": ["day", "week", "month", "year"]
            },
            "include_domains": {
                "type": "array",
                "items": {"type": "string"}
            },
            "exclude_domains": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["query"]
    }
}

WEB_EXTRACT_SCHEMA = {
    "name": "web_extract",
    "description": "Extract content from web page URLs. Returns clean page content in markdown/text (no LLM summarization — fast). Also works with PDF URLs (arxiv papers, documents) — pass the PDF link directly. Pages within the char budget (default 15000) return whole; larger pages return a head+tail window with a footer telling you the full text's saved file path and the read_file call to page through the omitted middle. Inline images appear as [IMAGE: alt] placeholders; real image URLs are kept as links. If a URL fails or times out, use the browser tool instead.",
    "parameters": {
        "type": "object",
        "properties": {
            "urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of URLs to extract content from (max 5 URLs per call)",
                "maxItems": 5
            },
            "char_limit": {
                "type": "integer",
                "description": "Optional per-page character budget sent back (default 15000). Pages larger than this are head+tail truncated with the full text stored to disk. Raise it when you need more of a long page inline.",
                "minimum": 2000
            },
            "backend": {
                "type": "string",
                "description": "Per-call extract backend override. Applies only to this call."
            }
        },
        "required": ["urls"]
    }
}

registry.register(
    name="web_search", toolset="web", schema=WEB_SEARCH_SCHEMA,
    handler=lambda args, **kw: web_search_tool(
        args.get("query", ""),
        limit=args.get("limit", 5),
        backend=args.get("backend"),
        mode=args.get("mode"),
        topic=args.get("topic"),
        time_range=args.get("time_range"),
        include_domains=args.get("include_domains"),
        exclude_domains=args.get("exclude_domains"),
    ),
    check_fn=check_web_api_key, requires_env=_web_requires_env(), emoji="🔍",
    max_result_size_chars=100_000,
)
registry.register(
    name="web_extract", toolset="web", schema=WEB_EXTRACT_SCHEMA,
    handler=lambda args, **kw: web_extract_tool(
        args.get("urls", [])[:5] if isinstance(args.get("urls"), list) else [], "markdown",
        char_limit=args.get("char_limit"),
        backend=args.get("backend"),
    ),
    check_fn=check_web_api_key, requires_env=_web_requires_env(), is_async=True, emoji="📄",
    max_result_size_chars=100_000,
)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Dict  # noqa: F401,E402
from typing import TYPE_CHECKING  # noqa: F401,E402
import asyncio  # noqa: F401,E402
import httpx  # noqa: F401,E402
import re  # noqa: F401,E402
import sys  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'DEFAULT_EXTRACT_CHAR_LIMIT': ('tools.web_tools_truncate', 'DEFAULT_EXTRACT_CHAR_LIMIT'),
    'Firecrawl': ('plugins.web.firecrawl.provider', 'Firecrawl'),
    'MAX_STORED_TEXT_CHARS': ('tools.web_tools_truncate', 'MAX_STORED_TEXT_CHARS'),
    'build_vendor_gateway_url': ('tools.managed_tool_gateway', 'build_vendor_gateway_url'),
    'managed_nous_tools_enabled': ('tools.tool_backend_helpers', 'managed_nous_tools_enabled'),
    'normalize_url_for_request': ('tools.url_safety', 'normalize_url_for_request'),
    'nous_tool_gateway_unavailable_message': ('tools.tool_backend_helpers', 'nous_tool_gateway_unavailable_message'),
    'prefers_gateway': ('tools.tool_backend_helpers', 'prefers_gateway'),
    'resolve_managed_tool_gateway': ('tools.managed_tool_gateway', 'resolve_managed_tool_gateway'),
    'sensitive_query_param_name': ('tools.url_safety', 'sensitive_query_param_name'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
