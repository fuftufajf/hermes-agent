"""Diagnostic exports must not expose structured fallback credentials."""

import pytest

from hermes_cli.dump import _config_overrides


@pytest.mark.parametrize("field,value", [
    ("api_key", 12345678901234567890),
    ("token", {"value": "fixture-token-not-valid"}),
    ("client_secret", "fixture***not-a-real-secret"),
    ("password", 98765432109876543210),
    ("clientApiKey", "fixture***not-an-api-key"),
])
def test_fallback_credentials_are_masked_by_field(field, value):
    cfg = {"fallback_providers": [{"provider": "custom", "model": "backup-model", "key_env": "BACKUP_KEY_ENV", "max_tokens": 4321, field: value}]}
    rendered = _config_overrides(cfg)["fallback_providers"]
    assert str(value) not in rendered
    assert "backup-model" in rendered
    assert "BACKUP_KEY_ENV" in rendered
    assert "4321" in rendered
    assert "***" in rendered


@pytest.mark.parametrize("url", [
    "https://fixture-opaque-credential:@backup.example/v1",
    "https://backup.example/v1?key=fixture-opaque-credential",
    "https://backup.example/v1?X-Amz-Signature=fixture-opaque-credential",
    "https://backup.example/v1?X-Goog-Signature=fixture-opaque-credential",
    "https://backup.example/v1?sig=fixture-opaque-credential",
    "https://backup.example/v1#access_token=fixture-opaque-credential",
])
def test_dump_never_exposes_fallback_url_credentials(url):
    cfg = {"fallback_providers": [{"provider": "custom", "model": "backup-model", "base_url": url}]}
    rendered = _config_overrides(cfg)["fallback_providers"]
    assert "fixture-opaque-credential" not in rendered
    assert "backup.example" in rendered


def test_external_dump_text_is_sanitized_before_sharing(monkeypatch):
    from hermes_cli import debug

    monkeypatch.setattr("hermes_cli.dump.run_dump", lambda _args: print("fallback https://fixture-opaque-credential:@backup.example/v1"))
    assert "fixture-opaque-credential" not in debug._capture_dump()


def test_provided_debug_dump_is_sanitized_at_final_boundary():
    from hermes_cli import debug

    snapshots = {name: debug.LogSnapshot(path=None, tail_text="ok", full_text=None) for name in debug._REPORT_LOGS}
    report = debug.collect_debug_report(dump_text="https://fixture-opaque-credential:@backup.example/v1", log_snapshots=snapshots)
    assert "fixture-opaque-credential" not in report
