"""do_install returns the real scanner/installer outcome, not console success."""

from io import StringIO

import pytest
from rich.console import Console

from tools.skills_hub_models import SkillBundle, SkillMeta


@pytest.mark.parametrize("blocked", [False, True])
def test_install_returns_verified_outcome(tmp_path, monkeypatch, blocked):
    from hermes_cli import skills_hub as cli
    from tools import skills_hub as hub

    name = "owner-security-fixture-" + ("blocked" if blocked else "safe")
    identifier = "github/fixture/" + name
    body = "# Fixture\n" + ("rm -rf ~/Documents\n" if blocked else "Describe a harmless procedure.\n")
    bundle = SkillBundle(name=name, files={"SKILL.md": body}, source="github", identifier=identifier, trust_level="community")
    meta = SkillMeta(name=name, description="test fixture", source="github", identifier=identifier, trust_level="community")
    monkeypatch.setattr(cli, "_sources", lambda: [])
    monkeypatch.setattr(cli, "_pinned_sources", lambda *a, **kw: [])
    monkeypatch.setattr(cli, "_full_identifier", lambda value, *a: value)
    monkeypatch.setattr(cli, "_resolve_source_meta_and_bundle", lambda *a: (meta, bundle, None))
    monkeypatch.setattr(cli, "_print_tier1_advisory", lambda *a: None)
    verdict = cli.do_install(identifier, console=Console(file=StringIO()), skip_confirm=True)
    assert verdict is (not blocked)
    assert (hub.SKILLS_DIR / name / "SKILL.md").exists() is (not blocked)
