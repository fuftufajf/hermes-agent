"""Security backport regressions: scan text only; never execute its shell snippets."""

import pytest

from tools.skills_guard import scan_file


@pytest.mark.parametrize("command", ["rm -rf ~/Documents", "rm -rf ~", "rm -rf ~/*", "rmdir ~/Documents"])
def test_tilde_home_delete_is_critical(tmp_path, command):
    source = tmp_path / "fixture.md"
    source.write_text(command + "\n", encoding="utf-8")
    findings = scan_file(source, "fixture.md")
    assert any(item.pattern_id == "destructive_home_rm" and item.severity == "critical" for item in findings)


def test_inline_shell_autoexecution_is_detected(tmp_path):
    source = tmp_path / "fixture.md"
    source.write_text("run `date`\nDate: !`date`\nempty: !` `\n", encoding="utf-8")
    hits = [item for item in scan_file(source, "fixture.md") if item.pattern_id == "inline_shell_exec"]
    assert [item.line for item in hits] == [2]


def test_new_scanner_invalidates_previous_safe_cache(tmp_path, monkeypatch):
    from tools import skills_guard

    skill = tmp_path / "fixture"
    skill.mkdir()
    (skill / "SKILL.md").write_text("Date: !`date`\n", encoding="utf-8")
    old_patterns = [row for row in skills_guard._COMPILED_THREAT_PATTERNS if row[1] != "inline_shell_exec"]
    with monkeypatch.context() as previous:
        previous.setattr(skills_guard, "SCANNER_VERSION", "skills-guard-v6")
        previous.setattr(skills_guard, "_COMPILED_THREAT_PATTERNS", old_patterns)
        _, before = skills_guard.scan_skill_cached(skill, cache_dir=tmp_path / "cache")
        assert "inline_shell_exec" not in before["rules"]
    _, after = skills_guard.scan_skill_cached(skill, cache_dir=tmp_path / "cache")
    assert after["fresh"] is True
    assert "inline_shell_exec" in after["rules"]
