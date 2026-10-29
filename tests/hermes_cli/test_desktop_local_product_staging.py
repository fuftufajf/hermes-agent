"""Regression for Nyx-Receiver#2: local macOS products keep signing and swaps off File Provider."""

import pytest

from hermes_cli import main_desktop


def test_default_layout_still_stages_next_to_release(tmp_path):
    desktop = tmp_path / "apps/desktop"
    desktop.mkdir(parents=True)
    stale = desktop / ".staging-old"
    stale.mkdir()
    stage = main_desktop._desktop_staging_dir(desktop)
    assert stage.parent == desktop
    assert not stale.exists()


@pytest.mark.platforms("macos")
def test_mac_release_link_stages_locally_and_preserves_products(tmp_path):
    desktop = tmp_path / "source/apps/desktop"
    desktop.mkdir(parents=True)
    release = tmp_path / "local-products/release"
    release.mkdir(parents=True)
    (desktop / "release").symlink_to(release, target_is_directory=True)
    marker = release / "existing-app"
    marker.write_text("previous app")
    unrelated = release.parent / ".staging-other-install"
    unrelated.mkdir()
    stage = main_desktop._desktop_staging_dir(desktop)
    assert stage.parent == release.parent.resolve()
    stage.mkdir()
    main_desktop._desktop_staging_dir(desktop)
    assert not stage.exists()
    assert marker.read_text() == "previous app"
    assert unrelated.exists()


@pytest.mark.platforms("macos")
def test_mac_installs_do_not_clean_each_others_staging(tmp_path):
    products = tmp_path / "products"
    products.mkdir()
    stages = []
    for name in ("first", "second"):
        desktop = tmp_path / name
        desktop.mkdir()
        (desktop / "release").symlink_to(products / name, target_is_directory=True)
        stage = main_desktop._desktop_staging_dir(desktop)
        stage.mkdir()
        stages.append(stage)
    assert stages[0] != stages[1]
    assert all(stage.exists() for stage in stages)


@pytest.mark.platforms("macos")
def test_mac_swap_promotes_through_release_link(tmp_path, monkeypatch):
    desktop = tmp_path / "source/apps/desktop"
    desktop.mkdir(parents=True)
    release = tmp_path / "products/release"
    release.mkdir(parents=True)
    (desktop / "release").symlink_to(release, target_is_directory=True)
    exe_rel = "mac-arm64/Hermes.app/Contents/MacOS/Hermes"
    live = release / exe_rel
    live.parent.mkdir(parents=True)
    live.write_text("old")
    stage = main_desktop._desktop_staging_dir(desktop)
    staged = stage / exe_rel
    staged.parent.mkdir(parents=True)
    staged.write_text("new")
    monkeypatch.setattr(main_desktop, "_stop_desktop_processes_locking_build", lambda *a, **kw: [])
    promoted = main_desktop._swap_staged_desktop_app(desktop, stage)
    assert promoted == desktop / "release" / exe_rel
    assert promoted.read_text() == "new"
    assert not stage.exists()
    assert (desktop / "release").is_symlink()
