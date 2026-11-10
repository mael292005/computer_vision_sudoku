"""Tests de la préparation du site web local."""

import shutil
from pathlib import Path

import pytest

from sudoku_vision.webapp import REPO_ROOT, SHARED_FILES, WEB_DIR, prepare_site


def test_site_is_in_the_repository() -> None:
    assert (WEB_DIR / "index.html").is_file()
    assert (WEB_DIR / "js" / "vision.js").is_file()


def test_prepare_site_copies_model_and_examples(tmp_path: Path) -> None:
    web = tmp_path / "web"
    web.mkdir()
    shutil.copyfile(WEB_DIR / "index.html", web / "index.html")
    copied = prepare_site(web, REPO_ROOT)
    assert len(copied) == len(SHARED_FILES)
    assert (web / "assets" / "digits_cnn.onnx").stat().st_size > 0
    assert (web / "assets" / "examples" / "evil.png").is_file()


def test_prepare_site_needs_the_site(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        prepare_site(tmp_path, REPO_ROOT)


def test_shared_files_match_the_node_script() -> None:
    script = (WEB_DIR / "scripts" / "sync-assets.mjs").read_text(encoding="utf-8")
    for source, target in SHARED_FILES:
        assert source in script
        assert target in script
