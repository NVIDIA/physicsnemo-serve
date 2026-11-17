# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import subprocess

import pytest

import guard


@pytest.fixture
def repo(tmp_path):
    def git(*args):
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "lib.py").write_text("x = 1\n")
    (tmp_path / "tests" / "test_lib.py").write_text(
        "def test_a():\n    assert True\n\n\ndef test_b():\n    assert 1 == 1\n"
    )
    git("add", "-A")
    git("commit", "-qm", "base")

    def violations():
        git("add", "-A")
        return guard.check(*guard.staged_diff(str(tmp_path)))

    return tmp_path, violations


def test_small_source_change_is_allowed(repo):
    path, violations = repo
    (path / "src" / "lib.py").write_text("x = 2\n")
    assert violations() == []


def test_forbidden_paths(repo):
    path, violations = repo
    (path / ".github").mkdir()
    (path / ".github" / "ci.yml").write_text("on: push\n")
    (path / ".gitattributes").write_text("* text\n")
    (path / "Cargo.lock").write_text("x\n")
    found = " ".join(violations())
    assert ".github/ci.yml" in found
    assert ".gitattributes" in found
    assert "Cargo.lock" in found


def test_symlink_is_rejected(repo):
    path, violations = repo
    os.symlink("/etc/passwd", path / "src" / "link")
    assert any("symlinks" in v for v in violations())


def test_binary_file_is_rejected(repo):
    path, violations = repo
    (path / "src" / "blob.bin").write_bytes(b"\x00\x01\x02")
    assert any("binary" in v for v in violations())


def test_deleting_a_test_file_is_rejected(repo):
    path, violations = repo
    (path / "tests" / "test_lib.py").unlink()
    assert any("may not be deleted" in v for v in violations())


def test_removing_a_test_function_is_rejected(repo):
    path, violations = repo
    (path / "tests" / "test_lib.py").write_text("def test_a():\n    assert True\n")
    assert any("removes test functions" in v for v in violations())


@pytest.mark.parametrize(
    "marker", ["@pytest.mark.skip", "@pytest.mark.xfail", "#[ignore]"]
)
def test_added_skip_marker_is_rejected(repo, marker):
    path, violations = repo
    (path / "tests" / "test_lib.py").write_text(
        f"{marker}\ndef test_a():\n    assert True\n\n\ndef test_b():\n    assert 1 == 1\n"
    )
    assert any("skip/xfail/ignore" in v for v in violations())


def test_large_diff_is_rejected(repo):
    path, violations = repo
    (path / "src" / "lib.py").write_text("y = 0\n" * (guard.MAX_CHANGED_LINES + 1))
    assert any("limit" in v for v in violations())


def test_updating_a_test_is_allowed(repo):
    path, violations = repo
    (path / "tests" / "test_lib.py").write_text(
        "def test_a():\n    assert True\n\n\ndef test_b():\n    assert 1 == 2 - 1\n"
    )
    assert violations() == []
