# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ci_autofix.failures import Failure
from ci_autofix.jobs import (
    DEFAULT_JOBS_FILE,
    find_module_dir,
    load_jobs,
    rerun_commands,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_registry_jobs_exist_in_ci_workflow():
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/ci.yml").read_text())
    ci_job_names = {job["name"] for job in workflow["jobs"].values()}
    assert set(load_jobs(DEFAULT_JOBS_FILE)) <= ci_job_names


def test_pytest_rerun_quotes_ids():
    spec = load_jobs()["Python Tests"]
    failures = [Failure("Python Tests", "pytest", "tests/t.py::test_x[a b]")]
    [command] = rerun_commands(spec, failures, REPO_ROOT)
    assert (
        command.script
        == "uv run pytest -p no:cacheprovider -q 'tests/t.py::test_x[a b]'"
    )
    assert command.env["PYTHONPATH"].startswith("/workspace")


def test_unittest_rerun_groups_by_module_dir(tmp_path):
    (tmp_path / "a" / "tests").mkdir(parents=True)
    (tmp_path / "a" / "tests" / "test_cli.py").write_text("")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "test_other.py").write_text("")
    spec = load_jobs()["Model Builder and Native SDK CPU"]
    failures = [
        Failure(spec.name, "unittest", "test_cli.T.test_1"),
        Failure(spec.name, "unittest", "test_other.T.test_2"),
        Failure(spec.name, "unittest", "test_cli.T.test_3"),
    ]
    commands = rerun_commands(spec, failures, tmp_path)
    assert [(c.cwd, c.script) for c in commands] == [
        ("a/tests", "python3 -S -m unittest -v test_cli.T.test_1 test_cli.T.test_3"),
        ("b", "python3 -S -m unittest -v test_other.T.test_2"),
    ]


def test_find_module_dir_rejects_ambiguous_modules(tmp_path):
    for d in ("x", "y"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "test_dup.py").write_text("")
    with pytest.raises(LookupError):
        find_module_dir(tmp_path, "test_dup")
    with pytest.raises(LookupError):
        find_module_dir(tmp_path, "test_absent")


def test_job_level_rerun_has_no_ids():
    spec = load_jobs()["Rust Format"]
    [command] = rerun_commands(spec, [Failure(spec.name, "fmt")], REPO_ROOT)
    assert command.script == "cargo fmt --check"
    assert spec.autofix == "cargo fmt"
