# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from ci_autofix.failures import Failure, JobFailures
from ci_autofix.jobs import load_jobs
from ci_autofix.sandbox import Result
from ci_autofix.triage import (
    AUTOFIXED,
    FLAKY,
    PRE_EXISTING,
    REPRODUCIBLE,
    UNRUNNABLE,
    triage,
)

HEAD = Path("/head")
BASE = Path("/base")
JOBS = load_jobs()


class FakeSandbox:
    """Answers each run from a {(checkout, script substring): Result} table."""

    def __init__(self, table):
        self.table = table
        self.calls = []

    def run(self, command, checkout, timeout=0):
        self.calls.append((checkout, command.script))
        for (where, needle), result in self.table.items():
            if where == checkout and needle in command.script:
                return result
        return Result(0, "")


def _pytest_job(*ids):
    return JobFailures(
        "Python Tests", [Failure("Python Tests", "pytest", i) for i in ids]
    )


def _status(classified):
    return {c.failure.test_id or c.failure.job: c.status for c in classified}


def test_flaky_pre_existing_and_reproducible():
    sandbox = FakeSandbox(
        {
            (HEAD, "pytest"): Result(1, "FAILED t.py::pre - x\nFAILED t.py::real - y"),
            (BASE, "pytest"): Result(1, "FAILED t.py::pre - x"),
        }
    )
    result = triage(
        sandbox,
        JOBS,
        [_pytest_job("t.py::flaky", "t.py::pre", "t.py::real")],
        HEAD,
        BASE,
    )
    assert _status(result) == {
        "t.py::flaky": FLAKY,
        "t.py::pre": PRE_EXISTING,
        "t.py::real": REPRODUCIBLE,
    }
    base_calls = [script for checkout, script in sandbox.calls if checkout == BASE]
    assert base_calls == ["uv run pytest -p no:cacheprovider -q t.py::pre t.py::real"]


def test_pytest_missing_on_base_means_added_by_pr():
    sandbox = FakeSandbox(
        {
            (HEAD, "pytest"): Result(1, "FAILED t.py::new - x"),
            (BASE, "pytest"): Result(4, ""),
        }
    )
    assert _status(triage(sandbox, JOBS, [_pytest_job("t.py::new")], HEAD, BASE)) == {
        "t.py::new": REPRODUCIBLE
    }


def test_crash_without_ids_counts_every_test_as_failing():
    sandbox = FakeSandbox(
        {
            (HEAD, "pytest"): Result(2, "Segmentation fault"),
            (BASE, "pytest"): Result(0, ""),
        }
    )
    assert set(
        _status(triage(sandbox, JOBS, [_pytest_job("a", "b")], HEAD, BASE)).values()
    ) == {REPRODUCIBLE}


def test_unlocatable_unittest_module_is_unrunnable(tmp_path):
    job = JobFailures(
        "Model Builder and Native SDK CPU",
        [Failure("Model Builder and Native SDK CPU", "unittest", "test_gone.T.t")],
    )
    result = triage(FakeSandbox({}), JOBS, [job], tmp_path, tmp_path)
    assert _status(result) == {"test_gone.T.t": UNRUNNABLE}


def test_fmt_failure_is_autofixed():
    state = {"fixed": False}

    class FmtSandbox(FakeSandbox):
        def run(self, command, checkout, timeout=0):
            self.calls.append((checkout, command.script))
            if command.script == "cargo fmt":
                state["fixed"] = True
                return Result(0, "")
            if checkout == HEAD and not state["fixed"]:
                return Result(1, "Diff in src/lib.rs")
            return Result(0, "")

    sandbox = FmtSandbox({})
    job = JobFailures("Rust Format", [Failure("Rust Format", "fmt")])
    assert _status(triage(sandbox, JOBS, [job], HEAD, BASE)) == {
        "Rust Format": AUTOFIXED
    }
