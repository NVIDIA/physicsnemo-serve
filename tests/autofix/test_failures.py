# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ci_autofix.failures import (
    Failure,
    JobFailures,
    clean_log,
    excerpt,
    extract_ids,
    parse_job_log,
)

TS = "2026-10-21T10:00:00.1234567Z "


def _log(*lines: str) -> str:
    return "\n".join(TS + line for line in lines)


def test_clean_log_strips_timestamps_and_ansi():
    assert clean_log(_log("\x1b[31mFAILED\x1b[0m x")) == ["FAILED x"]


def test_extract_pytest_ids_from_short_summary():
    lines = [
        "FAILED tests/test_a.py::test_one - AssertionError: boom",
        "FAILED tests/test_a.py::TestB::test_two[param-1] - KeyError",
        "ERROR tests/test_c.py - ImportError: no module",
        "ERROR: not a pytest line",
        "FAILED tests/test_a.py::test_one - AssertionError: boom",
    ]
    assert extract_ids(lines, "pytest") == [
        "tests/test_a.py::test_one",
        "tests/test_a.py::TestB::test_two[param-1]",
        "tests/test_c.py",
    ]


def test_extract_unittest_ids_handles_both_qualifier_styles():
    lines = [
        "FAIL: test_x (test_cli.TestCli.test_x)",
        "ERROR: test_y (test_cli.TestCli)",
        "ERROR: test_missing (unittest.loader._FailedTest.test_missing)",
    ]
    assert extract_ids(lines, "unittest") == [
        "test_cli.TestCli.test_x",
        "test_cli.TestCli.test_y",
        "test_missing",
    ]


def test_extract_cargo_test_names():
    lines = [
        "test scheduler::tests::fanout ... FAILED",
        "test other ... ok",
        "test result: FAILED. 1 passed",
    ]
    assert extract_ids(lines, "cargo") == ["scheduler::tests::fanout"]


def test_parse_job_log_job_level_kind():
    result = parse_job_log("Rust Format", _log("Diff in src/lib.rs"), ["fmt"])
    assert result.failures == [Failure("Rust Format", "fmt")]
    assert result.failures[0].key == "Rust Format"


def test_parse_job_log_reports_compile_failure_only_without_test_failures():
    log = _log(
        "error[E0425]: cannot find value `x`",
        "error: could not compile `worker-runtime`",
    )
    result = parse_job_log("Rust Test", log, ["cargo", "cargo-build"])
    assert result.failures == [Failure("Rust Test", "cargo-build")]

    log = _log("test a::b ... FAILED", "error: could not compile `x`")
    result = parse_job_log("Rust Test", log, ["cargo", "cargo-build"])
    assert result.failures == [Failure("Rust Test", "cargo", "a::b")]


def test_parse_job_log_unrecognised_failure_is_empty():
    result = parse_job_log(
        "Model Builder and Native SDK CPU",
        _log("The following tests FAILED:"),
        ["unittest"],
    )
    assert not result.parsed


def test_excerpt_starts_near_first_failure_and_is_bounded():
    lines = (
        [f"setup {i}" for i in range(100)]
        + ["=== FAILURES ==="]
        + [f"line {i}" for i in range(1000)]
    )
    text = excerpt(lines)
    assert text.startswith("setup 80")
    assert "lines elided" in text
    assert text.endswith("line 999")


def test_job_failures_round_trip():
    job = JobFailures(
        "Python Tests", [Failure("Python Tests", "pytest", "tests/t.py::x")], "log"
    )
    assert JobFailures.from_dict(job.to_dict()) == job
