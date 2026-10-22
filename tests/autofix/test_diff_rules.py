# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ci_autofix.diff_rules import (
    FileChange,
    is_golden_path,
    is_test_path,
    parse_diff,
    validate,
)
from ci_autofix.failures import Failure
from ci_autofix.verdict import (
    CODE_FIX,
    NEEDS_HUMAN,
    TEST_OUTDATED,
    Entry,
    Evidence,
    Verdict,
)

PY = Failure("Python Tests", "pytest", "tests/test_api.py::test_status")
RS = Failure("Rust Test", "cargo", "scheduler::tests::fanout")
FAILURES = {PY.key: PY, RS.key: RS}
PR_FILES = {"src/api.py", "crates/sched/src/lib.rs"}
EVIDENCE = [Evidence("src/api.py", "10-12", "status renamed to state")]


def _verdict(*entries):
    return Verdict("summary", list(entries))


def _check(changes, verdict, workspace):
    return validate(changes, verdict, FAILURES, PR_FILES, workspace)


def test_path_classification():
    assert is_test_path("tests/test_api.py")
    assert is_test_path("crates/x/tests/it.rs")
    assert is_test_path("pkg/foo_test.py")
    assert not is_test_path("src/api.py")
    assert is_golden_path("tests/golden/out.json")
    assert is_golden_path("qa/artifacts/ref.zarr")


def test_parse_diff():
    name_status = "M\tsrc/api.py\nA\tnew.py\nR100\told.py\tmoved.py\n"
    unified = (
        "diff --git a/src/api.py b/src/api.py\n--- a/src/api.py\n+++ b/src/api.py\n@@ -1 +1 @@\n-old\n+new\n"
        "diff --git a/new.py b/new.py\n--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+x = 1\n"
    )
    changes = {c.path: c for c in parse_diff(name_status, unified)}
    assert changes["src/api.py"].added == ["new"] and changes["src/api.py"].removed == [
        "old"
    ]
    assert changes["moved.py"].status == "R"


def test_code_fix_in_source_passes(tmp_path):
    result = _check(
        [FileChange("src/api.py", "M", ["a"], ["b"])],
        _verdict(Entry(PY.key, CODE_FIX, "r")),
        tmp_path,
    )
    assert result.ok and result.source_files == ["src/api.py"]


def test_code_fix_may_not_touch_tests(tmp_path):
    result = _check(
        [FileChange("tests/test_api.py", "M", ["a"], ["b"])],
        _verdict(Entry(PY.key, CODE_FIX, "r")),
        tmp_path,
    )
    assert any("without a test-outdated verdict" in v for v in result.violations)


def test_test_outdated_may_edit_only_the_failing_test_file(tmp_path):
    verdict = _verdict(Entry(PY.key, TEST_OUTDATED, "r", EVIDENCE))
    ok = _check(
        [
            FileChange(
                "tests/test_api.py",
                "M",
                ["assert r.state == 1"],
                ["assert r.status == 1"],
            )
        ],
        verdict,
        tmp_path,
    )
    assert ok.ok, ok.violations
    other = _check(
        [FileChange("tests/test_other.py", "M", ["x"], ["y"])], verdict, tmp_path
    )
    assert any("not the test file" in v for v in other.violations)


def test_rust_test_file_matched_by_function_name(tmp_path):
    (tmp_path / "crates/sched/tests").mkdir(parents=True)
    (tmp_path / "crates/sched/tests/sched.rs").write_text("#[test]\nfn fanout() {}\n")
    verdict = _verdict(
        Entry(
            RS.key, TEST_OUTDATED, "r", [Evidence("crates/sched/src/lib.rs", "1", "w")]
        )
    )
    result = _check(
        [
            FileChange(
                "crates/sched/tests/sched.rs",
                "M",
                ["assert_eq!(a, 2);"],
                ["assert_eq!(a, 1);"],
            )
        ],
        verdict,
        tmp_path,
    )
    assert result.ok, result.violations


def test_evidence_must_cite_pr_changes(tmp_path):
    verdict = _verdict(
        Entry(PY.key, TEST_OUTDATED, "r", [Evidence("src/unrelated.py", "1", "w")])
    )
    result = _check(
        [FileChange("tests/test_api.py", "M", ["assert 2"], ["assert 1"])],
        verdict,
        tmp_path,
    )
    assert any("does not change" in v for v in result.violations)


def test_weakening_tests_is_rejected(tmp_path):
    verdict = _verdict(Entry(PY.key, TEST_OUTDATED, "r", EVIDENCE))
    cases = {
        "adds a skip": FileChange(
            "tests/test_api.py", "M", ["@pytest.mark.skip(reason='x')"], []
        ),
        "reduces the number of assertions": FileChange(
            "tests/test_api.py", "M", [], ["assert r.ok"]
        ),
        "trivially true": FileChange(
            "tests/test_api.py", "M", ["assert True"], ["assert r.ok"]
        ),
        "removes test functions": FileChange(
            "tests/test_api.py", "M", [], ["def test_status():"]
        ),
        "may not be deleted": FileChange("tests/test_api.py", "D", [], []),
    }
    for message, change in cases.items():
        result = _check([change], verdict, tmp_path)
        assert any(message in v for v in result.violations), (
            message,
            result.violations,
        )


def test_forbidden_paths_and_size(tmp_path):
    verdict = _verdict(Entry(PY.key, CODE_FIX, "r"))
    assert not _check(
        [FileChange(".github/workflows/ci.yml", "M", ["x"], [])], verdict, tmp_path
    ).ok
    assert not _check([FileChange("uv.lock", "M", ["x"], [])], verdict, tmp_path).ok
    big = FileChange("src/api.py", "M", ["x"] * 401, [])
    assert any("limit 400" in v for v in _check([big], verdict, tmp_path).violations)


def test_golden_file_needs_a_changed_test_that_uses_it(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_api.py").write_text("load('status_ref.json')\n")
    verdict = _verdict(Entry(PY.key, TEST_OUTDATED, "r", EVIDENCE))
    golden = FileChange("tests/golden/status_ref.json", "M", ["{}"], ["[]"])
    test = FileChange("tests/test_api.py", "M", ["assert 2"], ["assert 1"])
    assert _check([golden, test], verdict, tmp_path).ok
    assert not _check([golden], verdict, tmp_path).ok


def test_no_fix_verdict_means_no_changes(tmp_path):
    verdict = _verdict(Entry(PY.key, NEEDS_HUMAN, "r"))
    assert _check([], verdict, tmp_path).ok
    assert not _check([FileChange("src/api.py", "M", ["x"], [])], verdict, tmp_path).ok
    assert not _check([], _verdict(Entry(PY.key, CODE_FIX, "r")), tmp_path).ok
