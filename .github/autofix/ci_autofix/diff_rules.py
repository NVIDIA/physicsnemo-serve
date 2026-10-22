# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic checks on the agent's diff. These cannot be overridden by the agent."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from ci_autofix.failures import Failure
from ci_autofix.verdict import CODE_FIX, TEST_OUTDATED, Verdict

MAX_CHANGED_LINES = 400
MAX_GOLDEN_FILES = 5

FORBIDDEN_PREFIXES = (".github/",)
FORBIDDEN_FILES = frozenset({"Cargo.lock", "uv.lock"})
_TEST_DIRS = frozenset({"tests", "test"})
_GOLDEN_DIRS = frozenset({"golden", "goldens", "fixtures", "testdata", "snapshots"})
_GOLDEN_SUFFIXES = (".golden", ".snap")

_SKIP_MARKERS = re.compile(
    r"@pytest\.mark\.(skip|skipif|xfail)\b|\bpytest\.(skip|xfail)\(|@unittest\.(skip|skipIf|skipUnless|expectedFailure)\b"
    r"|\bself\.skipTest\(|#\[ignore\b"
)
_TEST_DEFS = re.compile(r"^\s*(async\s+)?def test\w*\(|^\s*#\[(tokio::)?test\b")
_ASSERTS = re.compile(
    r"\bassert\b|\bassert(_eq|_ne)?!\(|\bself\.assert\w+\(|\bpytest\.raises\(|\bshould_panic\b"
)
_TRIVIAL_ASSERTS = re.compile(
    r"^\s*assert\s+(True|1|not\s+(False|0|None))\s*(#.*)?$|\bassert!\(\s*true\s*\)|\bself\.assertTrue\(\s*True\s*\)"
)


@dataclass
class FileChange:
    path: str
    status: str  # A, M, D, or R (renames use the new path)
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    binary: bool = False

    @property
    def changed_lines(self) -> int:
        return len(self.added) + len(self.removed)


@dataclass
class Validation:
    violations: list[str]
    source_files: list[str]
    test_files: list[str]

    @property
    def ok(self) -> bool:
        return not self.violations


def is_test_path(path: str) -> bool:
    p = PurePosixPath(path)
    return (
        bool(_TEST_DIRS & set(p.parts[:-1]))
        or p.name.startswith("test_")
        or p.stem.endswith("_test")
        or p.stem.endswith("_tests")
        or p.name == "conftest.py"
    )


def is_golden_path(path: str) -> bool:
    p = PurePosixPath(path)
    return (
        bool(_GOLDEN_DIRS & set(p.parts[:-1]))
        or p.suffix in _GOLDEN_SUFFIXES
        or path.startswith("qa/artifacts/")
    )


def is_forbidden_path(path: str) -> bool:
    return (
        path.startswith(FORBIDDEN_PREFIXES)
        or PurePosixPath(path).name in FORBIDDEN_FILES
    )


def parse_diff(name_status: str, unified: str) -> list[FileChange]:
    """Build FileChanges from ``git diff --name-status -M`` and ``git diff -U0`` output."""
    changes: dict[str, FileChange] = {}
    for line in name_status.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0][0]
        path = parts[-1]
        changes[path] = FileChange(path=path, status=status)

    current: FileChange | None = None
    for line in unified.splitlines():
        if line.startswith("diff --git "):
            path = line.split(" b/", 1)[1] if " b/" in line else ""
            current = changes.get(path)
        elif current is None:
            continue
        elif line.startswith("Binary files "):
            current.binary = True
        elif line.startswith("+++") or line.startswith("---"):
            continue
        elif line.startswith("+"):
            current.added.append(line[1:])
        elif line.startswith("-"):
            current.removed.append(line[1:])
    return list(changes.values())


def _count(pattern: re.Pattern, lines: list[str]) -> int:
    return sum(1 for line in lines if pattern.search(line))


def allowed_test_file(path: str, failure: Failure, workspace: Path) -> bool:
    """Whether ``path`` holds the test behind ``failure``."""
    test_id = failure.test_id or ""
    name = PurePosixPath(path).name
    if failure.kind == "pytest":
        return path == test_id.split("::")[0]
    if failure.kind == "unittest":
        return name == f"{test_id.split('.')[0]}.py"
    if failure.kind == "cargo" and path.endswith(".rs"):
        function = test_id.rsplit("::", 1)[-1]
        file = workspace / path
        return (
            file.is_file()
            and re.search(rf"\bfn\s+{re.escape(function)}\s*\(", file.read_text())
            is not None
        )
    return False


def _golden_used_by(golden: str, test_files: list[str], workspace: Path) -> bool:
    p = PurePosixPath(golden)
    for test_file in test_files:
        file = workspace / test_file
        if file.is_file():
            text = file.read_text(errors="replace")
            if p.name in text or p.stem in text:
                return True
    return False


def validate(
    changes: list[FileChange],
    verdict: Verdict,
    failures_by_key: dict[str, Failure],
    pr_changed_files: set[str],
    workspace: Path,
) -> Validation:
    violations: list[str] = []
    fixes = verdict.verdicts() & {CODE_FIX, TEST_OUTDATED}

    if not changes:
        if fixes:
            violations.append("verdict claims a fix but the diff is empty")
        return Validation(violations, [], [])
    if not fixes:
        violations.append(
            "the diff is not empty but no failure has a code-fix or test-outdated verdict"
        )

    outdated = [failures_by_key[e.failure] for e in verdict.by_verdict(TEST_OUTDATED)]
    for entry in verdict.by_verdict(TEST_OUTDATED):
        for item in entry.evidence:
            if item.file not in pr_changed_files:
                violations.append(
                    f"evidence for '{entry.failure}' cites {item.file}, which the PR does not change"
                )

    source, tests, goldens = [], [], []
    for change in changes:
        if is_forbidden_path(change.path):
            violations.append(
                f"{change.path}: this path may not be changed by the agent"
            )
        elif is_golden_path(change.path):
            goldens.append(change.path)
        elif is_test_path(change.path):
            tests.append(change.path)
        else:
            source.append(change.path)

    if source and CODE_FIX not in fixes:
        violations.append(f"source files changed without a code-fix verdict: {source}")
    if (tests or goldens) and TEST_OUTDATED not in fixes:
        violations.append(
            f"test files changed without a test-outdated verdict: {tests + goldens}"
        )
    for path in tests:
        if not any(allowed_test_file(path, f, workspace) for f in outdated):
            violations.append(f"{path}: not the test file of a test-outdated failure")
    if len(goldens) > MAX_GOLDEN_FILES:
        violations.append(
            f"{len(goldens)} golden/fixture files changed (limit {MAX_GOLDEN_FILES})"
        )
    for path in goldens:
        if not _golden_used_by(path, tests, workspace):
            violations.append(
                f"{path}: golden/fixture file changed without a changed test that uses it"
            )

    changed_lines = sum(c.changed_lines for c in changes if c.path not in goldens)
    if changed_lines > MAX_CHANGED_LINES:
        violations.append(
            f"diff changes {changed_lines} lines (limit {MAX_CHANGED_LINES})"
        )

    for change in changes:
        if change.status == "D" and (
            is_test_path(change.path) or is_golden_path(change.path)
        ):
            violations.append(f"{change.path}: test files may not be deleted")
        if _count(_SKIP_MARKERS, change.added) > _count(_SKIP_MARKERS, change.removed):
            violations.append(f"{change.path}: adds a skip/xfail/ignore marker")
        if _count(_TEST_DEFS, change.removed) > _count(_TEST_DEFS, change.added):
            violations.append(f"{change.path}: removes test functions")
        if is_test_path(change.path) or change.path.endswith(".rs"):
            if _count(_ASSERTS, change.removed) > _count(_ASSERTS, change.added):
                violations.append(f"{change.path}: reduces the number of assertions")
            if _count(_TRIVIAL_ASSERTS, change.added):
                violations.append(f"{change.path}: adds a trivially true assertion")

    return Validation(violations, source, tests + goldens)
