# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic checks on the agent's staged changes, run before anything is pushed.

Run with GIT_DIR and GIT_WORK_TREE pointing at the repository and the agent's
working copy. Prints a markdown list for the PR comment; exits 1 on violations.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import PurePosixPath

MAX_CHANGED_LINES = 400
FORBIDDEN_FILES = frozenset({"Cargo.lock", "uv.lock"})

_SKIP_MARKERS = re.compile(
    r"@pytest\.mark\.(skip|skipif|xfail)\b|\bpytest\.(skip|xfail)\("
    r"|@unittest\.(skip|skipIf|skipUnless|expectedFailure)\b|\bself\.skipTest\(|#\[ignore\b"
)
_TEST_DEFS = re.compile(r"^\s*(async\s+)?def test\w*\(|^\s*#\[(tokio::)?test\b")


def is_test_path(path: str) -> bool:
    p = PurePosixPath(path)
    return (
        bool({"tests", "test"} & set(p.parts[:-1]))
        or p.name.startswith("test_")
        or p.stem.endswith(("_test", "_tests"))
        or p.name == "conftest.py"
    )


def _count(pattern: re.Pattern, lines: list[str]) -> int:
    return sum(1 for line in lines if pattern.search(line))


def _lines_by_file(unified: str) -> dict[str, tuple[list[str], list[str]]]:
    files: dict[str, tuple[list[str], list[str]]] = {}
    current = None
    for line in unified.splitlines():
        if line.startswith("diff --git "):
            current = files.setdefault(line.rsplit(" b/", 1)[-1], ([], []))
        elif current is None or line.startswith(("+++", "---")):
            continue
        elif line.startswith("+"):
            current[0].append(line[1:])
        elif line.startswith("-"):
            current[1].append(line[1:])
    return files


def check(raw: str, numstat: str, unified: str) -> list[str]:
    """Violations in a diff given as ``--raw``, ``--numstat`` and ``-U0`` output."""
    problems = []
    for line in raw.splitlines():
        meta, _, path = line.partition("\t")
        _, mode, *_, status = meta.lstrip(":").split()
        parts = PurePosixPath(path).parts
        if any(p.startswith(".git") for p in parts) or parts[-1] in FORBIDDEN_FILES:
            problems.append(f"{path}: the agent may not change this path")
        if mode in ("120000", "160000"):
            problems.append(f"{path}: symlinks and submodules are not allowed")
        if status[0] == "D" and is_test_path(path):
            problems.append(f"{path}: test files may not be deleted")

    total = 0
    for line in numstat.splitlines():
        added, removed, path = line.split("\t", 2)
        if added == "-":
            problems.append(f"{path}: binary files are not allowed")
        else:
            total += int(added) + int(removed)
    if total > MAX_CHANGED_LINES:
        problems.append(f"the diff changes {total} lines (limit {MAX_CHANGED_LINES})")

    for path, (added, removed) in _lines_by_file(unified).items():
        if _count(_SKIP_MARKERS, added) > _count(_SKIP_MARKERS, removed):
            problems.append(f"{path}: adds a skip/xfail/ignore marker")
        if _count(_TEST_DEFS, removed) > _count(_TEST_DEFS, added):
            problems.append(f"{path}: removes test functions")
    return problems


def staged_diff(cwd: str | None = None) -> tuple[str, str, str]:
    def git(*args: str) -> str:
        cmd = ["git", "diff", "--cached", "--no-renames", *args]
        return subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, check=True
        ).stdout

    return git("--raw"), git("--numstat"), git("-U0")


def main() -> int:
    raw, numstat, unified = staged_diff()
    problems = check(raw, numstat, unified)
    tests = [
        line.partition("\t")[2]
        for line in raw.splitlines()
        if is_test_path(line.partition("\t")[2])
    ]
    for problem in problems:
        print(f"- {problem}")
    if tests:
        print(f"- note: test files changed: {', '.join(tests)}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
