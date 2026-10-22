# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extract failing test IDs from GitHub Actions job logs."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

# Kinds that represent the whole job rather than individual tests.
JOB_LEVEL_KINDS = frozenset({"clippy", "fmt"})

_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z ?")
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_PYTEST = re.compile(r"^(?:FAILED|ERROR) (\S+?)(?: - .*)?$")
_UNITTEST = re.compile(r"^(?:FAIL|ERROR): (\w+) \(([\w.]+)\)")
_CARGO_TEST = re.compile(r"^test (\S+) \.\.\. FAILED$")
_CARGO_BUILD = re.compile(r"^error: could not compile ")
_MARKERS = re.compile(
    r"(=+ (FAILURES|ERRORS) =+|^failures:$|^(FAIL|ERROR): |^error(\[E\d+\])?: |^Traceback |^Diff in |^warning: )"
)

EXCERPT_HEAD = 100
EXCERPT_TAIL = 300
EXCERPT_LEAD = 20


@dataclass(frozen=True)
class Failure:
    job: str
    kind: str
    test_id: str | None = None

    @property
    def key(self) -> str:
        """Stable identifier the agent uses in verdict.json."""
        return f"{self.job} :: {self.test_id}" if self.test_id else self.job


@dataclass
class JobFailures:
    job: str
    failures: list[Failure] = field(default_factory=list)
    excerpt: str = ""

    @property
    def parsed(self) -> bool:
        return bool(self.failures)

    def to_dict(self) -> dict:
        return {
            "job": self.job,
            "failures": [asdict(f) for f in self.failures],
            "excerpt": self.excerpt,
        }

    @classmethod
    def from_dict(cls, data: dict) -> JobFailures:
        return cls(
            job=data["job"],
            failures=[Failure(**f) for f in data["failures"]],
            excerpt=data.get("excerpt", ""),
        )


def clean_log(text: str) -> list[str]:
    """Strip Actions timestamps and ANSI colour codes."""
    return [_ANSI.sub("", _TIMESTAMP.sub("", line)) for line in text.splitlines()]


def _unittest_id(method: str, qualifier: str) -> str:
    if qualifier.startswith("unittest.loader._FailedTest"):
        # Import failure: unittest reports the module name as the "method".
        return method
    if qualifier.endswith(f".{method}"):
        return qualifier
    return f"{qualifier}.{method}"


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def extract_ids(lines: list[str], kind: str) -> list[str]:
    if kind == "pytest":
        ids = [m.group(1) for line in lines if (m := _PYTEST.match(line))]
        return _dedupe([i for i in ids if "::" in i or i.endswith(".py")])
    if kind == "unittest":
        return _dedupe(
            [
                _unittest_id(m.group(1), m.group(2))
                for line in lines
                if (m := _UNITTEST.match(line))
            ]
        )
    if kind == "cargo":
        return _dedupe([m.group(1) for line in lines if (m := _CARGO_TEST.match(line))])
    raise ValueError(f"unknown test-id kind: {kind}")


def excerpt(lines: list[str]) -> str:
    """Keep the part of the log around the first failure, bounded in size."""
    start = next((i for i, line in enumerate(lines) if _MARKERS.search(line)), 0)
    kept = lines[max(0, start - EXCERPT_LEAD) :]
    if len(kept) > EXCERPT_HEAD + EXCERPT_TAIL:
        elided = len(kept) - EXCERPT_HEAD - EXCERPT_TAIL
        kept = [
            *kept[:EXCERPT_HEAD],
            f"... [{elided} lines elided] ...",
            *kept[-EXCERPT_TAIL:],
        ]
    return "\n".join(kept)


def parse_job_log(job: str, log_text: str, kinds: list[str]) -> JobFailures:
    """Find failures of the given kinds in one failed job's log.

    A job with no recognisable failure is returned with an empty list, which the
    caller reports as needing a human.
    """
    lines = clean_log(log_text)
    failures: list[Failure] = []
    for kind in kinds:
        if kind in JOB_LEVEL_KINDS:
            failures.append(Failure(job, kind))
        elif kind == "cargo-build":
            continue
        else:
            failures.extend(
                Failure(job, kind, test_id) for test_id in extract_ids(lines, kind)
            )
    if (
        "cargo-build" in kinds
        and not failures
        and any(_CARGO_BUILD.match(line) for line in lines)
    ):
        failures.append(Failure(job, "cargo-build"))
    return JobFailures(job=job, failures=failures, excerpt=excerpt(lines))
