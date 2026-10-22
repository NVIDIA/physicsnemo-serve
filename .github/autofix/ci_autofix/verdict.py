# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schema for the agent's verdict.json."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

CODE_FIX = "code-fix"
TEST_OUTDATED = "test-outdated"
NOT_CAUSED = "not-caused-by-pr"
NEEDS_HUMAN = "needs-human"
VERDICTS = (CODE_FIX, TEST_OUTDATED, NOT_CAUSED, NEEDS_HUMAN)
FIX_VERDICTS = (CODE_FIX, TEST_OUTDATED)

EXAMPLE = {
    "summary": "One sentence describing what happened.",
    "failures": [
        {
            "failure": "<failure key exactly as listed in the task>",
            "verdict": "|".join(VERDICTS),
            "reason": "Why this verdict is correct.",
            "evidence": [
                {
                    "file": "path/changed/by/the/pr.py",
                    "lines": "40-52",
                    "why": "Required for test-outdated: the PR change that intentionally altered this behaviour.",
                }
            ],
        }
    ],
}


class VerdictError(ValueError):
    pass


@dataclass
class Evidence:
    file: str
    lines: str
    why: str


@dataclass
class Entry:
    failure: str
    verdict: str
    reason: str
    evidence: list[Evidence] = field(default_factory=list)


@dataclass
class Verdict:
    summary: str
    entries: list[Entry]

    def verdicts(self) -> set[str]:
        return {e.verdict for e in self.entries}

    def by_verdict(self, verdict: str) -> list[Entry]:
        return [e for e in self.entries if e.verdict == verdict]

    def to_dict(self) -> dict:
        return {"summary": self.summary, "failures": [asdict(e) for e in self.entries]}


def needs_human_for_all(keys: list[str], reason: str) -> Verdict:
    return Verdict(
        summary=reason, entries=[Entry(k, NEEDS_HUMAN, reason) for k in keys]
    )


def _require_str(data: dict, key: str, where: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise VerdictError(f"{where}: '{key}' must be a non-empty string")
    return value.strip()


def parse_verdict(text: str, expected_keys: list[str]) -> Verdict:
    """Parse verdict.json and check it covers exactly the failures the agent was given."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise VerdictError(f"verdict.json is not valid JSON: {error}") from error
    if not isinstance(data, dict) or not isinstance(data.get("failures"), list):
        raise VerdictError("verdict.json must be an object with a 'failures' list")

    entries = []
    for index, raw in enumerate(data["failures"]):
        where = f"failures[{index}]"
        if not isinstance(raw, dict):
            raise VerdictError(f"{where} must be an object")
        verdict = _require_str(raw, "verdict", where)
        if verdict not in VERDICTS:
            raise VerdictError(f"{where}: unknown verdict '{verdict}'")
        evidence = []
        for j, item in enumerate(raw.get("evidence") or []):
            if not isinstance(item, dict):
                raise VerdictError(f"{where}.evidence[{j}] must be an object")
            evidence.append(
                Evidence(
                    *(
                        _require_str(item, k, f"{where}.evidence[{j}]")
                        for k in ("file", "lines", "why")
                    )
                )
            )
        if verdict == TEST_OUTDATED and not evidence:
            raise VerdictError(
                f"{where}: test-outdated requires evidence from the PR diff"
            )
        entries.append(
            Entry(
                _require_str(raw, "failure", where),
                verdict,
                _require_str(raw, "reason", where),
                evidence,
            )
        )

    seen = [e.failure for e in entries]
    duplicates = sorted({k for k in seen if seen.count(k) > 1})
    unknown = sorted(set(seen) - set(expected_keys))
    missing = sorted(set(expected_keys) - set(seen))
    if duplicates or unknown or missing:
        raise VerdictError(
            f"failure keys mismatch: missing={missing} unknown={unknown} duplicate={duplicates}"
        )
    summary = data.get("summary") if isinstance(data.get("summary"), str) else ""
    return Verdict(summary=summary.strip(), entries=entries)
