# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

from ci_autofix.verdict import CODE_FIX, TEST_OUTDATED, VerdictError, parse_verdict

KEYS = ["Python Tests :: t.py::a", "Python Tests :: t.py::b"]


def _entry(key, verdict=CODE_FIX, evidence=None):
    return {
        "failure": key,
        "verdict": verdict,
        "reason": "because",
        "evidence": evidence or [],
    }


def test_parses_valid_verdict():
    evidence = [{"file": "src/x.py", "lines": "3-4", "why": "renamed field"}]
    text = json.dumps(
        {
            "summary": "s",
            "failures": [_entry(KEYS[0]), _entry(KEYS[1], TEST_OUTDATED, evidence)],
        }
    )
    verdict = parse_verdict(text, KEYS)
    assert verdict.verdicts() == {CODE_FIX, TEST_OUTDATED}
    assert verdict.by_verdict(TEST_OUTDATED)[0].evidence[0].file == "src/x.py"


@pytest.mark.parametrize(
    "payload, message",
    [
        ("not json", "not valid JSON"),
        ({"failures": [_entry(KEYS[0])]}, "missing"),
        ({"failures": [_entry(KEYS[0]), _entry(KEYS[1]), _entry("other")]}, "unknown"),
        (
            {"failures": [_entry(KEYS[0]), _entry(KEYS[0]), _entry(KEYS[1])]},
            "duplicate",
        ),
        ({"failures": [_entry(KEYS[0], "maybe"), _entry(KEYS[1])]}, "unknown verdict"),
        (
            {"failures": [_entry(KEYS[0], TEST_OUTDATED), _entry(KEYS[1])]},
            "requires evidence",
        ),
    ],
)
def test_rejects_invalid_verdicts(payload, message):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    with pytest.raises(VerdictError, match=message):
        parse_verdict(text, KEYS)
