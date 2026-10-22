# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decide, without a model, which failures the PR caused.

Each job's failures are rerun on the PR head. Those that pass are flaky. The
rest are rerun on the merge-base with the target branch: failures there are
pre-existing, and only the remainder ("reproducible") go to the agent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from ci_autofix.failures import (
    JOB_LEVEL_KINDS,
    Failure,
    JobFailures,
    clean_log,
    extract_ids,
)
from ci_autofix.jobs import Command, JobSpec, rerun_commands

FLAKY = "flaky"
PRE_EXISTING = "pre-existing"
REPRODUCIBLE = "reproducible"
AUTOFIXED = "autofixed"
UNRUNNABLE = "unrunnable"

# pytest exit codes for "usage error / path not found" and "no tests collected".
_PYTEST_MISSING = (4, 5)


@dataclass
class Classified:
    failure: Failure
    status: str
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "failure": asdict(self.failure),
            "status": self.status,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Classified:
        return cls(Failure(**data["failure"]), data["status"], data.get("detail", ""))


def still_failing(
    failures: list[Failure], results: list, *, on_base: bool
) -> set[Failure]:
    """Which of ``failures`` failed again, given the rerun results for them."""
    failing: set[Failure] = set()
    for failure_group, result in results:
        if result.ok:
            continue
        lines = clean_log(result.output)
        kind = failure_group[0].kind
        if kind in JOB_LEVEL_KINDS or kind == "cargo-build":
            failing.update(failure_group)
            continue
        ids = set(extract_ids(lines, kind))
        if ids:
            failing.update(f for f in failure_group if f.test_id in ids)
        elif on_base and kind == "pytest" and result.returncode in _PYTEST_MISSING:
            continue  # the test does not exist on the merge-base, so the PR added it
        else:
            failing.update(failure_group)
    return failing


def _run_group(
    sandbox, spec: JobSpec, failures: list[Failure], checkout: Path
) -> tuple[list, list[Failure]]:
    """Rerun failures, one command per kind (and per module directory for unittest).

    Returns ``(results, missing)`` where ``missing`` are failures whose test
    module could not be located in this checkout.
    """
    results, missing = [], []
    for kind in dict.fromkeys(f.kind for f in failures):
        group = [f for f in failures if f.kind == kind]
        try:
            commands = rerun_commands(spec, group, checkout)
        except LookupError:
            missing.extend(group)
            continue
        for command in commands:
            results.append((group, sandbox.run(command, checkout)))
    return results, missing


def triage_job(
    sandbox, spec: JobSpec, job: JobFailures, head: Path, base: Path
) -> list[Classified]:
    if not job.failures:
        return []
    head_results, missing = _run_group(sandbox, spec, job.failures, head)
    failing_head = still_failing(job.failures, head_results, on_base=False)

    classified = [
        Classified(f, UNRUNNABLE, "could not locate the test module") for f in missing
    ]
    classified += [
        Classified(f, FLAKY, "passed on rerun")
        for f in job.failures
        if f not in failing_head and f not in missing
    ]
    candidates = [f for f in job.failures if f in failing_head]
    if not candidates:
        return classified

    # Tests missing from the merge-base were added by the PR, so they count as not failing there.
    base_results, _ = _run_group(sandbox, spec, candidates, base)
    failing_base = still_failing(candidates, base_results, on_base=True)
    for failure in candidates:
        if failure in failing_base:
            classified.append(
                Classified(failure, PRE_EXISTING, "also fails on the merge-base")
            )
        else:
            classified.append(
                Classified(
                    failure,
                    REPRODUCIBLE,
                    "fails on the PR head, passes on the merge-base",
                )
            )
    return classified


def apply_autofix(
    sandbox, spec: JobSpec, classified: list[Classified], head: Path
) -> None:
    """Run the job's deterministic fixer (cargo fmt, clippy --fix) for reproducible failures."""
    targets = [
        c
        for c in classified
        if c.status == REPRODUCIBLE and c.failure.kind in JOB_LEVEL_KINDS
    ]
    if not spec.autofix or not targets:
        return
    sandbox.run(Command(spec.autofix, ".", spec.env), head)
    results, _ = _run_group(sandbox, spec, [c.failure for c in targets], head)
    still = still_failing([c.failure for c in targets], results, on_base=False)
    for item in targets:
        if item.failure not in still:
            item.status = AUTOFIXED
            item.detail = f"fixed by `{spec.autofix}`"


def triage(
    sandbox, specs: dict[str, JobSpec], jobs: list[JobFailures], head: Path, base: Path
) -> list[Classified]:
    classified: list[Classified] = []
    for job in jobs:
        spec = specs[job.job]
        job_classified = triage_job(sandbox, spec, job, head, base)
        apply_autofix(sandbox, spec, job_classified, head)
        classified.extend(job_classified)
    return classified
