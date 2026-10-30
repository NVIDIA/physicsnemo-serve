"""Provider qualification must prove every serialized BLAS policy on real outputs."""

import hashlib
import json
import struct

import pytest

from qa.native_inference import provider


def math_evidence(tmp_path):
    records = []
    for name in ("linear", "gemm", "slice_bmm", "deslice_bmm"):
        for version in ("1", "2"):
            stem = f"{name}-v{version}"
            data = struct.pack("<2f", 1.0, -2.0)
            (tmp_path / f"{stem}-reference.bin").write_bytes(data)
            for repeat in range(2):
                (tmp_path / f"{stem}-actual-{repeat}.bin").write_bytes(data)
            records.append(
                dict(
                    plugin=name,
                    creator_version=version,
                    byte_identical=True,
                    repeats=2,
                    sha256=hashlib.sha256(data).hexdigest(),
                )
            )
    (tmp_path / "summary.json").write_text(
        json.dumps({"status": "passed", "cases": records})
    )
    return records


def test_math_verifier_checks_all_eight_cases_and_both_saved_executions(tmp_path):
    math_evidence(tmp_path)
    assert len(provider.verify_math(tmp_path)["cases"]) == 8
    (tmp_path / "linear-v2-actual-1.bin").write_bytes(struct.pack("<2f", 1.0, -2.1))
    with pytest.raises(ValueError, match="byte parity"):
        provider.verify_math(tmp_path)


@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate", "version", "repeat", "status", "hash", "nonfinite"],
)
def test_math_flags_cannot_replace_complete_finite_evidence(tmp_path, mutation):
    records = math_evidence(tmp_path)
    status = "passed"
    if mutation == "missing":
        records.pop()
    elif mutation == "duplicate":
        records[-1] = records[0]
    elif mutation == "version":
        records[-1]["creator_version"] = "3"
    elif mutation == "repeat":
        records[-1]["repeats"] = 1
    elif mutation == "status":
        status = "running"
    elif mutation == "hash":
        records[-1]["sha256"] = "0" * 64
    else:
        data = struct.pack("<2f", float("nan"), 1.0)
        for suffix in ("reference", "actual-0", "actual-1"):
            (tmp_path / f"deslice_bmm-v2-{suffix}.bin").write_bytes(data)
        records[-1]["sha256"] = hashlib.sha256(data).hexdigest()
    (tmp_path / "summary.json").write_text(
        json.dumps({"status": status, "cases": records})
    )
    with pytest.raises(ValueError):
        provider.verify_math(tmp_path)
