"""Exercise the same installed-SDK consumer locally through the mock backend."""

import json
import os
from pathlib import Path
import struct
import subprocess

import pytest


@pytest.fixture
def executable():
    path = os.environ.get("NATIVE_QA_CONSUMER")
    if not path:
        pytest.skip("set NATIVE_QA_CONSUMER to the installed CPU QA consumer")
    assert Path(path).is_file(), "configured consumer is missing"
    return path


@pytest.fixture
def package(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "identity.mock").write_text("identity")
    (root / "model.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "model": {"name": "identity", "version": "1"},
                "inputs": [{"name": "input", "dtype": "float32", "shape": [4]}],
                "outputs": [{"name": "output", "dtype": "float32", "shape": [4]}],
                "artifacts": [
                    {
                        "backend": "mock",
                        "target": "cpu",
                        "precision": "fp32",
                        "path": "identity.mock",
                    }
                ],
            }
        )
    )
    return root


def test_executor_consumes_distinct_inputs_and_preserves_output_bytes(
    executable, package, tmp_path
):
    inputs = [
        struct.pack("<4f", *values)
        for values in ([1, 2, 3, 4], [-9, 0, 3.5, 7], [8, 1, -2, 6])
    ]
    paths = []
    for index, values in enumerate(inputs):
        path = tmp_path / f"input-{index}"
        path.write_bytes(values)
        paths.append(str(path))
    output = tmp_path / "outputs"
    output.mkdir()
    result = subprocess.run(
        [executable, str(package), "mock", str(output), *paths],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "metadata.json").is_file(), (
        "consumer must report completed repeated execution"
    )
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["requests"] == 3
    assert metadata["executor_count"] == 1
    for index, expected in enumerate(inputs):
        assert (output / f"output-{index}.f32").read_bytes() == expected


def test_truncated_request_fails_without_completion_metadata(
    executable, package, tmp_path
):
    bad = tmp_path / "truncated"
    bad.write_bytes(b"\0")
    output = tmp_path / "outputs"
    output.mkdir()
    result = subprocess.run(
        [executable, str(package), "mock", str(output), str(bad), str(bad)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not (output / "metadata.json").exists()
