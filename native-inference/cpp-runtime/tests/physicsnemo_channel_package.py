from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


COORDINATES = (
    0.0,
    -1.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.5,
    -1.0,
    0.5,
    0.0,
    0.5,
    1.0,
    1.0,
    -1.0,
    1.0,
    0.0,
    1.0,
    1.0,
)

EXPECTED_FLOW = (
    0.0,
    0.0,
    1.0,
    1.0,
    0.0,
    1.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.5,
    1.0,
    0.0,
    0.5,
    0.0,
    0.0,
    0.5,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnmir", required=True, type=Path)
    parser.add_argument("--package", required=True, type=Path)
    args = parser.parse_args()

    manifest = json.loads((args.package / "model.json").read_text(encoding="utf-8"))
    assert manifest["model"] == {
        "name": "physicsnemo-channel-flow",
        "version": "0.1.0",
    }
    assert manifest["inputs"] == [
        {"name": "coordinates", "dtype": "float32", "shape": [9, 2]}
    ]
    assert manifest["outputs"] == [
        {"name": "flow", "dtype": "float32", "shape": [9, 3]}
    ]
    assert manifest["artifacts"] == [
        {
            "backend": "onnxruntime",
            "target": "cpu",
            "precision": "fp32",
            "path": "artifacts/onnx/model.onnx",
        }
    ]

    result = subprocess.run(
        [
            str(args.pnmir),
            "run",
            str(args.package),
            "--values",
            ",".join(str(value) for value in COORDINATES),
            "--backend",
            "onnxruntime",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"channel-flow inference failed with exit {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    name, separator, values_text = result.stdout.strip().partition(":")
    assert separator == ":" and name == "flow", result.stdout
    actual = tuple(float(value) for value in values_text.split())
    assert len(actual) == len(EXPECTED_FLOW)
    max_error = max(
        abs(actual_value - expected_value)
        for actual_value, expected_value in zip(actual, EXPECTED_FLOW, strict=True)
    )
    assert max_error <= 0.01, max_error
    print(f"channel-flow maximum analytic error: {max_error:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
