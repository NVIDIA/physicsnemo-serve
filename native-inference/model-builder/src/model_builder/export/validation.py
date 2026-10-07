from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import torch

from .exporter import _tensor_outputs
from .parity import ParityMetrics, assert_tensor_parity


def validate_pnmir_package(
    pnmir: str | Path,
    package: str | Path,
    inputs: tuple[torch.Tensor, ...],
    expected_outputs: torch.Tensor | tuple[torch.Tensor, ...],
    *,
    backend: str,
    device: str,
    max_abs_limit: float = 1.0e-4,
    relative_l2_limit: float = 1.0e-4,
) -> tuple[ParityMetrics, ...]:
    """Run a package through standalone PNM-IR and check output parity."""
    executable = Path(pnmir).expanduser().resolve()
    package_path = Path(package).expanduser().resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"PNM-IR executable does not exist: {executable}")
    manifest = json.loads((package_path / "model.json").read_text(encoding="utf-8"))
    input_specs = manifest["inputs"]
    output_specs = manifest["outputs"]
    if len(inputs) != len(input_specs):
        raise ValueError(
            f"expected {len(input_specs)} validation inputs, got {len(inputs)}"
        )
    outputs = _tensor_outputs(expected_outputs)
    if len(outputs) != len(output_specs):
        raise ValueError(
            f"expected {len(output_specs)} validation outputs, got {len(outputs)}"
        )

    with tempfile.TemporaryDirectory(
        prefix=f"pnm-ir-{backend}-validation-", dir=package_path.parent
    ) as temporary:
        root = Path(temporary)
        command = [
            str(executable),
            "run",
            str(package_path),
            "--backend",
            backend,
            "--device",
            device,
        ]
        for index, (spec, value) in enumerate(zip(input_specs, inputs, strict=True)):
            path = root / f"input-{index}.bin"
            path.write_bytes(value.detach().cpu().contiguous().numpy().tobytes())
            command.extend(("--input-file", f"{spec['name']}={path}"))
            if -1 in spec["shape"]:
                shape = ",".join(str(dimension) for dimension in value.shape)
                command.extend(("--input-shape", f"{spec['name']}={shape}"))
        output_paths = []
        for index, spec in enumerate(output_specs):
            path = root / f"output-{index}.bin"
            output_paths.append(path)
            command.extend(("--output-file", f"{spec['name']}={path}"))
        dynamic_outputs = any(-1 in spec["shape"] for spec in output_specs)
        metadata_path = root / "output-metadata.json"
        if dynamic_outputs:
            command.extend(("--output-metadata", str(metadata_path)))

        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"PNM-IR {backend} validation failed with exit "
                f"{result.returncode}\nstdout:\n{result.stdout}"
                f"\nstderr:\n{result.stderr}"
            )

        output_shapes = {spec["name"]: spec["shape"] for spec in output_specs}
        if dynamic_outputs:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            output_shapes = {
                spec["name"]: spec["shape"] for spec in metadata["outputs"]
            }
        metrics = []
        for spec, path, expected in zip(
            output_specs, output_paths, outputs, strict=True
        ):
            if spec["dtype"] != "float32":
                raise ValueError(
                    "package validation currently requires float32 outputs"
                )
            storage = bytearray(path.read_bytes())
            actual = torch.frombuffer(storage, dtype=torch.float32).clone()
            actual = actual.reshape(tuple(output_shapes[spec["name"]]))
            metrics.append(
                assert_tensor_parity(
                    actual,
                    expected.detach().cpu(),
                    label=spec["name"],
                    max_abs_limit=max_abs_limit,
                    relative_l2_limit=relative_l2_limit,
                )
            )
        return tuple(metrics)


def validate_tensorrt_package(
    pnmir: str | Path,
    package: str | Path,
    inputs: tuple[torch.Tensor, ...],
    expected_outputs: torch.Tensor | tuple[torch.Tensor, ...],
    *,
    max_abs_limit: float = 1.0e-4,
    relative_l2_limit: float = 1.0e-4,
) -> tuple[ParityMetrics, ...]:
    return validate_pnmir_package(
        pnmir,
        package,
        inputs,
        expected_outputs,
        backend="tensorrt",
        device="cuda",
        max_abs_limit=max_abs_limit,
        relative_l2_limit=relative_l2_limit,
    )


def validate_onnxruntime_package(
    pnmir: str | Path,
    package: str | Path,
    inputs: tuple[torch.Tensor, ...],
    expected_outputs: torch.Tensor | tuple[torch.Tensor, ...],
    *,
    max_abs_limit: float = 1.0e-4,
    relative_l2_limit: float = 1.0e-4,
) -> tuple[ParityMetrics, ...]:
    return validate_pnmir_package(
        pnmir,
        package,
        inputs,
        expected_outputs,
        backend="onnxruntime",
        device="cuda",
        max_abs_limit=max_abs_limit,
        relative_l2_limit=relative_l2_limit,
    )
