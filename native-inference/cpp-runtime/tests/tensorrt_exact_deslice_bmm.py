"""Require deslicing registration and byte parity with original CUDA einsum."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import tempfile


OPERATOR = "pnmir.tensorrt-exact-deslice-bmm"


def tensor(name, shape):
    return {"name": name, "dtype": "float32", "shape": list(shape)}


def run(cli, package, backend, scratch, *arguments):
    # AOTI initialization may clean TEMP; keep the parent Python process isolated.
    scratch.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TEMP=str(scratch), TMP=str(scratch))
    return subprocess.run(
        [
            str(cli),
            "run",
            str(package),
            "--backend",
            backend,
            "--device",
            "cpu" if backend == "mock" else "cuda",
            *arguments,
        ],
        capture_output=True,
        text=True,
        env=env,
    )


def require_success(result):
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"


def manifest(inputs, outputs, backend, artifact):
    return {
        "format_version": 1,
        "model": {"name": "exact-deslice", "version": "1"},
        "inputs": inputs,
        "outputs": outputs,
        "artifacts": [
            {
                "backend": backend,
                "target": "cpu" if backend == "mock" else "cuda",
                "precision": "fp32",
                "path": artifact,
                "required_operators": [{"id": OPERATOR, "abi": "1"}],
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pnmir", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="deslice-", dir=args.output)).resolve()
    print(f"Test artifacts: {root}", flush=True)

    # An older SDK fails this behavioral capability gate before loading a plugin.
    registration = root / "registration"
    registration.mkdir()
    (registration / "identity.mock").write_text("pnmir mock identity artifact v1\n")
    doc = manifest(
        [tensor("input", [3])], [tensor("output", [3])], "mock", "identity.mock"
    )
    path = registration / "model.json"
    path.write_text(json.dumps(doc))
    result = run(
        args.pnmir, registration, "mock", root / "child-temp", "--values", "1,2,3"
    )
    require_success(result)
    assert result.stdout.strip() == "output: 1 2 3", result.stdout
    doc["artifacts"][0]["required_operators"][0]["abi"] = "unsupported"
    path.write_text(json.dumps(doc))
    result = run(
        args.pnmir, registration, "mock", root / "child-temp", "--values", "1,2,3"
    )
    assert result.returncode != 0 and OPERATOR in result.stderr, result.stderr

    import tensorrt as trt
    import torch

    if not torch.cuda.is_available():
        print("CUDA unavailable; deslicing arithmetic was not tested")
        return 77
    torch.backends.cuda.matmul.allow_tf32 = False
    library = ctypes.CDLL(str(args.plugin.resolve()), mode=ctypes.RTLD_GLOBAL)
    assert library.pnmir_tensorrt_exact_deslice_bmm_register() == 0
    creator = trt.get_plugin_registry().get_creator("PNMIRExactDesliceBmm", "1", "")
    assert creator is not None, "deslicing creator must be registered"
    generator = torch.Generator(device="cuda").manual_seed(927)
    for tokens in (75, 2048):
        shape = (1, tokens, 8, 32)
        specifications = [
            tensor("weights", (1, tokens, 8, 512)),
            tensor("attended", (1, 512, 8, 32)),
        ]
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        inputs = [
            network.add_input(item["name"], trt.float32, tuple(item["shape"]))
            for item in specifications
        ]
        plugin = creator.create_plugin(
            "deslice", trt.PluginFieldCollection([]), trt.TensorRTPhase.BUILD
        )
        layer = network.add_plugin_v3(inputs, [], plugin)
        assert layer is not None
        restore = network.add_shuffle(layer.get_output(0))
        restore.second_transpose = (0, 2, 1, 3)
        output = restore.get_output(0)
        output.name = "output"
        network.mark_output(output)
        config = builder.create_builder_config()
        config.clear_flag(trt.BuilderFlag.TF32)
        plan = builder.build_serialized_network(network, config)
        assert plan is not None, "deslice engine must compile"
        package = root / f"n{tokens}"
        package.mkdir()
        (package / "model.plan").write_bytes(bytes(plan))
        doc = manifest(
            specifications, [tensor("output", shape)], "tensorrt", "model.plan"
        )
        doc["artifacts"][0]["runtime_version"] = trt.__version__
        (package / "model.json").write_text(json.dumps(doc))
        for index in range(3):
            weights = torch.randn(
                specifications[0]["shape"], device="cuda", generator=generator
            )
            attended = torch.randn(
                specifications[1]["shape"], device="cuda", generator=generator
            )
            # Original operation and original physical layouts, without a contiguous BHSD copy.
            expected = torch.einsum(
                "bths,bhsd->bthd", weights, attended.permute(0, 2, 1, 3)
            )
            case = package / f"case-{index}"
            case.mkdir()
            arguments = []
            for spec, value in zip(specifications, (weights, attended), strict=True):
                binary = case / f"{spec['name']}.bin"
                binary.write_bytes(value.cpu().numpy().tobytes())
                arguments += ["--input-file", f"{spec['name']}={binary}"]
            actual = case / "output.bin"
            result = run(
                args.pnmir,
                package,
                "tensorrt",
                case / "child-temp",
                *arguments,
                "--output-file",
                str(actual),
            )
            require_success(result)
            expected_bytes = expected.cpu().numpy().tobytes()
            (case / "expected.bin").write_bytes(expected_bytes)
            assert actual.read_bytes() == expected_bytes, (
                f"N={tokens} case {index} differs from eager"
            )
    print("Deslice registration and ABI gate passed; six CUDA cases byte-identical.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
