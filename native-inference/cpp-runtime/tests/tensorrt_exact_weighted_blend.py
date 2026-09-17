"""Exercise exact operator registration and standalone weighted-blend inference."""

import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import tempfile


OPERATOR = "pnmir.tensorrt-exact-weighted-blend"


def run(cli, package, backend, *arguments):
    result = subprocess.run(
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
    )
    return result


def require_success(result):
    assert result.returncode == 0, (
        f"native inference failed ({result.returncode})\n"
        f"{result.stdout}\n{result.stderr}"
    )


def tensor(name, shape):
    return {"name": name, "dtype": "float32", "shape": list(shape)}


def check_registration(cli, output):
    package = output / "registration"
    package.mkdir()
    (package / "identity.mock").write_text("pnmir mock identity artifact v1\n")
    # WeightedBlend is first so an older eight-plugin SDK fails at this
    # behavioral assertion before any new library or producer imports are used.
    operators = [
        OPERATOR,
        *[
            f"pnmir.tensorrt-exact-{name}"
            for name in (
                "linear",
                "gemm",
                "token-sum",
                "slice-bmm",
                "layer-norm",
                "softmax",
                "attention",
                "gelu",
            )
        ],
    ]
    manifest = {
        "format_version": 1,
        "model": {"name": "exact-registration", "version": "1"},
        "inputs": [tensor("input", [3])],
        "outputs": [tensor("output", [3])],
        "artifacts": [
            {
                "backend": "mock",
                "target": "cpu",
                "precision": "fp32",
                "path": "identity.mock",
                "required_operators": [{"id": name, "abi": "1"} for name in operators],
            }
        ],
    }
    path = package / "model.json"
    path.write_text(json.dumps(manifest))
    result = run(cli, package, "mock", "--values", "1,2,3")
    require_success(result)
    assert result.stdout.strip() == "output: 1 2 3", result.stdout
    manifest["artifacts"][0]["required_operators"][0]["abi"] = "unsupported"
    path.write_text(json.dumps(manifest))
    result = run(cli, package, "mock", "--values", "1,2,3")
    assert result.returncode != 0, "wrong weighted-blend ABI must be rejected"
    assert OPERATOR in result.stderr, result.stderr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pnmir", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.output = Path(tempfile.mkdtemp(prefix="weighted-blend-", dir=args.output))
    print(f"Test artifacts: {args.output}", flush=True)
    check_registration(args.pnmir, args.output)

    import tensorrt as trt
    import torch

    if not torch.cuda.is_available():
        print("CUDA unavailable; weighted-blend arithmetic was not tested")
        return 77
    library = ctypes.CDLL(str(args.plugin.resolve()), mode=ctypes.RTLD_GLOBAL)
    assert library.pnmir_tensorrt_exact_weighted_blend_register() == 0
    creator = trt.get_plugin_registry().get_creator("PNMIRExactWeightedBlend", "1", "")
    assert creator is not None, "weighted-blend TensorRT creator must be registered"
    plugin = creator.create_plugin(
        "weighted-blend", trt.PluginFieldCollection([]), trt.TensorRTPhase.BUILD
    )
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    shape = (3, 7, 17)
    specifications = [
        tensor("left", shape),
        tensor("left_weight", []),
        tensor("right", shape),
        tensor("right_weight", []),
    ]
    inputs = [
        network.add_input(item["name"], trt.float32, tuple(item["shape"]))
        for item in specifications
    ]
    layer = network.add_plugin_v3(inputs, [], plugin)
    assert layer is not None
    result = layer.get_output(0)
    result.name = "output"
    network.mark_output(result)
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    engine = builder.build_serialized_network(network, config)
    assert engine is not None, "weighted-blend network must compile"
    package = args.output / "package"
    package.mkdir()
    (package / "model.plan").write_bytes(bytes(engine))
    (package / "model.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "model": {"name": "exact-weighted-blend", "version": "1"},
                "inputs": specifications,
                "outputs": [tensor("output", shape)],
                "artifacts": [
                    {
                        "backend": "tensorrt",
                        "target": "cuda",
                        "precision": "fp32",
                        "path": "model.plan",
                        "runtime_version": trt.__version__,
                        "required_operators": [{"id": OPERATOR, "abi": "1"}],
                    }
                ],
            }
        )
    )
    generator = torch.Generator(device="cuda").manual_seed(17)
    for index, (a, b) in enumerate(((1 + 2**-23, 1.0), (0.1, 0.9), (-0.75, 1.125))):
        left = torch.randn(shape, generator=generator, device="cuda")
        right = torch.randn(shape, generator=generator, device="cuda")
        if index == 0:
            left.flatten()[0] = 1 + 2**-23
            right.flatten()[0] = -(1 + 2**-22)
        left_weight = torch.tensor(a, dtype=torch.float32, device="cuda")
        right_weight = torch.tensor(b, dtype=torch.float32, device="cuda")
        expected = left * left_weight + right * right_weight
        if index == 0:
            fused = (
                left.double() * left_weight.double() + (right * right_weight).double()
            ).float()
            assert expected.flatten()[0].item() == 0
            assert fused.flatten()[0].item() != 0, "fixture must detect unwanted FMA"
        case = args.output / f"case-{index}"
        case.mkdir()
        arguments = []
        for specification, value in zip(
            specifications, (left, left_weight, right, right_weight), strict=True
        ):
            path = case / f"{specification['name']}.bin"
            path.write_bytes(value.cpu().numpy().tobytes())
            arguments += ["--input-file", f"{specification['name']}={path}"]
        output = case / "output.bin"
        metadata = case / "output.json"
        result = run(
            args.pnmir,
            package,
            "tensorrt",
            *arguments,
            "--output-file",
            str(output),
            "--output-metadata",
            str(metadata),
        )
        require_success(result)
        expected_bytes = expected.cpu().numpy().tobytes()
        (case / "expected.bin").write_bytes(expected_bytes)
        assert output.read_bytes() == expected_bytes, (
            f"case {index} differs from eager CUDA"
        )
        report = json.loads(metadata.read_text())
        assert report["completed"] and report["backend"] == "tensorrt", report
        assert report["execution_device"]["type"] == "cuda", report
        assert report["outputs"][0]["shape"] == list(shape), report
    print(
        "All nine operator IDs registered; ABI gate passed; three CUDA cases byte-identical."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
