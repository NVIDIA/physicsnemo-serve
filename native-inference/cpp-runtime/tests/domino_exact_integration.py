"""Verify DoMINO native registration, ATen boundaries, and exact TRT arithmetic."""

import argparse
import ctypes
import json
from pathlib import Path
import tempfile

from tensorrt_exact_weighted_blend import require_success, run, tensor


BOUNDARY_ID = "physicsnemo-cfd.domino-exact-boundary"
BOUNDARY_ABI = "b1c60ddada2438469a1d24b4e53ae196425b73648f6d8ae45ecf64043755d7e6"
PLUGINS = {
    "scalar_div": "PNMIRExactScalarDiv",
    "inverse_distance_blend": "PNMIRExactInverseDistanceBlend",
}


def check_registration(cli, output):
    package = output / "registration"
    package.mkdir()
    (package / "identity.mock").write_text("pnmir mock identity artifact v1\n")
    required = [
        {"id": BOUNDARY_ID, "abi": BOUNDARY_ABI},
        *[
            {"id": "pnmir.tensorrt-exact-" + name.replace("_", "-"), "abi": "1"}
            for name in PLUGINS
        ],
    ]
    manifest = {
        "format_version": 1,
        "model": {"name": "domino-registration", "version": "1"},
        "inputs": [tensor("input", [3])],
        "outputs": [tensor("output", [3])],
        "artifacts": [
            {
                "backend": "mock",
                "target": "cpu",
                "precision": "fp32",
                "path": "identity.mock",
                "required_operators": required,
            }
        ],
    }
    path = package / "model.json"
    path.write_text(json.dumps(manifest))
    result = run(cli, package, "mock", "--values", "1,2,3")
    require_success(result)
    assert result.stdout.strip() == "output: 1 2 3", result.stdout
    required[0]["abi"] = "unsupported"
    path.write_text(json.dumps(manifest))
    result = run(cli, package, "mock", "--values", "1,2,3")
    assert result.returncode != 0 and BOUNDARY_ID in result.stderr, result.stderr


def assert_bytes(torch, actual, expected, name):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype, name
    assert (
        actual.detach().cpu().numpy().tobytes()
        == expected.detach().cpu().numpy().tobytes()
    ), name


def check_sidecar(torch, library, device):
    torch.ops.load_library(str(library.resolve()))
    ops = torch.ops.pnmir_domino
    value = torch.linspace(-2, 3, 18, device=device).reshape(2, 3, 3)
    scalar = torch.tensor(0.7, device=device)
    cases = {
        "tensor_scalar_add": (
            ops.tensor_scalar_add(value, scalar),
            value + scalar.item(),
        ),
        "tensor_scalar_div": (
            ops.tensor_scalar_div(value, scalar),
            value / scalar.item(),
        ),
        "tensor_scalar_mul": (
            ops.tensor_scalar_mul(value, scalar),
            value * scalar.item(),
        ),
        "reciprocal": (ops.reciprocal(value + 4), torch.reciprocal(value + 4)),
        "tensor_sub": (ops.tensor_sub(value, value.flip(-1)), value - value.flip(-1)),
        "vector_norm_last_dim": (
            ops.vector_norm_last_dim(value),
            torch.linalg.vector_norm(value, dim=-1, keepdim=True),
        ),
    }
    index = torch.tensor([1, 0, 1], dtype=torch.int32, device=device)
    for dimension in (0, 1):
        name = f"index_select_dim{dimension}"
        cases[name] = (
            getattr(ops, name)(value, index),
            torch.index_select(value, dimension, index.long()),
        )
    volume = value.reshape(1, 1, 2, 3, 3)
    cases["nearest_upsample3d_2x"] = (
        ops.nearest_upsample3d_2x(volume),
        torch.nn.functional.interpolate(volume, scale_factor=2, mode="nearest"),
    )
    grid = torch.linspace(-1, 1, 27, device=device).reshape(1, 3, 3, 3)
    factors = torch.tensor([0.2, 0.7], device=device)
    sdf = grid.unsqueeze(1)
    features = [
        sdf,
        *[sdf / (factor + sdf.abs()) for factor in factors],
        torch.where(sdf >= 0, 0.0, 1.0),
        *torch.gradient(sdf, dim=(2, 3, 4)),
    ]
    cases["sdf_features"] = (ops.sdf_features(grid, factors), torch.cat(features, 1))
    for name, (actual, expected) in cases.items():
        assert_bytes(torch, actual, expected, f"{device}: {name}")
    try:
        ops.tensor_scalar_div(value, scalar.reshape(1))
    except RuntimeError as error:
        assert "zero-dimensional" in str(error), str(error)
    else:
        raise AssertionError("sidecar must reject a non-scalar division operand")
    return {"device": device, "operators": sorted(cases), "byte_identical": True}


def check_tensorrt(args, torch):
    import tensorrt as trt

    handles = []
    for name, creator_name in PLUGINS.items():
        library = getattr(args, name + "_plugin")
        handles.append(ctypes.CDLL(str(library.resolve()), mode=ctypes.RTLD_GLOBAL))
        assert trt.get_plugin_registry().get_creator(creator_name, "1", "") is not None
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    shape = (3, 7, 17)
    specs = [
        tensor(name, shape)
        for name in ("center", "prediction0", "distance0", "prediction1", "distance1")
    ]
    specs.append(tensor("scale", []))
    inputs = [
        network.add_input(item["name"], trt.float32, tuple(item["shape"]))
        for item in specs
    ]

    def add_plugin(name, arguments):
        creator = trt.get_plugin_registry().get_creator(PLUGINS[name], "1", "")
        plugin = creator.create_plugin(
            name, trt.PluginFieldCollection([]), trt.TensorRTPhase.BUILD
        )
        layer = network.add_plugin_v3(arguments, [], plugin)
        assert layer is not None
        return layer.get_output(0)

    divided = add_plugin("scalar_div", [inputs[0], inputs[-1]])
    result = add_plugin("inverse_distance_blend", [divided, *inputs[1:-1]])
    result.name = "output"
    network.mark_output(result)
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    engine = builder.build_serialized_network(network, config)
    assert engine is not None
    package = args.output / "package"
    package.mkdir()
    (package / "model.plan").write_bytes(bytes(engine))
    (package / "model.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "model": {"name": "domino-exact-arithmetic", "version": "1"},
                "inputs": specs,
                "outputs": [tensor("output", shape)],
                "artifacts": [
                    {
                        "backend": "tensorrt",
                        "target": "cuda",
                        "precision": "fp32",
                        "path": "model.plan",
                        "runtime_version": trt.__version__,
                        "required_operators": [
                            {
                                "id": "pnmir.tensorrt-exact-" + name.replace("_", "-"),
                                "abi": "1",
                            }
                            for name in PLUGINS
                        ],
                    }
                ],
            }
        )
    )
    generator = torch.Generator(device="cuda").manual_seed(31)
    for index in range(3):
        center, p0, p1 = [
            torch.randn(shape, generator=generator, device="cuda") for _ in range(3)
        ]
        d0, d1 = [
            torch.rand(shape, generator=generator, device="cuda") + 0.01
            for _ in range(2)
        ]
        scale = torch.tensor((0.7, 0.1, -3.0)[index], device="cuda")
        inv0, inv1 = torch.reciprocal(d0), torch.reciprocal(d1)
        expected = (center / scale.item()) * 0.5 + ((p0 * inv0 + p1 * inv1) * 0.5) / (
            inv0 + inv1
        )
        case = args.output / f"case-{index}"
        case.mkdir()
        arguments = []
        for spec, value in zip(specs, (center, p0, d0, p1, d1, scale), strict=True):
            path = case / f"{spec['name']}.bin"
            path.write_bytes(value.cpu().numpy().tobytes())
            arguments.extend(("--input-file", f"{spec['name']}={path}"))
        result = run(
            args.pnmir,
            package,
            "tensorrt",
            *arguments,
            "--output-file",
            str(case / "output.bin"),
            "--output-metadata",
            str(case / "output.json"),
        )
        require_success(result)
        expected_bytes = expected.cpu().numpy().tobytes()
        (case / "expected.bin").write_bytes(expected_bytes)
        assert (case / "output.bin").read_bytes() == expected_bytes, (
            f"CUDA case {index} differs"
        )
        metadata = json.loads((case / "output.json").read_text())
        assert metadata["completed"] and metadata["backend"] == "tensorrt", metadata
        assert metadata["execution_device"]["type"] == "cuda", metadata
        assert metadata["outputs"][0]["shape"] == list(shape), metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pnmir", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--scalar-div-plugin", type=Path, required=True)
    parser.add_argument("--inverse-distance-blend-plugin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.output = Path(tempfile.mkdtemp(prefix="domino-exact-", dir=args.output))
    print(f"Test artifacts: {args.output}", flush=True)
    # First behavioral assertion can run against the older SDK without any new DSOs.
    check_registration(args.pnmir, args.output)
    import torch

    reports = [check_sidecar(torch, args.sidecar, "cpu")]
    if not torch.cuda.is_available():
        print("CUDA unavailable; CUDA boundaries and TRT arithmetic were not tested")
        return 77
    reports.append(check_sidecar(torch, args.sidecar, "cuda"))
    check_tensorrt(args, torch)
    (args.output / "sidecar-checks.json").write_text(
        json.dumps(reports, indent=2) + "\n"
    )
    print(
        "DoMINO operator/ABI gates passed; ten sidecar ops exact on CPU/CUDA; three TRT CUDA cases exact."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
