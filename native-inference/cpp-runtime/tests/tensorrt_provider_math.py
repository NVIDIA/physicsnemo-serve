"""Real H100 parity for serialized provider and legacy IEEE BLAS plugins."""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path


# TensorRT retains a process-global logger pointer. Keep its Python owner alive
# until all builder/runtime objects and their execution contexts are destroyed.
_LOGGER = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import tensorrt as trt
    import torch

    global _LOGGER
    _LOGGER = trt.Logger(trt.Logger.WARNING)
    logger = _LOGGER
    if not torch.cuda.is_available():
        return 77
    assert torch.cuda.get_device_capability() == (9, 0), "qualified H100 required"
    args.output.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.preferred_blas_library("cublas")
    generator = torch.Generator(device="cuda").manual_seed(27531)

    def generated(shape):
        return torch.randn(shape, device="cuda", generator=generator)

    def raw(tensor):
        return tensor.detach().cpu().contiguous().numpy().tobytes()

    weights = generated((1, 257, 8, 512))
    features = generated((1, 257, 8, 32))
    attended = generated((1, 512, 8, 32))
    matrix_input, matrix_weight, bias = (
        generated((1, 257, 256)),
        generated((256, 256)),
        generated((256,)),
    )
    cases = (
        (
            "linear",
            "PNMIRExactLinear",
            (matrix_input, matrix_weight, bias),
            lambda x, weight, bias: torch.nn.functional.linear(x, weight, bias),
            False,
        ),
        (
            "gemm",
            "PNMIRExactGemm",
            (matrix_input, matrix_weight),
            lambda x, weight: x @ weight.t(),
            False,
        ),
        (
            "slice_bmm",
            "PNMIRExactSliceBmm",
            (weights, features),
            lambda weights, features: torch.einsum(
                "bths,bthd->bhsd", weights, features
            ),
            False,
        ),
        (
            "deslice_bmm",
            "PNMIRExactDesliceBmm",
            (weights, attended),
            lambda weights, attended: torch.einsum(
                "bths,bhsd->bthd", weights, attended.permute(0, 2, 1, 3)
            ),
            True,
        ),
    )
    handles, records = [], []
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    with torch.inference_mode(), torch.cuda.stream(stream):
        for name, creator_name, inputs, reference, restore_deslice in cases:
            handles.append(
                ctypes.CDLL(
                    str(args.plugin_root / f"libpnmir_tensorrt_exact_{name}_plugin.so"),
                    mode=ctypes.RTLD_GLOBAL,
                )
            )
            for version in ("1", "2"):
                print(f"START {name} v{version}", flush=True)
                torch.set_float32_matmul_precision(
                    "high" if version == "2" else "highest"
                )
                expected = reference(*inputs).contiguous()
                stream.synchronize()
                expected_data = raw(expected)
                creator = trt.get_plugin_registry().get_creator(
                    creator_name, version, ""
                )
                assert creator is not None
                policy = (
                    dict(
                        compute_mode=1,
                        cublas_workspace_bytes=33554432,
                        cublaslt_workspace_bytes=1048576,
                    )
                    if version == "2"
                    else {}
                )
                fields = trt.PluginFieldCollection(
                    [
                        trt.PluginField(
                            key,
                            np.array([value], dtype=np.int32),
                            trt.PluginFieldType.INT32,
                        )
                        for key, value in policy.items()
                    ]
                )
                builder = trt.Builder(logger)
                network = builder.create_network(
                    1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
                )
                native_inputs = [
                    network.add_input(f"input{index}", trt.float32, tuple(value.shape))
                    for index, value in enumerate(inputs)
                ]
                plugin = creator.create_plugin(name, fields, trt.TensorRTPhase.BUILD)
                assert plugin is not None
                layer = network.add_plugin_v3(native_inputs, [], plugin)
                assert layer is not None
                output = layer.get_output(0)
                if restore_deslice:
                    restore = network.add_shuffle(output)
                    restore.second_transpose = (0, 2, 1, 3)
                    output = restore.get_output(0)
                output.name = "output"
                network.mark_output(output)
                config = builder.create_builder_config()
                config.clear_flag(trt.BuilderFlag.TF32)
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
                plan = builder.build_serialized_network(network, config)
                assert plan is not None, f"{name} v{version}: build failed"
                print(f"BUILT {name} v{version}", flush=True)
                (args.output / f"{name}-v{version}.plan").write_bytes(bytes(plan))
                runtime = trt.Runtime(logger)
                engine = runtime.deserialize_cuda_engine(plan)
                assert engine is not None, f"{name} v{version}: deserialize failed"
                context = engine.create_execution_context()
                assert context is not None
                print(f"LOADED {name} v{version}", flush=True)
                actual = torch.empty_like(expected)
                for index, value in enumerate(inputs):
                    assert context.set_tensor_address(f"input{index}", value.data_ptr())
                assert context.set_tensor_address("output", actual.data_ptr())
                # The serialized policy, not mutable Torch process settings,
                # must control each plugin's private BLAS handle after reload.
                torch.set_float32_matmul_precision(
                    "highest" if version == "2" else "high"
                )
                for repeat in range(2):
                    actual.fill_(float("nan"))
                    assert context.execute_async_v3(stream.cuda_stream)
                    stream.synchronize()
                    actual_data = raw(actual)
                    assert actual_data == expected_data, (
                        f"{name} v{version} repeat{repeat}: byte parity failed"
                    )
                records.append(
                    {
                        "plugin": name,
                        "creator_version": version,
                        "byte_identical": True,
                        "repeats": 2,
                        "sha256": hashlib.sha256(expected_data).hexdigest(),
                    }
                )
                (args.output / "summary.json").write_text(
                    json.dumps({"status": "running", "cases": records}, indent=2) + "\n"
                )
                print(
                    f"PASS {name} v{version}: byte parity on both executions",
                    flush=True,
                )
    (args.output / "summary.json").write_text(
        json.dumps({"status": "passed", "cases": records}, indent=2) + "\n"
    )
    print(json.dumps({"passed": len(records), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
