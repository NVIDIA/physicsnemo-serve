"""Require exact Linear parity with original eager CUDA, including singleton axes."""

import argparse
import ctypes
import json
import math
from pathlib import Path
import random
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import tensorrt as trt
    import torch

    if not torch.cuda.is_available():
        print("CUDA unavailable; exact Linear arithmetic was not tested")
        return 77
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="linear-", dir=args.output)).resolve()
    print(f"Test artifacts: {root}", flush=True)
    library = ctypes.CDLL(str(args.plugin.resolve()), mode=ctypes.RTLD_GLOBAL)
    assert library.pnmir_tensorrt_exact_linear_register() == 0
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    creator = trt.get_plugin_registry().get_creator("PNMIRExactLinear", "1", "")
    assert creator is not None, "Exact Linear creator must be registered"

    def generated(shape, seed, scale=1.0):
        # Round CPU-generated values to FP32 once, independently of GPU RNGs.
        rng = random.Random(seed)
        values = [scale * rng.uniform(-1.0, 1.0) for _ in range(math.prod(shape))]
        return torch.tensor(values, dtype=torch.float32).reshape(shape).cuda()

    def raw(value):
        return value.detach().cpu().contiguous().numpy().tobytes()

    cases = (
        ("rows-one", (1, 32), 32),
        ("columns-one", (32, 32), 1),
        ("inner-one", (32, 1), 32),
        ("rows-columns-one", (1, 32), 1),
        ("rows-inner-one", (1, 1), 32),
        ("columns-inner-one", (32, 1), 1),
        ("all-one", (1, 1), 1),
        ("scalar-odd-inner", (1, 33, 31), 1),
        ("scalar-many-rows", (1, 257, 33), 1),
        ("scalar-four-dimensions", (2, 3, 5, 32), 1),
        ("lt-square-control", (1, 32, 32), 32),
        ("lt-odd-control", (3, 7, 31), 17),
    )
    results = []
    stream = torch.cuda.Stream()
    with torch.inference_mode(), torch.cuda.stream(stream):
        for index, (name, input_shape, columns) in enumerate(cases):
            x = generated(input_shape, 7100 + index)
            weight = generated((columns, input_shape[-1]), 8100 + index, 0.25)
            bias = generated((columns,), 9100 + index, 0.1)
            output_shape = input_shape[:-1] + (columns,)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
            )
            inputs = [
                network.add_input(key, trt.float32, tuple(value.shape))
                for key, value in (("input", x), ("weight", weight), ("bias", bias))
            ]
            plugin = creator.create_plugin(
                "linear", trt.PluginFieldCollection([]), trt.TensorRTPhase.BUILD
            )
            layer = network.add_plugin_v3(inputs, [], plugin)
            assert layer is not None, f"{name}: plugin layer must be created"
            output = layer.get_output(0)
            output.name = "output"
            network.mark_output(output)
            config = builder.create_builder_config()
            config.clear_flag(trt.BuilderFlag.TF32)
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
            plan = builder.build_serialized_network(network, config)
            assert plan is not None, f"{name}: engine must compile"
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(plan)
            assert engine is not None, f"{name}: engine must deserialize"
            context = engine.create_execution_context()
            assert context is not None, f"{name}: context must be created"
            actual = torch.empty(output_shape, device="cuda", dtype=torch.float32)
            # Reuse the context/output to catch stale bias or accumulated output.
            variants = (
                ("original", x, bias),
                ("zero-bias", x, torch.zeros_like(bias)),
                ("changed-input-bias", -x, -bias + 0.125),
                ("restored", x, bias),
            )
            for variant, value, offset in variants:
                case = root / name / variant
                case.mkdir(parents=True)
                bindings = (
                    ("input", value),
                    ("weight", weight),
                    ("bias", offset),
                    ("output", actual),
                )
                for key, tensor in bindings:
                    assert context.set_tensor_address(key, tensor.data_ptr())
                expected = torch.nn.functional.linear(value, weight, offset)
                assert context.execute_async_v3(
                    torch.cuda.current_stream().cuda_stream
                ), f"{name}/{variant}: enqueue must succeed"
                torch.cuda.synchronize()
                expected_bytes, actual_bytes = raw(expected), raw(actual)
                for key, tensor in bindings:
                    (case / f"{key}.bin").write_bytes(raw(tensor))
                (case / "expected.bin").write_bytes(expected_bytes)
                result = {
                    "case": f"{name}/{variant}",
                    "byte_identical": actual_bytes == expected_bytes,
                    "different_values": int((actual != expected).sum().item()),
                    "max_abs": float(
                        (actual.double() - expected.double()).abs().max().item()
                    ),
                }
                results.append(result)
                print(json.dumps(result), flush=True)
            del context, engine, runtime
    (root / "report.json").write_text(json.dumps(results, indent=2) + "\n")
    failures = [item["case"] for item in results if not item["byte_identical"]]
    assert not failures, "Exact Linear differs from eager CUDA bytes: " + ", ".join(
        failures
    )
    print(f"All {len(results)} exact Linear comparisons are byte-identical.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
