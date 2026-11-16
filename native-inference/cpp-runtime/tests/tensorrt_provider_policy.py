"""GPU-free validation of the provider plugin creators and serialized policy."""

import argparse
import ctypes
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin-root", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import tensorrt as trt

    libraries = []
    for name, creator_name in (
        ("linear", "PNMIRExactLinear"),
        ("gemm", "PNMIRExactGemm"),
        ("slice_bmm", "PNMIRExactSliceBmm"),
        ("deslice_bmm", "PNMIRExactDesliceBmm"),
    ):
        libraries.append(
            ctypes.CDLL(
                str(args.plugin_root / f"libpnmir_tensorrt_exact_{name}_plugin.so"),
                mode=ctypes.RTLD_GLOBAL,
            )
        )
        registry = trt.get_plugin_registry()
        assert registry.get_creator(creator_name, "1", "") is not None
        creator = registry.get_creator(creator_name, "2", "")
        assert creator is not None, (
            f"{creator_name}: provider creator version 2 missing"
        )

        def fields(**overrides):
            values = dict(
                compute_mode=1,
                cublas_workspace_bytes=33554432,
                cublaslt_workspace_bytes=1048576,
            )
            values.update(overrides)
            return trt.PluginFieldCollection(
                [
                    trt.PluginField(
                        key,
                        np.array([value], dtype=np.int32),
                        trt.PluginFieldType.INT32,
                    )
                    for key, value in values.items()
                ]
            )

        for phase in (trt.TensorRTPhase.BUILD, trt.TensorRTPhase.RUNTIME):
            assert creator.create_plugin("valid", fields(), phase) is not None
            for invalid in (
                fields(compute_mode=0),
                fields(cublas_workspace_bytes=0),
                fields(cublaslt_workspace_bytes=0),
                trt.PluginFieldCollection([]),
            ):
                assert creator.create_plugin("invalid", invalid, phase) is None, (
                    f"{creator_name}: invalid serialized policy accepted"
                )
    print(json.dumps({"provider_creators": 4, "invalid_policy_rejected": True}))


if __name__ == "__main__":
    main()
