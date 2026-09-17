from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch._inductor.codecache


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--outputs", required=True, type=Path)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()

    device = torch.device(args.target)
    if device.type == "cuda":
        # Match the exporter and C++ runtime, including fresh NGC processes
        # whose environment enables TF32 by default.
        torch.set_float32_matmul_precision("highest")
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    inputs = tuple(
        value.to(device)
        for value in torch.load(args.inputs, map_location="cpu", weights_only=True)
    )
    compiled = torch._inductor.aoti_load_package(str(args.package))
    with torch.inference_mode():
        outputs = compiled(*inputs)
    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)
    elif isinstance(outputs, (tuple, list)) and all(
        isinstance(value, torch.Tensor) for value in outputs
    ):
        outputs = tuple(outputs)
    else:
        raise TypeError("compiled model outputs must be tensors")
    torch.save(tuple(value.detach().cpu() for value in outputs), args.outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
