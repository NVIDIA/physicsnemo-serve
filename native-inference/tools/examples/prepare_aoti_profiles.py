"""Prepare a small compiler-profile example in a new directory outside source."""

import argparse
import json
from pathlib import Path
import runpy
import shutil

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a new directory; existing files are preserved")
    source = Path(__file__).resolve().parents[2] / "examples/aoti-profiles"
    create_model = runpy.run_path(str(source / "adapter.py"))["create_model"]
    document = json.loads((source / "model-build.json").read_text())
    generator = torch.Generator().manual_seed(20260917)
    model = create_model(document["config"], {})
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("bias"):
                parameter.fill_(0.5)
            else:
                parameter.copy_(
                    torch.randint(-8, 9, parameter.shape, generator=generator)
                    .to(torch.float32)
                    .div_(128)
                )
    cases = [
        (
            torch.randint(
                -16, 17, (16, document["config"]["input_dim"]), generator=generator
            )
            .to(torch.float32)
            .div_(16),
        )
        for _ in range(3)
    ]
    args.output.mkdir(parents=True)
    for name in ("model-build.json", "adapter.py"):
        shutil.copyfile(source / name, args.output / name)
    torch.save(model.state_dict(), args.output / "weights.pt")
    torch.save(cases, args.output / "cases.pt")
    (args.output / "features.bin").write_bytes(
        cases[0][0].numpy().astype("<f4", copy=False).tobytes()
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
