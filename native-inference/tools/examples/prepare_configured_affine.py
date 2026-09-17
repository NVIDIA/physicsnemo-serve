"""Stage the affine example and two checkpoints in a fresh output directory."""

import argparse
import json
from pathlib import Path
import shutil

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a new directory; existing files are preserved")
    source = Path(__file__).resolve().parents[2] / "examples/configured-affine"
    args.output.mkdir(parents=True)
    for name in ("model-build.json", "adapter.py"):
        shutil.copyfile(source / name, args.output / name)
    (args.output / "normalization.json").write_text(
        json.dumps({"mean": 1.0, "std": 2.0}, indent=2) + "\n"
    )
    for name, scale, bias in (
        ("checkpoint-a.pt", 2.0, 1.0),
        ("checkpoint-b.pt", -1.0, 0.5),
    ):
        destination = args.output / name
        torch.save(
            {
                "scale": torch.tensor(scale, dtype=torch.float32),
                "bias": torch.tensor(bias, dtype=torch.float32),
            },
            destination,
        )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
