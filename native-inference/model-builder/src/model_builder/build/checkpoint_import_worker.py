"""Standalone CPU checkpoint conversion; usable inside existing builder images."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import zipfile


def _identity(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "size_bytes": size}


def _verify_source(checkpoint, expected_sha256):
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise ValueError("Checkpoint must be a regular file, not a symlink.")
    identity = _identity(checkpoint)
    if identity["sha256"] != expected_sha256:
        raise ValueError("Checkpoint SHA256 changed; select the original input again.")
    return identity


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _source_state(checkpoint, torch):
    # Detect legacy tar first: it can contain a Torch ZIP that also matches ZIP.
    if tarfile.is_tarfile(checkpoint):
        with tarfile.open(checkpoint) as archive:
            members = [
                entry for entry in archive.getmembers() if entry.name == "model.pt"
            ]
            if len(members) != 1 or not members[0].isfile():
                raise ValueError("Checkpoint must contain one regular model.pt member.")
            with archive.extractfile(members[0]) as source:
                payload = source.read()
    elif zipfile.is_zipfile(checkpoint):
        with zipfile.ZipFile(checkpoint) as archive:
            if archive.namelist().count("model.pt") != 1:
                raise ValueError("Checkpoint must contain one model.pt member.")
            payload = archive.read("model.pt")
    else:
        raise ValueError("Expected a PhysicsNeMo ZIP or legacy tar checkpoint.")
    return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)


def _tensor_inventory(state, torch):
    if (
        not isinstance(state, dict)
        or not state
        or any(
            not isinstance(name, str) or not isinstance(value, torch.Tensor)
            for name, value in state.items()
        )
    ):
        raise ValueError("Checkpoint state must be a nonempty mapping of tensors.")
    inventory = Counter()
    for value in state.values():
        if value.layout != torch.strided or value.is_quantized:
            raise ValueError("Import requires dense, unquantized checkpoint tensors.")
        value = value.detach().cpu().resolve_conj().resolve_neg().contiguous()
        payload = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        inventory[
            (str(value.dtype), tuple(value.shape), hashlib.sha256(payload).hexdigest())
        ] += 1
    return inventory


def execute(checkpoint: Path, output: Path, expected_sha256: str) -> dict:
    checkpoint = Path(checkpoint).absolute()
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("Output already exists; choose a fresh directory.")
    identity = _verify_source(checkpoint, expected_sha256)
    # Host orchestration needs neither dependency; only this isolated worker does.
    try:
        import torch
        import physicsnemo
    except ImportError as exc:
        raise ValueError(
            "Checkpoint import requires Torch and PhysicsNeMo in the selected "
            "builder environment, including the checkpoint model's dependencies."
        ) from exc

    output.mkdir(parents=True, exist_ok=False)
    try:
        source_inventory = _tensor_inventory(_source_state(checkpoint, torch), torch)
        with torch.device("cpu"):
            model = physicsnemo.Module.from_checkpoint(str(checkpoint), strict=True)
            model = model.cpu().eval()
        try:
            arguments = model._args["__args__"]
            if not isinstance(arguments, dict) or not all(
                isinstance(key, str) for key in arguments
            ):
                raise TypeError("Constructor arguments must be a keyword dictionary.")
            config = json.loads(json.dumps(arguments, allow_nan=False))
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Checkpoint constructor arguments must be JSON-compatible keyword "
                "arguments. For nested modules or custom objects, use a custom "
                "adapter and explicit configuration."
            ) from exc

        original = model.state_dict()
        if not original or any(
            not isinstance(name, str) or not isinstance(value, torch.Tensor)
            for name, value in original.items()
        ):
            raise ValueError("Checkpoint state must be a nonempty mapping of tensors.")
        state = {
            name: value.detach().cpu().contiguous().clone()
            for name, value in original.items()
        }
        # PhysicsNeMo may rename legacy keys, but no tensor can be cast, replaced,
        # split or merged while claiming that original checkpoint weights survived.
        if _tensor_inventory(state, torch) != source_inventory:
            raise ValueError(
                "PhysicsNeMo loading changed original checkpoint tensors in dtype, "
                "shape, values or count. Key renames are supported; casts and other "
                "compatibility transformations require an explicit custom adapter."
            )
        checkpoint_path = output / "checkpoint.pt"
        config_path = output / "config.json"
        torch.save(state, checkpoint_path)
        _write_json(config_path, config)

        try:
            with torch.device("cpu"):
                restored = type(model)(**config).cpu().eval()
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            restored.load_state_dict(saved, strict=True)
        except Exception as exc:
            raise ValueError(
                "Cannot reconstruct and strictly reload this checkpoint using its "
                "JSON constructor arguments; provide a custom adapter/configuration. "
                f"Details: {exc}"
            ) from exc
        actual = restored.state_dict()
        if actual.keys() != state.keys() or saved.keys() != state.keys():
            raise ValueError("Strict reload changed the checkpoint tensor names.")
        for name, expected in state.items():
            for candidate in (saved[name], actual[name]):
                if (
                    candidate.dtype != expected.dtype
                    or candidate.shape != expected.shape
                    or not torch.equal(candidate, expected)
                ):
                    raise ValueError(
                        f"Reloaded tensor {name!r} differs in dtype, shape or values; "
                        "checkpoint import cannot silently cast or change weights."
                    )

        _verify_source(checkpoint, expected_sha256)
        report = {
            "format_version": 1,
            "status": "imported",
            "checkpoint": identity,
            "model": {"module": type(model).__module__, "name": type(model).__name__},
            "environment": {
                "torch": str(torch.__version__),
                "physicsnemo": str(getattr(physicsnemo, "__version__", "unknown")),
            },
            "verification": {
                "strict_reload": True,
                "tensor_equality": True,
                "source_tensors_unchanged": True,
            },
            "artifacts": {
                "checkpoint": {
                    "path": checkpoint_path.name,
                    **_identity(checkpoint_path),
                },
                "config": {"path": config_path.name, **_identity(config_path)},
            },
        }
        _write_json(output / "import.json", report)
        return report
    except BaseException:
        shutil.rmtree(output)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    args = parser.parse_args(argv)
    execute(args.checkpoint, args.output, args.checkpoint_sha256)


if __name__ == "__main__":
    main()
