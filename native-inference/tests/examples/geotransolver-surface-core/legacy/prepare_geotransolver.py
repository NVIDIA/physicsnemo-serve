"""Prepare explicit trusted GeoTransolver weights and verification fixtures."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys
import tarfile
import zipfile


def _source_snapshot(path):
    source = path.read_bytes()
    return source, {
        "path": str(path),
        "sha256": hashlib.sha256(source).hexdigest(),
        "size_bytes": len(source),
    }


TEMPLATE = Path(__file__).resolve().parent


def _adapter(path=None):
    path = Path(path or TEMPLATE / "adapter.py").resolve()
    source, identity = _source_snapshot(path)
    spec = importlib.util.spec_from_file_location(
        "_pnmir_geotransolver_preparation_adapter", path
    )
    module = importlib.util.module_from_spec(spec)
    # Execute the same captured bytes that will be inventoried and emitted;
    # asking the loader to reread the mutable source could select other code.
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module, source, identity


def _identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_portable_references(output, recipe, cases, report):
    directory = output / "references"
    directory.mkdir()
    manifest = {
        "format_version": 1,
        "reference_kind": "upstream-full-model",
        "byte_order": "little",
        "checkpoint": report["checkpoint"],
        "model_state_sha256": report["model_state_sha256"],
        "source": report["source"],
        "comparisons": report["comparisons"],
        "prepared_files": list(report["files"]),
        "cases": [],
    }
    for index, case in enumerate(cases):
        case_dir = directory / f"case-{index}"
        case_dir.mkdir()
        record = {"case": index, "inputs": [], "outputs": []}
        for kind, values in (
            ("inputs", case["inputs"]),
            ("outputs", (case["expected"],)),
        ):
            for number, (specification, value) in enumerate(
                zip(recipe[kind], values, strict=True)
            ):
                path = case_dir / f"{kind}-{number}.bin"
                path.write_bytes(
                    value.detach()
                    .cpu()
                    .contiguous()
                    .numpy()
                    .astype("<f4", copy=False)
                    .tobytes()
                )
                identity = _identity(path) | {"path": str(path.relative_to(output))}
                record[kind].append({**specification, **identity})
                report["files"].append(identity)
        manifest["cases"].append(record)
    path = directory / "manifest.json"
    _json(path, manifest)
    report["reference_manifest"] = _identity(path) | {
        "path": str(path.relative_to(output))
    }
    report["files"].append(report["reference_manifest"])


def _archive_metadata(path):
    """Read metadata members without extracting the trusted checkpoint archive."""
    names = ("args.json", "metadata.json")
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for name in names:
                if archive.getinfo(name).file_size > 1024 * 1024:
                    raise ValueError("checkpoint JSON metadata is unexpectedly large")
            return tuple(json.loads(archive.read(name)) for name in names)
    with tarfile.open(path) as archive:
        values = []
        for name in names:
            member = archive.getmember(name)
            if not member.isfile() or member.size > 1024 * 1024:
                raise ValueError(
                    "checkpoint JSON metadata must be a small regular file"
                )
            with archive.extractfile(member) as handle:
                values.append(json.load(handle))
        return tuple(values)


def _full_inputs(points, geometry_points, case_index, device):
    import torch

    theta = torch.arange(points, dtype=torch.float32, device=device) * (
        2.0 * math.pi / points
    )
    positions = torch.stack(
        (
            0.5 * torch.cos(theta),
            0.25 * torch.sin(theta),
            torch.linspace(-0.5, 0.5, points, device=device),
        ),
        dim=-1,
    ).unsqueeze(0)
    normals = torch.stack(
        (torch.cos(theta), torch.sin(theta), torch.zeros_like(theta)), dim=-1
    ).unsqueeze(0)
    geometry_theta = torch.arange(
        geometry_points, dtype=torch.float32, device=device
    ) * (2.0 * math.pi / geometry_points)
    geometry = torch.stack(
        (
            0.6 * torch.cos(geometry_theta),
            0.3 * torch.sin(geometry_theta),
            torch.linspace(-0.6, 0.6, geometry_points, device=device),
        ),
        dim=-1,
    ).unsqueeze(0)
    if case_index == 2:
        positions = positions * 0.85 + torch.tensor([0.03, -0.02, 0.01], device=device)
        geometry = geometry * 1.15 + torch.tensor([-0.02, 0.01, 0.03], device=device)
    global_embedding = torch.tensor(
        [[[1.1, 35.0] if case_index == 1 else [1.205, 30.0]]],
        dtype=torch.float32,
        device=device,
    )
    return (
        torch.cat((positions, normals), dim=-1),
        positions,
        global_embedding,
        geometry,
    )


def _cached_inputs(model, values):
    import torch

    embedding, positions, global_embedding, geometry = values
    context, local_features, _ = model.context_builder.build_context(
        (embedding,), (positions,), geometry, global_embedding
    )
    global_context = model.context_builder.global_tokenizer(global_embedding)
    width = global_context.shape[-1]
    if (
        width <= 0
        or width >= context.shape[-1]
        or not torch.equal(context[..., -width:], global_context)
    ):
        raise ValueError(
            "upstream GeoTransolver context layout does not match the cached-core boundary"
        )
    return (
        embedding,
        local_features[0],
        context[..., :-width].contiguous(),
        global_embedding,
    )


def prepare(
    checkpoint,
    output,
    *,
    points=32,
    geometry_points=64,
    device="cuda",
    adapter_path=None,
):
    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    output = Path(output).expanduser().absolute()
    if not checkpoint.is_file() or checkpoint.suffix != ".mdlus":
        raise ValueError("--checkpoint must select an existing trusted .mdlus file")
    if output.exists():
        raise FileExistsError(f"preparation output already exists: {output}")
    if (
        type(points) is not int
        or type(geometry_points) is not int
        or points < 2
        or geometry_points < 2
    ):
        raise ValueError("query and geometry point counts must be at least two")
    import torch
    from physicsnemo import Module

    adapter, adapter_source, adapter_identity = _adapter(adapter_path)
    if metadata.version("nvidia-physicsnemo") != adapter.PHYSICSNEMO_VERSION:
        raise ValueError(
            f"preparation requires nvidia-physicsnemo=={adapter.PHYSICSNEMO_VERSION}"
        )
    target = torch.device(device)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "format_version": 1,
        "status": "preparing",
        "scope": "GeoTransolver surface cached-core parity on three deterministic model-space cases; not raw-mesh or scientific CFD qualification",
        "checkpoint": _identity(checkpoint),
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "physicsnemo": metadata.version("nvidia-physicsnemo"),
            "warp": metadata.version("warp-lang"),
            "device": str(target),
        },
        "source": {
            "adapter": adapter_identity,
            "preparation": _source_snapshot(Path(__file__).resolve())[1],
        },
        "comparisons": [],
    }
    _json(output / "preparation.json", report)
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        checkpoint_constructor, checkpoint_metadata = _archive_metadata(checkpoint)
        if (
            checkpoint_constructor.get("__name__") != "GeoTransolver"
            or checkpoint_constructor.get("__module__")
            != "physicsnemo.experimental.models.geotransolver.geotransolver"
        ):
            raise ValueError(
                "checkpoint must declare the pinned upstream GeoTransolver class"
            )
        report.update(
            checkpoint_constructor=checkpoint_constructor,
            checkpoint_metadata=checkpoint_metadata,
        )
        with torch.device("cpu"):
            official = Module.from_checkpoint(str(checkpoint), strict=True)
        official.eval()
        constructor = json.loads(json.dumps(official._args))
        config = {
            "format_version": 1,
            "physicsnemo_version": adapter.PHYSICSNEMO_VERSION,
            "model_import": adapter.MODEL_IMPORT,
            "model_args": checkpoint_constructor["__args__"],
        }
        state = {
            name: value.detach().cpu().clone()
            for name, value in official.state_dict().items()
        }
        if any(not bool(torch.isfinite(value).all()) for value in state.values()):
            raise ValueError("checkpoint contains nonfinite weights")
        digest = adapter.state_sha256(state)
        _json(output / "config.json", config)
        torch.save(state, output / "checkpoint.pt")
        official.to(target)
        cases = []
        with torch.inference_mode():
            for index in range(3):
                values = _full_inputs(points, geometry_points, index, target)
                cached = _cached_inputs(official, values)
                expected = official(*values)
                cases.append(
                    {
                        "inputs": tuple(
                            value.detach().cpu().contiguous().clone()
                            for value in cached
                        ),
                        "expected": expected.detach().cpu().contiguous().clone(),
                        "full_inputs": tuple(
                            value.detach().cpu().contiguous().clone()
                            for value in values
                        ),
                    }
                )
        if adapter.state_sha256(official.state_dict()) != digest:
            raise ValueError(
                "upstream model state changed during reference preparation"
            )
        fixtures = {
            "format_version": 1,
            "config": config,
            "state_sha256": digest,
            "cases": cases,
        }
        fixture_path = output / "fixtures.pt"
        torch.save(fixtures, fixture_path)
        core = adapter.create_model(config, {"fixtures": fixture_path})
        core.load_state_dict(state, strict=True)
        core.eval().to(target)
        with torch.inference_mode():
            for index, case in enumerate(cases):
                actual = core(*(value.to(target) for value in case["inputs"])).cpu()
                expected = case["expected"]
                difference = (actual - expected).abs()
                if not torch.equal(actual, expected):
                    raise ValueError(
                        f"cached-core/full-model parity differs for case {index}: max_abs={float(difference.max())}"
                    )
                report["comparisons"].append(
                    {
                        "case": index,
                        "passed": True,
                        "max_abs": float(difference.max()),
                        "bitwise_equal": torch.equal(
                            actual.contiguous().view(torch.uint8),
                            expected.contiguous().view(torch.uint8),
                        ),
                    }
                )
        if _identity(checkpoint) != report["checkpoint"]:
            raise ValueError("checkpoint changed during preparation")
        if _identity(Path(__file__).resolve()) != report["source"]["preparation"]:
            raise ValueError("preparation source changed during preparation")
        emitted_adapter = output / "adapter.py"
        emitted_adapter.write_bytes(adapter_source)
        emitted_identity = _identity(emitted_adapter)
        if any(
            emitted_identity[key] != adapter_identity[key]
            for key in ("sha256", "size_bytes")
        ):
            raise ValueError(
                "emitted adapter differs from the verified source snapshot"
            )

        def descriptor(name):
            return {"path": name, "sha256": _identity(output / name)["sha256"]}

        recipe = {
            "format_version": 2,
            "name": "geotransolver-surface-core",
            "version": "0.501",
            "adapter": "adapter.py",
            "factory": "create_model",
            "cases": "create_cases",
            "supported_backends": ["aoti", "tensorrt"],
            "default_backend": "aoti",
            "aoti_profile": "aten-boundary-exact-v2",
            "config": descriptor("config.json"),
            "checkpoint": {"format": "torch-state-dict", **descriptor("checkpoint.pt")},
            "assets": {"fixtures": descriptor("fixtures.pt")},
            "inputs": [
                {"name": name, "dtype": "float32", "shape": list(value.shape)}
                for name, value in zip(
                    (
                        "local_embedding",
                        "local_features",
                        "static_context",
                        "global_embedding",
                    ),
                    cases[0]["inputs"],
                    strict=True,
                )
            ],
            "outputs": [
                {
                    "name": "surface_fields_standardized",
                    "dtype": "float32",
                    "shape": list(cases[0]["expected"].shape),
                }
            ],
        }
        _json(output / "recipe.json", recipe)
        report.update(
            status="complete",
            upstream_constructor=constructor,
            model_state_sha256=digest,
            point_count=points,
            geometry_point_count=geometry_points,
        )
        report["files"] = [
            _identity(output / name) | {"path": name}
            for name in (
                "recipe.json",
                "adapter.py",
                "config.json",
                "checkpoint.pt",
                "fixtures.pt",
            )
        ]
        _write_portable_references(output, recipe, cases, report)
        _json(output / "preparation.json", report)
        return report
    except BaseException as error:
        report.update(
            status="failed", error={"type": type(error).__name__, "message": str(error)}
        )
        _json(output / "preparation.json", report)
        raise
    finally:
        torch.set_float32_matmul_precision(previous_precision)


def prepare_project(checkpoint, output, **options):
    """Stage the two-file template and verified data in a fresh working project."""
    output = Path(output).expanduser().absolute()
    document = (TEMPLATE / "model-build.json").read_bytes()
    output.mkdir(parents=True, exist_ok=False)
    report = prepare(checkpoint, output / "prepared", **options)
    # Use the adapter snapshot that actually passed preparation, rather than
    # rereading a source file that could have changed while the model ran.
    (output / "adapter.py").write_bytes((output / "prepared/adapter.py").read_bytes())
    (output / "model-build.json").write_bytes(document)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Create a fresh GeoTransolver model project from a trusted local .mdlus checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--points", default=32, type=int)
    parser.add_argument("--geometry-points", default=64, type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    try:
        report = prepare_project(
            args.checkpoint,
            args.output,
            points=args.points,
            geometry_points=args.geometry_points,
            device=args.device,
        )
        print(json.dumps(report, indent=2))
        return 0
    except Exception as error:
        print(f"GeoTransolver preparation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
