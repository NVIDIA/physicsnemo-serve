"""Real-checkpoint, raw-surface QA through the public builder and native CLI."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import shutil

import numpy as np

if __package__:
    from . import reference_transolver as reference
    from .assets import prepare_assets
    from .contract import sha256, verify_model
else:
    import reference_transolver as reference
    from assets import prepare_assets
    from contract import sha256, verify_model


ASSET_NAMES = ("checkpoint", "stats", "vtp", "stl")
POINT_COUNTS = (75, 11)
BACKENDS = ("aoti", "tensorrt")
TENSORRT_PLUGINS = (
    "exact_linear",
    "exact_gemm",
    "exact_token_sum",
    "exact_slice_bmm",
    "exact_layer_norm",
    "exact_softmax",
    "exact_attention",
    "exact_gelu",
    "exact_deslice_bmm",
)
DENSITY = 1.205
VELOCITY = 30.0
SEED = 0


def verify_assets(assets: dict) -> dict:
    for name in ASSET_NAMES:
        item = assets.get(name)
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError(f"{name}: expected a local path and SHA256")
        if not isinstance(item.get("sha256"), str) or not re.fullmatch(
            r"[0-9a-fA-F]{64}", item["sha256"]
        ):
            raise ValueError(f"{name}: SHA256 must contain 64 hexadecimal characters")
        path = Path(item["path"])
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{name}: local asset is missing or empty: {path}")
        if sha256(path) != item["sha256"].lower():
            raise ValueError(f"{name}: SHA256 mismatch: {path}")
    if Path(assets["checkpoint"]["path"]).suffix.lower() != ".mdlus":
        raise ValueError("checkpoint must be the original trusted .mdlus archive")
    return assets


def load_assets(path: Path) -> dict:
    path = Path(path).resolve()
    document = json.loads(path.read_text())
    if not isinstance(document, dict) or document.get("format_version", 1) != 1:
        raise ValueError("assets require format_version 1")
    assets = {}
    for name in ASSET_NAMES:
        value = document.get(name)
        if not isinstance(value, dict) or not isinstance(value.get("path"), str):
            raise ValueError(f"{name}: expected a local path and SHA256")
        selected = Path(value["path"])
        if not selected.is_absolute():
            selected = path.parent / selected
        assets[name] = {"path": str(selected.resolve()), "sha256": value.get("sha256")}
    if "revision" in document:
        if not isinstance(document["revision"], str):
            raise ValueError("asset revision must be a string")
        assets["revision"] = document["revision"]
    return verify_assets(assets)


def prepare_and_build(ctx, assets_path) -> list[dict]:
    def prepare():
        manifest = assets_path
        if manifest is None:
            # Runs live at <mount>/native-inference/<run-id>; cache across runs.
            manifest = prepare_assets(
                Path(ctx.root).parent.parent / "assets/transolver"
            )
        return load_assets(manifest)

    assets = ctx.case("transolver.assets", prepare)
    projects = Path(ctx.root) / "projects"
    projects.mkdir(parents=True, exist_ok=True)
    project75 = projects / "transolver75"
    shutil.copytree(Path(ctx.native_root) / "examples/transolver-surface", project75)

    def import_checkpoint():
        # Capture the installed SDK's exact-profile plugins as builder assets.
        # The 11-point project inherits these same bytes from the 75-point copy.
        configuration = json.loads((project75 / "model-build.json").read_text())
        plugin_assets = {
            f"tensorrt_{name}_plugin": f"assets/tensorrt/libpnmir_tensorrt_{name}_plugin.so"
            for name in TENSORRT_PLUGINS
        }
        if (
            set(configuration.get("backends", [])) != set(BACKENDS)
            or configuration.get("aoti_profile") != "aten-boundary-exact-v2"
            or configuration.get("tensorrt_profile") != "layout-order-exact-v2"
            or configuration.get("assets") != plugin_assets
        ):
            raise ValueError(
                "Transolver QA requires both exact-v2 backend profiles and all nine TensorRT plugin assets"
            )
        runtime = Path(shutil.which(str(ctx.runtime)) or ctx.runtime).resolve()
        plugin_dir = project75 / "assets/tensorrt"
        plugin_dir.mkdir(parents=True, exist_ok=True)
        for name in TENSORRT_PLUGINS:
            filename = f"libpnmir_tensorrt_{name}_plugin.so"
            shutil.copyfile(
                runtime.parent.parent / "lib" / filename, plugin_dir / filename
            )
        result = ctx.run(
            "transolver-import",
            [
                str(ctx.builder),
                "import-checkpoint",
                assets["checkpoint"]["path"],
                "--project",
                str(project75),
                "--output",
                str(project75 / "weights"),
                "--json",
            ],
        )
        report = json.loads(result.stdout)
        if report.get("status") != "imported":
            raise ValueError("checkpoint import did not report imported status")
        return report

    ctx.case("transolver.import", import_checkpoint)
    records = []
    for points in POINT_COUNTS:
        project = projects / f"transolver{points}"
        if points != 75:
            shutil.copytree(project75, project)
            # Source the documented template; only its static point count differs.
            adapter = project / "adapter.py"
            text, replacements = re.subn(
                r"(?m)^POINT_COUNT = 75$",
                f"POINT_COUNT = {points}",
                adapter.read_text(),
            )
            if replacements != 1:
                raise ValueError(
                    "Transolver template no longer has the expected point-count setting"
                )
            adapter.write_text(text)
            # The copied 75-point build lock describes a different adapter.
            (project / "model-build.lock.json").unlink(missing_ok=True)
        record = ctx.case(
            f"transolver.build{points}",
            lambda points=points, project=project: ctx.build_project(
                f"transolver{points}",
                project,
                list(BACKENDS),
                Path(ctx.root) / "build" / f"transolver{points}",
            ),
        )
        record.update(point_count=points, assets=assets)
        records.append(record)
    return records


def read_tensor(path: Path, count: int, width: int) -> np.ndarray:
    payload = Path(path).read_bytes()
    if len(payload) != count * width * 4:
        raise ValueError(
            f"{path.name}: expected {count * width * 4} FP32 bytes, got {len(payload)}"
        )
    values = np.frombuffer(payload, dtype="<f4").reshape(1, count, width)
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name}: contains nonfinite values")
    return values


def compare_tensor(actual, expected, *, name, max_abs=1e-4, relative_l2=1e-4):
    if (
        actual.shape != expected.shape
        or actual.dtype != np.dtype("float32")
        or expected.dtype != np.dtype("float32")
    ):
        raise ValueError(f"{name}: tensor metadata mismatch")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError(f"{name}: contains nonfinite values")
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    maximum = float(np.max(np.abs(difference)))
    relative = float(
        np.linalg.norm(difference)
        / max(np.linalg.norm(expected.astype(np.float64)), np.finfo(np.float64).eps)
    )
    result = {
        "max_abs": maximum,
        "relative_l2": relative,
        "limits": {"max_abs": max_abs, "relative_l2": relative_l2},
        "bitwise_equal": actual.tobytes() == expected.tobytes(),
        "actual_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
        "reference_shape": list(expected.shape),
        "reference_dtype": "float32",
    }
    if maximum > max_abs or relative > relative_l2:
        raise ValueError(f"{name}: parity failed: {result}")
    return result


def compare_physical(actual, expected, standardized_reference, mean, std):
    if (
        actual.shape != expected.shape
        or actual.dtype != np.float32
        or expected.dtype != np.float32
    ):
        raise ValueError("physical output metadata mismatch")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("physical output contains nonfinite values")
    q = DENSITY * VELOCITY * VELOCITY
    results = {}
    for channel, name in enumerate(("pressure", "wss_x", "wss_y", "wss_z")):
        rounding_scale = max(
            1.0,
            float(np.abs(expected[..., channel]).max()),
            q * abs(float(mean[channel])),
            q
            * abs(float(std[channel]))
            * float(np.abs(standardized_reference[..., channel]).max()),
        )
        limit = (
            q * abs(float(std[channel])) * 1e-4
            + 8 * np.finfo(np.float32).eps * rounding_scale
        )
        difference = actual[..., channel].astype(np.float64) - expected[
            ..., channel
        ].astype(np.float64)
        maximum = float(np.abs(difference).max())
        if maximum > limit:
            raise ValueError(
                f"physical {name}: max_abs {maximum} exceeds channel-scaled limit {limit}"
            )
        results[name] = {"max_abs": maximum, "max_abs_limit": float(limit)}
    results.update(
        bitwise_equal=actual.tobytes() == expected.tobytes(),
        actual_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
        reference_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
        reference_shape=list(expected.shape),
        reference_dtype="float32",
    )
    return results


def validate_metadata(
    metadata, *, backend, count, packages, assets, output, vtk_version
):
    expected = {
        "schema_version": 2,
        "domain": "surface",
        "backend": backend,
        "device": "cuda:0",
        "point_count": count,
        "point_limit": count,
        "block_size": 75,
        "block_count": len(reference.block_plan(count, 75)),
        "permutation": "torch.cuda.randperm",
        "permutation_seed": SEED,
        "air_density": DENSITY,
        "stream_velocity": VELOCITY,
        "output_dtype": "float32",
        "output_shape": [1, count, 4],
        "mesh_reader": "vtk-xml-polydata",
        "vtk_version": vtk_version,
        "mesh": assets["vtp"]["path"],
        "stl": assets["stl"]["path"],
        "standardized_output": str(output / "standardized.f32"),
        "physical_output": str(output / "physical.f32"),
        "packages": [str(path) for _, path in packages],
        "package_profiles": [
            {"path": str(path), "point_dimension": points} for points, path in packages
        ],
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"native workflow metadata mismatch: {key}: {metadata.get(key)!r} != {value!r}"
            )
    for key in ("preparation_ms", "inference_ms"):
        value = metadata.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid native timing: {key}")


def workflow_command(ctx, assets, packages, count, output, *, backend):
    args = [str(ctx.workflow), "--backend", backend]
    for _, path in packages:
        args.extend(("--package", str(path)))
    args.extend(
        (
            "--mesh",
            assets["vtp"]["path"],
            "--stl",
            assets["stl"]["path"],
            "--stats",
            assets["stats"]["path"],
            "--domain",
            "surface",
            "--device",
            "cuda",
            "--point-limit",
            str(count),
            "--block-size",
            "75",
            "--seed",
            str(SEED),
            "--air-density",
            str(DENSITY),
            "--stream-velocity",
            str(VELOCITY),
            "--standardized-output",
            str(output / "standardized.f32"),
            "--physical-output",
            str(output / "physical.f32"),
            "--metadata",
            str(output / "metadata.json"),
            "--dump-input-dir",
            str(output / "inputs"),
        )
    )
    return args


def run_consumers(ctx, records):
    selected = {record["point_count"]: record for record in records}
    if set(selected) != set(POINT_COUNTS) or len(records) != 2:
        raise ValueError(
            "Transolver requires exactly the 75- and 11-point package records"
        )
    assets = selected[75]["assets"]
    if selected[11]["assets"] != assets:
        raise ValueError("Transolver packages refer to different source assets")
    for record in records:
        if set(record["packages"]) != set(BACKENDS):
            raise ValueError(
                "Transolver requires exactly both AOTI and TensorRT backends"
            )
        verified = verify_model(record["model_dir"], BACKENDS)
        if verified != {
            backend: str(Path(path).resolve())
            for backend, path in record["packages"].items()
        }:
            raise ValueError(
                "Transolver package paths do not match their release backends"
            )
    model = None
    references = {}

    def prepare_reference(count):
        nonlocal model
        if model is None:
            reference.configure_determinism()
            model = reference.load_original_model(Path(assets["checkpoint"]["path"]))
        mean, std = reference.surface_statistics(Path(assets["stats"]["path"]))
        fx, embedding, vtk_version = reference.prepare_surface(
            Path(assets["vtp"]["path"]), Path(assets["stl"]["path"]), count
        )
        expected = reference.eager_surface(model, fx, embedding, mean, std)
        reference_dir = (
            Path(ctx.root) / "consumer" / "transolver" / "reference" / f"surface{count}"
        ).resolve()
        reference_dir.mkdir(parents=True, exist_ok=False)
        for name, value in expected.items():
            (reference_dir / f"{name}.f32").write_bytes(
                value.astype("<f4", copy=False).tobytes()
            )
        return expected, mean, std, vtk_version, reference_dir

    def consume(backend, count):
        verify_assets(assets)
        if count not in references:
            references[count] = prepare_reference(count)
        expected, mean, std, vtk_version, reference_dir = references[count]
        output = (
            Path(ctx.root) / "consumer" / "transolver" / backend / f"surface{count}"
        ).resolve()
        output.mkdir(parents=True, exist_ok=False)
        points = (75,) if count == 75 else POINT_COUNTS
        packages = [
            (point, Path(selected[point]["packages"][backend]).resolve())
            for point in points
        ]
        ctx.run(
            f"transolver-{backend}-surface{count}",
            workflow_command(ctx, assets, packages, count, output, backend=backend),
            cwd=output,
        )
        metadata = json.loads((output / "metadata.json").read_text())
        validate_metadata(
            metadata,
            backend=backend,
            count=count,
            packages=packages,
            assets=assets,
            output=output,
            vtk_version=vtk_version,
        )
        comparisons = {}
        for name, width in (("fx", 2), ("embedding", 6), ("standardized_output", 4)):
            path = (
                output / "inputs" / f"{name}.f32"
                if name in ("fx", "embedding")
                else output / "standardized.f32"
            )
            actual = read_tensor(path, count, width)
            tolerance = 0.0 if name == "fx" else 1e-6 if name == "embedding" else 1e-4
            comparisons[name] = compare_tensor(
                actual,
                expected[name],
                name=name,
                max_abs=tolerance,
                relative_l2=tolerance,
            )
        physical = read_tensor(output / "physical.f32", count, 4)
        comparisons["physical_output"] = compare_physical(
            physical,
            expected["physical_output"],
            expected["standardized_output"],
            mean,
            std,
        )
        report = {
            "native_metadata": metadata,
            "assets": assets,
            "reference_kind": "original-checkpoint-eager-raw-geometry",
            "reference_directory": str(reference_dir),
            "reference_implementation_sha256": sha256(Path(reference.__file__)),
            "comparisons": comparisons,
        }
        (output / "comparison.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        return report

    def missing_tail(backend):
        output = (
            Path(ctx.root) / "consumer" / "transolver" / backend / "missing-tail"
        ).resolve()
        output.mkdir(parents=True, exist_ok=False)
        packages = [(75, Path(selected[75]["packages"][backend]).resolve())]
        result = ctx.run(
            f"transolver-{backend}-missing-tail",
            workflow_command(ctx, assets, packages, 161, output, backend=backend),
            cwd=output,
            check=False,
        )
        if (
            result.returncode == 0
            or "no package accepts a 11-point block" not in result.stderr
        ):
            raise ValueError(
                "missing-tail workflow did not reject its absent 11-point package"
            )
        if any(
            (output / filename).exists()
            for filename in ("standardized.f32", "physical.f32", "metadata.json")
        ):
            raise ValueError("rejected missing-tail workflow published a result")
        return {"returncode": result.returncode, "rejected_point_count": 11}

    for backend in BACKENDS:
        for count in (75, 161):
            ctx.case(
                f"transolver.{backend}.surface{count}",
                lambda backend=backend, count=count: consume(backend, count),
            )
        ctx.case(
            f"transolver.{backend}.missing_tail",
            lambda backend=backend: missing_tail(backend),
        )
