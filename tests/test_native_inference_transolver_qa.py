"""CPU contracts for the independent Transolver QA reference and asset boundary."""

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import types

import numpy as np
import pytest

from qa.native_inference import reference_transolver as reference
from qa.native_inference import transolver


def assets_fixture(tmp_path):
    entries = {}
    for name, filename in (
        ("checkpoint", "model.mdlus"),
        ("stats", "stats.json"),
        ("vtp", "surface.vtp"),
        ("stl", "body.stl"),
    ):
        path = tmp_path / filename
        path.write_bytes(f"trusted {name}".encode())
        entries[name] = {
            "path": filename,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest = tmp_path / "assets.json"
    manifest.write_text(
        json.dumps(
            {"format_version": 1, "revision": "pinned-model-revision", **entries}
        )
    )
    return manifest, entries


PLUGIN_NAMES = (
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


def installed_runtime(tmp_path):
    runtime = tmp_path / "sdk/bin/physicsnemo-infer"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("fixture runtime")
    libraries = runtime.parent.parent / "lib"
    libraries.mkdir()
    for name in PLUGIN_NAMES:
        (libraries / f"libpnmir_tensorrt_{name}_plugin.so").write_bytes(name.encode())
    return runtime


def test_asset_manifest_resolves_local_paths_and_requires_verified_bytes(tmp_path):
    manifest, entries = assets_fixture(tmp_path)
    result = transolver.load_assets(manifest)
    assert result["revision"] == "pinned-model-revision"
    assert result["checkpoint"]["path"] == str(tmp_path / "model.mdlus")
    assert result["stl"]["sha256"] == entries["stl"]["sha256"]
    (tmp_path / "body.stl").write_bytes(b"new geometry")
    with pytest.raises(ValueError, match="stl: SHA256 mismatch"):
        transolver.verify_assets(result)


@pytest.mark.parametrize("automatic", [True, False])
def test_build_prepares_downloaded_or_explicit_assets_within_asset_case(
    tmp_path, monkeypatch, automatic
):
    manifest, _ = assets_fixture(tmp_path)
    template = tmp_path / "native/examples/transolver-surface"
    template.mkdir(parents=True)
    (template / "adapter.py").write_text("POINT_COUNT = 75\n")
    (template / "model-build.json").write_text(
        (
            Path(__file__).resolve().parents[1]
            / "native-inference/examples/transolver-surface/model-build.json"
        ).read_text()
    )
    cases, downloads = [], []

    def case(name, callback):
        cases.append(name)
        return callback()

    def download(cache_root):
        assert cases == ["transolver.assets"]
        downloads.append(cache_root)
        return manifest

    monkeypatch.setattr(transolver, "prepare_assets", download)
    ctx = SimpleNamespace(
        root=tmp_path / "custom/shared/native-inference/run",
        native_root=tmp_path / "native",
        builder="builder",
        runtime=installed_runtime(tmp_path),
        case=case,
        run=lambda *args: SimpleNamespace(stdout='{"status":"imported"}'),
        build_project=lambda name, *args: {"name": name},
    )
    records = transolver.prepare_and_build(ctx, None if automatic else manifest)
    assert downloads == (
        [tmp_path / "custom/shared/assets/transolver"] if automatic else []
    )
    assert cases == [
        "transolver.assets",
        "transolver.import",
        "transolver.build75",
        "transolver.build11",
    ]
    assert [record["point_count"] for record in records] == [75, 11]
    assert records[0]["assets"]["checkpoint"]["path"] == str(tmp_path / "model.mdlus")


def test_explicit_invalid_manifest_never_falls_back_to_download(tmp_path, monkeypatch):
    manifest, _ = assets_fixture(tmp_path)
    (tmp_path / "body.stl").write_bytes(b"changed")
    monkeypatch.setattr(
        transolver,
        "prepare_assets",
        lambda *args: pytest.fail("explicit assets must remain offline"),
    )
    ctx = SimpleNamespace(case=lambda name, callback: callback())
    with pytest.raises(ValueError, match="stl: SHA256 mismatch"):
        transolver.prepare_and_build(ctx, manifest)


@pytest.mark.parametrize("missing", transolver.ASSET_NAMES)
def test_missing_hash_is_not_an_unpinned_download_request(tmp_path, missing):
    manifest, entries = assets_fixture(tmp_path)
    del entries[missing]["sha256"]
    manifest.write_text(json.dumps(entries))
    with pytest.raises(ValueError, match="SHA256"):
        transolver.load_assets(manifest)


def test_plain_tensor_checkpoint_is_not_a_valid_upstream_reference(tmp_path):
    manifest, entries = assets_fixture(tmp_path)
    (tmp_path / "model.mdlus").rename(tmp_path / "converted.pt")
    entries["checkpoint"]["path"] = "converted.pt"
    manifest.write_text(json.dumps(entries))
    with pytest.raises(ValueError, match="original trusted .mdlus"):
        transolver.load_assets(manifest)


def test_empty_and_remote_asset_paths_fail_closed(tmp_path):
    manifest, entries = assets_fixture(tmp_path)
    entries["vtp"]["path"] = "https://example.invalid/huge.vtp"
    manifest.write_text(json.dumps(entries))
    with pytest.raises(ValueError, match="local asset is missing"):
        transolver.load_assets(manifest)


def test_block_plan_reuses_full_package_and_routes_tail():
    assert reference.block_plan(75, 75) == [75]
    assert reference.block_plan(161, 75) == [75, 75, 11]


@pytest.mark.parametrize("count,size", [(1, 75), (75, 1), (151, 75), (76, 75)])
def test_block_plan_rejects_unqualified_single_point_blocks(count, size):
    with pytest.raises(ValueError, match="at least two|one.point"):
        reference.block_plan(count, size)


def test_surface_statistics_channel_order_and_physical_units(tmp_path):
    path = tmp_path / "statistics.json"
    path.write_text(
        json.dumps(
            {
                "mean": {"pressure": [1, 999], "shear_stress": [2, 3, 4]},
                "std_dev": {"pressure": [2], "shear_stress": [3, 4, 5]},
            }
        )
    )
    mean, std = reference.surface_statistics(path)
    standardized = np.array([[[1, 2, 3, 4]]], dtype=np.float32)
    decoded = reference.decode_surface(standardized, mean, std, density=2, velocity=3)
    np.testing.assert_array_equal(
        decoded, np.array([[[54, 144, 270, 432]]], dtype=np.float32)
    )
    assert decoded.dtype == np.float32


def test_invalid_statistics_cannot_produce_passing_reference(tmp_path):
    path = tmp_path / "statistics.json"
    path.write_text(
        json.dumps(
            {
                "mean": {"pressure": [0], "shear_stress": [0, 0, 0]},
                "std": {"pressure": [0], "shear_stress": [1, 1, 1]},
            }
        )
    )
    with pytest.raises(ValueError, match="positive"):
        reference.surface_statistics(path)


def test_eager_reference_restores_source_order_for_reused_and_tail_batches():
    torch = pytest.importorskip("torch")

    class BatchSensitiveModel:
        def __init__(self):
            self.sizes = []

        def __call__(self, fx, embedding):
            self.sizes.append(embedding.shape[1])
            assert fx.is_contiguous() and embedding.is_contiguous()
            return embedding[..., :4] + embedding[..., :1].mean(dim=1, keepdim=True)

    count, seed = 161, 17
    embedding = (
        torch.arange(count, dtype=torch.float32)
        .reshape(1, count, 1)
        .expand(1, count, 6)
    )
    fx = torch.tensor([1.205, 30.0]).reshape(1, 1, 2).expand(1, count, 2)
    mean, std = (
        np.array([1, 2, 3, 4], dtype=np.float32),
        np.array([2, 3, 4, 5], dtype=np.float32),
    )
    model = BatchSensitiveModel()
    result = reference.eager_surface(
        model, fx, embedding, mean, std, seed=seed, density=2, velocity=3
    )
    assert model.sizes == [75, 75, 11]
    permutation = torch.randperm(
        count, generator=torch.Generator().manual_seed(seed)
    ).numpy()
    expected = np.empty((1, count, 4), dtype=np.float32)
    for indices in (permutation[:75], permutation[75:150], permutation[150:]):
        expected[0, indices] = (
            indices.astype(np.float32) + np.mean(indices.astype(np.float32))
        )[:, None]
    np.testing.assert_array_equal(result["standardized_output"], expected)
    np.testing.assert_array_equal(
        result["physical_output"], (expected * std + mean) * np.float32(18)
    )
    np.testing.assert_array_equal(result["embedding"], embedding.numpy())


def test_eager_reference_rejects_wrong_model_shape():
    torch = pytest.importorskip("torch")
    fx = torch.ones((1, 11, 2), dtype=torch.float32)
    embedding = torch.ones((1, 11, 6), dtype=torch.float32)
    with pytest.raises(ValueError, match="incompatible surface output"):
        reference.eager_surface(
            lambda a, b: torch.ones((1, 11, 5)), fx, embedding, np.zeros(4), np.ones(4)
        )


@pytest.mark.parametrize(
    "actual,expected", [([1000000.0625], [1000000.0]), ([1e-7], [0.0])]
)
def test_standardized_parity_requires_both_absolute_and_relative_limits(
    actual, expected
):
    with pytest.raises(ValueError, match="parity failed"):
        transolver.compare_tensor(
            np.array(actual, dtype=np.float32),
            np.array(expected, dtype=np.float32),
            name="standardized",
        )


def test_original_checkpoint_loader_preserves_upstream_model_and_weights(
    tmp_path, monkeypatch
):
    torch = pytest.importorskip("torch")

    class Transolver(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.25, -2.5]))

    model = Transolver()
    checkpoint = tmp_path / "original.mdlus"
    calls = []

    def from_checkpoint(path, *, strict):
        calls.append((path, strict))
        return model

    upstream = types.ModuleType("physicsnemo")
    upstream.Module = SimpleNamespace(from_checkpoint=from_checkpoint)
    implementation = types.ModuleType("physicsnemo.models.transolver")
    implementation.Transolver = Transolver
    monkeypatch.setitem(sys.modules, "physicsnemo", upstream)
    monkeypatch.setitem(sys.modules, "physicsnemo.models.transolver", implementation)
    actual = reference.load_original_model(checkpoint, device="cpu")
    assert actual is model and not actual.training
    assert calls == [(str(checkpoint), True)]
    torch.testing.assert_close(
        actual.weight, torch.tensor([1.25, -2.5]), rtol=0, atol=0
    )
    model.half()
    with pytest.raises(ValueError, match="preserve FP32"):
        reference.load_original_model(checkpoint, device="cpu")


def test_physical_tolerance_scales_each_channel_and_rejects_pressure_factor_error():
    standardized = np.ones((1, 75, 4), dtype=np.float32)
    mean = np.array([1, 0.01, 0.02, 0.03], dtype=np.float32)
    std = np.array([2, 0.01, 0.02, 0.03], dtype=np.float32)
    expected = reference.decode_surface(standardized, mean, std)
    close = reference.decode_surface(standardized + np.float32(5e-5), mean, std)
    metrics = transolver.compare_physical(close, expected, standardized, mean, std)
    assert metrics["pressure"]["max_abs"] > 1e-4
    assert metrics["pressure"]["max_abs_limit"] > metrics["wss_x"]["max_abs_limit"]
    with pytest.raises(ValueError, match="physical pressure"):
        transolver.compare_physical(
            expected * np.float32(0.5), expected, standardized, mean, std
        )


@pytest.mark.parametrize(
    "payload", [b"\0" * 15, np.array([0, 0, np.nan, 0], dtype="<f4").tobytes()]
)
def test_tensor_reader_rejects_truncation_and_nonfinite_values(tmp_path, payload):
    path = tmp_path / "output.f32"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="FP32 bytes|nonfinite"):
        transolver.read_tensor(path, 1, 4)


def test_preparation_imports_original_once_and_builds_distinct_static_shapes(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    manifest, _ = assets_fixture(assets)
    calls = []
    builds = []

    class Context:
        root = tmp_path / "run"
        native_root = Path(__file__).resolve().parents[1] / "native-inference"
        builder = "/tools/pnms-model-builder"
        runtime = installed_runtime(tmp_path)

        def case(self, name, fn):
            calls.append(name)
            return fn()

        def run(self, name, args, **kwargs):
            assert args[1:3] == ["import-checkpoint", str(assets / "model.mdlus")]
            weights = Path(args[args.index("--output") + 1])
            weights.mkdir()
            (weights / "checkpoint.pt").write_bytes(b"converted only for builder")
            (weights / "config.json").write_text("{}")
            return SimpleNamespace(stdout='{"status": "imported"}')

        def build_project(self, name, project, backends, output):
            points = int(name.removeprefix("transolver"))
            assert f"POINT_COUNT = {points}\n" in (project / "adapter.py").read_text()
            assert (
                project / "weights/checkpoint.pt"
            ).read_bytes() == b"converted only for builder"
            assert backends == ["aoti", "tensorrt"]
            for plugin in PLUGIN_NAMES:
                assert (
                    project
                    / "assets/tensorrt"
                    / f"libpnmir_tensorrt_{plugin}_plugin.so"
                ).read_bytes() == plugin.encode()
            builds.append(name)
            # Simulate the real builder publishing a lock after the 75-point build.
            assert not (project / "model-build.lock.json").exists()
            (project / "model-build.lock.json").write_text("{}")
            return {
                "name": name,
                "model_dir": str(output / "model"),
                "packages": {
                    backend: str(output / "model/backends" / backend)
                    for backend in ("aoti", "tensorrt")
                },
            }

    records = transolver.prepare_and_build(Context(), manifest)
    assert calls == [
        "transolver.assets",
        "transolver.import",
        "transolver.build75",
        "transolver.build11",
    ]
    assert builds == ["transolver75", "transolver11"]
    assert [record["point_count"] for record in records] == [75, 11]
    assert all(
        record["assets"]["checkpoint"]["path"] == str(assets / "model.mdlus")
        for record in records
    )


def test_consumer_fails_closed_when_a_shape_package_was_not_built():
    with pytest.raises(ValueError, match="exactly the 75- and 11-point"):
        transolver.run_consumers(None, [{"point_count": 75}])


def test_surface_reference_geometry_matches_hand_calculated_centers_and_normals(
    tmp_path,
):
    torch = pytest.importorskip("torch")
    vtk = pytest.importorskip("vtkmodules.all")
    from vtkmodules.util.numpy_support import numpy_to_vtk

    points = vtk.vtkPoints()
    for point in ((0, 0, 0), (3, 0, 0), (3, 3, 0), (0, 3, 0)):
        points.InsertNextPoint(*point)
    triangles = vtk.vtkCellArray()
    for indices in ((0, 1, 2), (0, 2, 3)):
        triangle = vtk.vtkTriangle()
        for local, index in enumerate(indices):
            triangle.GetPointIds().SetId(local, index)
        triangles.InsertNextCell(triangle)
    mesh = vtk.vtkPolyData()
    mesh.SetPoints(points)
    mesh.SetPolys(triangles)
    mesh.GetCellData().SetNormals(
        numpy_to_vtk(np.array([[0, 0, 2], [0, 0, 3]], dtype=np.float32))
    )
    vtp, stl = tmp_path / "surface.vtp", tmp_path / "body.stl"
    for writer, path in ((vtk.vtkXMLPolyDataWriter(), vtp), (vtk.vtkSTLWriter(), stl)):
        writer.SetFileName(str(path))
        writer.SetInputData(mesh)
        assert writer.Write() == 1
    fx, embedding, version = reference.prepare_surface(
        vtp, stl, 2, device="cpu", density=2, velocity=3
    )
    torch.testing.assert_close(
        fx, torch.tensor([[[2, 3], [2, 3]]], dtype=torch.float32), rtol=0, atol=0
    )
    expected = torch.tensor(
        [[[0.5 / 12, -0.5 / 4.5, 0, 0, 0, 1], [-0.5 / 12, 0.5 / 4.5, 0, 0, 0, 1]]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(embedding, expected, rtol=0, atol=1e-7)
    assert version == vtk.vtkVersion.GetVTKVersion()


def consumer_records(tmp_path):
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    manifest, _ = assets_fixture(asset_dir)
    assets = transolver.load_assets(manifest)
    records = []
    for points in (75, 11):
        model = tmp_path / f"model{points}"
        variants, packages = {}, {}
        for backend in ("aoti", "tensorrt"):
            package = model / "backends" / backend
            package.mkdir(parents=True)
            (package / "payload.bin").write_bytes(f"{backend}-{points}".encode())
            (package / "model.json").write_text(
                json.dumps(
                    {
                        "artifacts": [
                            {
                                "backend": backend,
                                "target": "cuda",
                                "path": "payload.bin",
                            }
                        ]
                    }
                )
            )
            variants[backend] = {
                "package": str(package.relative_to(model)),
                "files": [
                    {
                        "path": str(path.relative_to(model)),
                        "size_bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for path in package.iterdir()
                ],
            }
            packages[backend] = str(package)
        (model / "model-release.json").write_text(
            json.dumps({"format_version": 1, "variants": variants})
        )
        records.append(
            {
                "point_count": points,
                "assets": assets,
                "model_dir": str(model),
                "packages": packages,
            }
        )
    return records


def workflow_metadata(*, backend, count, packages, assets, output):
    return {
        "schema_version": 2,
        "domain": "surface",
        "backend": backend,
        "device": "cuda:0",
        "point_count": count,
        "point_limit": count,
        "block_size": 75,
        "block_count": 1 if count == 75 else 3,
        "permutation": "torch.cuda.randperm",
        "permutation_seed": 0,
        "air_density": 1.205,
        "stream_velocity": 30.0,
        "output_dtype": "float32",
        "output_shape": [1, count, 4],
        "mesh_reader": "vtk-xml-polydata",
        "vtk_version": "9.1.0",
        "mesh": assets["vtp"]["path"],
        "stl": assets["stl"]["path"],
        "standardized_output": str(output / "standardized.f32"),
        "physical_output": str(output / "physical.f32"),
        "packages": [str(path) for _, path in packages],
        "package_profiles": [
            {"path": str(path), "point_dimension": points} for points, path in packages
        ],
        "preparation_ms": 1.0,
        "inference_ms": 2.0,
    }


@pytest.mark.parametrize("backend", ["aoti", "tensorrt"])
@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "backend",
        "device",
        "output_shape",
        "packages",
        "package_profiles",
        "inference_ms",
    ],
)
def test_consumer_metadata_binds_actual_backend_shape_and_packages(
    tmp_path, backend, mutation
):
    records = consumer_records(tmp_path)
    assets = records[0]["assets"]
    packages = [
        (record["point_count"], Path(record["packages"][backend])) for record in records
    ]
    output = tmp_path / "out"
    metadata = workflow_metadata(
        backend=backend, count=161, packages=packages, assets=assets, output=output
    )
    mutations = {
        "backend": "tensorrt" if backend == "aoti" else "aoti",
        "device": "cpu",
        "output_shape": [1, 75, 4],
        "packages": [],
        "package_profiles": [],
        "inference_ms": float("nan"),
    }
    if mutation:
        metadata[mutation] = mutations[mutation]
        with pytest.raises(ValueError, match="metadata mismatch|invalid native timing"):
            transolver.validate_metadata(
                metadata,
                backend=backend,
                count=161,
                packages=packages,
                assets=assets,
                output=output,
                vtk_version="9.1.0",
            )
    else:
        transolver.validate_metadata(
            metadata,
            backend=backend,
            count=161,
            packages=packages,
            assets=assets,
            output=output,
            vtk_version="9.1.0",
        )


@pytest.mark.parametrize("mutation", ["missing", "swapped", "corrupt"])
def test_consumer_rejects_missing_mislabelled_or_corrupt_backend_before_inference(
    tmp_path, mutation
):
    records = consumer_records(tmp_path)
    record = records[0]
    if mutation == "missing":
        del record["packages"]["tensorrt"]
    elif mutation == "swapped":
        record["packages"]["tensorrt"] = record["packages"]["aoti"]
    else:
        (Path(record["packages"]["tensorrt"]) / "payload.bin").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="backend|artifact identity"):
        transolver.run_consumers(None, records)


def test_consumer_runs_cpp_for_both_backends_with_shared_python_reference(
    tmp_path, monkeypatch
):
    records = consumer_records(tmp_path)
    assets = records[0]["assets"]
    cases, invocations, prepared, eager, loaded = [], [], [], [], []
    mean, std = np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32)
    expected = {}
    for count in (75, 161):
        standardized = np.ones((1, count, 4), dtype=np.float32)
        expected[count] = {
            "fx": np.ones((1, count, 2), dtype=np.float32),
            "embedding": np.ones((1, count, 6), dtype=np.float32),
            "standardized_output": standardized,
            "physical_output": reference.decode_surface(standardized, mean, std),
        }
    monkeypatch.setattr(reference, "configure_determinism", lambda: None)
    monkeypatch.setattr(
        reference, "load_original_model", lambda path: loaded.append(path) or object()
    )
    monkeypatch.setattr(reference, "surface_statistics", lambda path: (mean, std))

    def prepare(vtp, stl, count):
        prepared.append(count)
        return expected[count]["fx"], expected[count]["embedding"], "9.1.0"

    def evaluate(model, fx, embedding, mean, std):
        count = fx.shape[1]
        eager.append(count)
        return expected[count]

    monkeypatch.setattr(reference, "prepare_surface", prepare)
    monkeypatch.setattr(reference, "eager_surface", evaluate)

    def run(name, args, *, cwd, check=True):
        backend = args[args.index("--backend") + 1]
        count = int(args[args.index("--point-limit") + 1])
        selected = [
            Path(args[index + 1])
            for index, value in enumerate(args)
            if value == "--package"
        ]
        points = [75] if count == 75 or not check else [75, 11]
        assert selected == [
            Path(records[index]["packages"][backend]) for index in range(len(points))
        ]
        assert Path(cwd).parent.name == backend
        assert backend in name
        invocations.append((backend, count, check, Path(cwd)))
        if not check:
            return SimpleNamespace(
                returncode=1, stderr="no package accepts a 11-point block"
            )
        output = Path(cwd)
        (output / "inputs").mkdir()
        for key, value in expected[count].items():
            filename = (
                output / "inputs" / f"{key}.f32"
                if key in ("fx", "embedding")
                else output
                / (
                    "standardized.f32"
                    if key == "standardized_output"
                    else "physical.f32"
                )
            )
            filename.write_bytes(value.tobytes())
        metadata = workflow_metadata(
            backend=backend,
            count=count,
            packages=list(zip(points, selected)),
            assets=assets,
            output=output,
        )
        (output / "metadata.json").write_text(json.dumps(metadata))
        return SimpleNamespace(returncode=0)

    def case(name, callback):
        cases.append(name)
        return callback()

    ctx = SimpleNamespace(
        root=tmp_path / "run", workflow="physicsnemo-transolver", run=run, case=case
    )
    transolver.run_consumers(ctx, records)
    assert cases == [
        f"transolver.{backend}.{operation}"
        for backend in ("aoti", "tensorrt")
        for operation in ("surface75", "surface161", "missing_tail")
    ]
    assert [(backend, count, check) for backend, count, check, _ in invocations] == [
        (backend, count, check)
        for backend in ("aoti", "tensorrt")
        for count, check in ((75, True), (161, True), (161, False))
    ]
    assert len({output for _, _, _, output in invocations}) == 6
    assert len(loaded) == 1 and prepared == [75, 161] and eager == [75, 161]


@pytest.mark.parametrize(
    "mutation", ["aoti_profile", "tensorrt_profile", "plugin_asset", "plugin_file"]
)
def test_build_rejects_weakened_profiles_or_missing_exact_plugin(tmp_path, mutation):
    manifest, _ = assets_fixture(tmp_path)
    template = tmp_path / "native/examples/transolver-surface"
    template.mkdir(parents=True)
    (template / "adapter.py").write_text("POINT_COUNT = 75\n")
    config = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "native-inference/examples/transolver-surface/model-build.json"
        ).read_text()
    )
    runtime = installed_runtime(tmp_path)
    if mutation.endswith("profile"):
        config[mutation] = "baseline"
    elif mutation == "plugin_asset":
        del config["assets"]["tensorrt_exact_deslice_bmm_plugin"]
    else:
        (
            runtime.parent.parent / "lib/libpnmir_tensorrt_exact_deslice_bmm_plugin.so"
        ).unlink()
    (template / "model-build.json").write_text(json.dumps(config))
    cases = []

    def case(name, callback):
        cases.append(name)
        return callback()

    ctx = SimpleNamespace(
        root=tmp_path / "run",
        native_root=tmp_path / "native",
        runtime=runtime,
        case=case,
        run=lambda *args: pytest.fail("must reject before import/build"),
    )
    error, message = (
        (FileNotFoundError, "exact_deslice_bmm")
        if mutation == "plugin_file"
        else (ValueError, "exact-v2")
    )
    with pytest.raises(error, match=message):
        transolver.prepare_and_build(ctx, manifest)
    assert cases == ["transolver.assets", "transolver.import"]


@pytest.mark.parametrize("backend", ["aoti", "tensorrt"])
@pytest.mark.parametrize("failure", ["accepted", "published"])
def test_missing_tail_case_rejects_success_or_published_output_for_each_backend(
    tmp_path, backend, failure
):
    records = consumer_records(tmp_path)

    def run(name, args, *, cwd, check):
        assert not check
        assert args[args.index("--backend") + 1] == backend
        if failure == "published":
            (Path(cwd) / "metadata.json").write_text("{}")
        return SimpleNamespace(
            returncode=0 if failure == "accepted" else 1,
            stderr="no package accepts a 11-point block",
        )

    def case(name, callback):
        # Isolate the negative scenario without loading CUDA-dependent Python.
        if name == f"transolver.{backend}.missing_tail":
            return callback()

    ctx = SimpleNamespace(
        root=tmp_path / "run", workflow="physicsnemo-transolver", case=case, run=run
    )
    with pytest.raises(ValueError, match="did not reject|published a result"):
        transolver.run_consumers(ctx, records)
