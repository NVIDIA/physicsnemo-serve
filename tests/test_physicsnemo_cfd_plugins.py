# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import yaml
from jsonschema import validate

from physicsnemo_cfd_runtime import safe_files as cfd_safe_files
from physicsnemo_cfd_runtime import supervisor as cfd_supervisor
from plugin_sdk import (
    ExecutionContext,
    OutputRegistry,
    PluginCancelledError,
    PrepareContext,
    RawRequest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = REPO_ROOT / "plugins" / "physicsnemo-cfd-surface-benchmark"
MANIFEST_PATH = PLUGIN_ROOT / "plugin.yaml"
FIXTURE_PATH = PLUGIN_ROOT / "fixtures" / "tiny_surface.vtp"
PUBLIC_E2E_REQUEST_PATH = PLUGIN_ROOT / "examples" / "public_run_1_request.json"
TINY_STL_BYTES = b"solid drivaer\nendsolid drivaer\n"


def _load_surface_implementation():
    module_name = "physicsnemo_cfd_surface_benchmark_test_impl"
    spec = importlib.util.spec_from_file_location(
        module_name,
        PLUGIN_ROOT / "surface_benchmark_impl.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


surface = _load_surface_implementation()


def _config() -> dict:
    return surface.load_surface_plugin_config(MANIFEST_PATH)


def _case_fields(**overrides) -> dict:
    content = FIXTURE_PATH.read_bytes()
    geometry_content = TINY_STL_BYTES
    fields = {
        "case_id": "run_1",
        "mesh_uri": "https://assets.example.com/run_1/boundary_1.vtp",
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
        "geometry_uri": "https://assets.example.com/run_1/drivaer_1.stl",
        "geometry_sha256": hashlib.sha256(geometry_content).hexdigest(),
        "geometry_size_bytes": len(geometry_content),
    }
    fields.update(overrides)
    return fields


def _request_fields(**overrides) -> dict:
    fields = {
        "models": ["domino_surface"],
        "cases": [_case_fields()],
        "metrics": ["l2_pressure", "drag"],
        "seed": 42,
        "save_inference_mesh": False,
        "visual_case_ids": [],
    }
    fields.update(overrides)
    return fields


def _inputs(**overrides) -> surface.SurfaceBenchmarkInput:
    return surface.normalize_surface_request(_request_fields(**overrides), _config())


def _prefetched(
    source: Path,
    inputs: surface.SurfaceBenchmarkInput,
    *,
    geometry_source: Path | None = None,
) -> list[dict]:
    case = inputs.cases[0]
    if geometry_source is None:
        geometry_source = source.with_suffix(".stl")
        geometry_source.write_bytes(TINY_STL_BYTES)
    return [
        {
            "name": f"surface-mesh-{case.case_id}",
            "source_uri": case.mesh_uri,
            "storage_path": str(source),
            "size_bytes": case.size_bytes,
            "sha256": case.sha256,
            "media_type": "application/vnd.vtk.vtp",
        },
        {
            "name": f"surface-geometry-{case.case_id}",
            "source_uri": case.geometry_uri,
            "storage_path": str(geometry_source),
            "size_bytes": case.geometry_size_bytes,
            "sha256": case.geometry_sha256,
            "media_type": "model/stl",
        },
    ]


def test_manifest_contains_complete_pinned_surface_contract():
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    configuration = manifest["configuration"]
    provider = configuration["provider"]
    benchmark = configuration["benchmark"]

    assert provider == {
        "repository": "https://github.com/NVIDIA/physicsnemo-cfd.git",
        "tag": "v0.0.2",
        "version": "0.0.2",
        "commit": "921f14dc2ac14c04aabffaba3290deb792379dd8",
        "physicsnemo_version": "2.1.1",
        "python_version": "3.12",
    }
    assert benchmark["module_argv"] == [
        "{python}",
        "-m",
        "physicsnemo.cfd.evaluation.benchmarks.run",
        "--config",
        "{resolved_config}",
    ]
    assert benchmark["dataset"]["layout"] == "run_{number}/boundary_{number}.vtp"
    assert benchmark["execution"]["timeout_seconds"] == 21600
    assert benchmark["execution"]["sequential"] is True
    assert benchmark["execution"]["batch_size"] == 1
    assert benchmark["benchmark_config"]["reproducibility"]["log_env"] is False
    package_uris = {model["package"] for model in benchmark["models"].values()}
    assert len(package_uris) == 5
    for package_uri in package_uris:
        prefix, separator, revision = package_uri.rpartition("@")
        assert separator == "@"
        assert prefix.startswith("hf://nvidia/")
        assert len(revision) == 40
        assert all(character in "0123456789abcdef" for character in revision)
    assert manifest["runtime"]["executor_class"] == "physicsnemo-cfd-gpu"
    assert manifest["runtime"]["readiness_executor_class"] == "physicsnemo-cfd-gpu"
    assert "prepare_executor_class" not in manifest["runtime"]
    assert manifest["developer"]["readiness"]["python_modules"] == [
        "physicsnemo_cfd_runtime",
        "physicsnemo_cfd_runtime.artifacts",
        "yaml",
        "physicsnemo.cfd.evaluation.benchmarks.run",
    ]
    assert "batch_size" not in manifest["ingress"]["json_schema_inline"]["properties"]
    request_properties = manifest["ingress"]["json_schema_inline"]["properties"]
    assert (
        request_properties["cases"]["items"]["properties"]["case_id"]["maxLength"] == 64
    )
    assert request_properties["visual_case_ids"]["items"]["maxLength"] == 64
    result_properties = manifest["outputs"]["result_schema_inline"]["properties"]
    assert result_properties["case_ids"]["items"]["maxLength"] == 64
    assert (
        result_properties["case_digests"]["items"]["properties"]["case_id"]["maxLength"]
        == 64
    )
    assert benchmark["limits"]["max_case_id_length"] == 64

    provider_lock = json.loads(
        (PLUGIN_ROOT / "provider.lock.json").read_text(encoding="utf-8")
    )
    for key, value in provider.items():
        assert provider_lock[key] == value


def test_public_run_1_e2e_request_is_schema_valid_and_content_pinned():
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    request = json.loads(PUBLIC_E2E_REQUEST_PATH.read_text(encoding="utf-8"))

    validate(request, manifest["ingress"]["json_schema_inline"])
    inputs = surface.normalize_surface_request(request, _config())

    assert inputs.models == ["domino_surface"]
    assert inputs.metrics == ["l2_pressure"]
    assert len(inputs.cases) == 1
    assert dataclasses.asdict(inputs.cases[0]) == {
        "case_id": "run_1",
        "mesh_uri": (
            "https://huggingface.co/datasets/neashton/drivaerml/resolve/"
            "f26d75e0d3dee10ba0e42829bafd0e0d95ca5acc/run_1/boundary_1.vtp"
        ),
        "sha256": "01d388402dad7a783db9c666ddb18e6db745aac16a3193c275e0726dd108bb40",
        "size_bytes": 659606189,
        "geometry_uri": (
            "https://huggingface.co/datasets/neashton/drivaerml/resolve/"
            "f26d75e0d3dee10ba0e42829bafd0e0d95ca5acc/run_1/drivaer_1.stl"
        ),
        "geometry_sha256": "411e6651284a26fc94924106b833fd79febc6deba63922c929dd8acfc99720d2",
        "geometry_size_bytes": 142385186,
    }

    workflow = surface.SurfaceBenchmarkWorkflow(MANIFEST_PATH)
    prepared = workflow.prepare(
        RawRequest(
            content_type="application/json",
            operation="run",
            raw_fields=request,
            input_artifacts=[],
        ),
        PrepareContext(
            run_id="public-run-1-contract",
            workflow_id="physicsnemo-cfd-surface-benchmark",
            run_dir=PLUGIN_ROOT,
        ),
    )
    assert prepared.prefetch_plan == [
        {
            "kind": "http_fetch",
            "source_uri": inputs.cases[0].mesh_uri,
            "target_artifact_name": "surface-mesh-run_1",
            "required": True,
            "expected_sha256": inputs.cases[0].sha256,
            "expected_size_bytes": inputs.cases[0].size_bytes,
            "media_type": "application/vnd.vtk.vtp",
        },
        {
            "kind": "http_fetch",
            "source_uri": inputs.cases[0].geometry_uri,
            "target_artifact_name": "surface-geometry-run_1",
            "required": True,
            "expected_sha256": inputs.cases[0].geometry_sha256,
            "expected_size_bytes": inputs.cases[0].geometry_size_bytes,
            "media_type": "model/stl",
        },
    ]


def test_plugin_discovery_does_not_import_physicsnemo_cfd():
    code = f"""
import importlib.util
import sys
import types
from pathlib import Path
path = Path({str(PLUGIN_ROOT / "workflow.py")!r})
class ForeignWorkflow:
    pass
poison = types.ModuleType('surface_benchmark_impl')
poison.SurfaceBenchmarkWorkflow = ForeignWorkflow
sys.modules['surface_benchmark_impl'] = poison
spec = importlib.util.spec_from_file_location('surface_workflow_discovery_test', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert not any(name == 'physicsnemo' or name.startswith('physicsnemo.') for name in sys.modules)
assert module.WORKFLOW.__name__ == 'PhysicsNeMoCfdSurfaceBenchmarkWorkflow'
assert not issubclass(module.WORKFLOW, ForeignWorkflow)
assert sys.modules['surface_benchmark_impl'] is poison
assert not hasattr(module, 'prepare_model_cache')
implementation = module._SURFACE_IMPLEMENTATION
assert Path(implementation.__file__).resolve().parent == path.parent
assert 'physicsnemo_cfd_runtime' in sys.modules
workflow = module.WORKFLOW()
assert workflow.cache_scope == 'process'
assert not hasattr(workflow, 'model_cache_names')
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "scripts"), str(REPO_ROOT / "python")]
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_load_config_rejects_policy_mutation(tmp_path: Path):
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["configuration"]["benchmark"]["module_argv"][-1] = "{client_path}"
    mutated = tmp_path / "plugin.yaml"
    mutated.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="fixed argv contract"):
        surface.load_surface_plugin_config(mutated)

    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["configuration"]["benchmark"]["benchmark_config"]["reproducibility"][
        "log_env"
    ] = True
    mutated.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="environment logging"):
        surface.load_surface_plugin_config(mutated)


def test_normalize_surface_request_applies_defaults_and_allowlists():
    raw = _request_fields()
    raw.pop("metrics")
    raw.pop("seed")
    inputs = surface.normalize_surface_request(raw, _config())

    assert inputs.metrics == [
        "l2_pressure",
        "l2_shear_stress",
        "l2_pressure_area_weighted",
        "drag",
        "lift",
    ]
    assert inputs.seed == 42
    assert inputs.cases[0].case_id == "run_1"


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"models": ["arbitrary_python_target"]}, "unsupported surface model"),
        ({"batch_size": 2}, "unsupported request fields: batch_size"),
        ({"metrics": ["arbitrary_metric"]}, "unsupported surface metric"),
        ({"cases": []}, "between 1 and 8"),
        (
            {"cases": [_case_fields(case_id="../escape")]},
            "must match run_<number>",
        ),
        (
            {"cases": [_case_fields(mesh_uri="http://assets.example.com/file.vtp")]},
            "must use https",
        ),
        (
            {"cases": [_case_fields(mesh_uri="https://127.0.0.1/file.vtp")]},
            "DNS hostname",
        ),
        (
            {
                "cases": [
                    _case_fields(
                        mesh_uri="https://assets.example.com/file.vtp?token=secret"
                    )
                ]
            },
            "secret-bearing query",
        ),
        ({"cases": [_case_fields(sha256="bad")]}, "64 hexadecimal"),
        ({"visual_case_ids": ["run_99"]}, "subset"),
    ],
)
def test_normalize_surface_request_rejects_unsafe_values(overrides, message):
    with pytest.raises((TypeError, ValueError), match=message):
        surface.normalize_surface_request(_request_fields(**overrides), _config())


def test_normalize_surface_request_enforces_case_id_length_boundary():
    boundary_case_id = "run_" + ("1" * 60)
    inputs = surface.normalize_surface_request(
        _request_fields(cases=[_case_fields(case_id=boundary_case_id)]),
        _config(),
    )
    assert inputs.cases[0].case_id == boundary_case_id
    assert len(inputs.cases[0].case_id) == 64

    overlong_case_id = "run_" + ("1" * 61)
    with pytest.raises(ValueError, match="at most 64 characters"):
        surface.normalize_surface_request(
            _request_fields(cases=[_case_fields(case_id=overlong_case_id)]),
            _config(),
        )


def test_prepare_emits_integrity_prefetch_and_gpu_profile(tmp_path: Path):
    workflow = surface.SurfaceBenchmarkWorkflow(MANIFEST_PATH)
    result = workflow.prepare(
        RawRequest(
            content_type="application/json",
            operation="run",
            raw_fields=_request_fields(),
            input_artifacts=[],
        ),
        PrepareContext(
            run_id="prepare-1",
            workflow_id="physicsnemo-cfd-surface-benchmark",
            run_dir=tmp_path,
        ),
    )

    case = result.inputs.cases[0]
    assert result.prefetch_plan == [
        {
            "kind": "http_fetch",
            "source_uri": case.mesh_uri,
            "target_artifact_name": "surface-mesh-run_1",
            "required": True,
            "expected_sha256": case.sha256,
            "expected_size_bytes": case.size_bytes,
            "media_type": "application/vnd.vtk.vtp",
        },
        {
            "kind": "http_fetch",
            "source_uri": case.geometry_uri,
            "target_artifact_name": "surface-geometry-run_1",
            "required": True,
            "expected_sha256": case.geometry_sha256,
            "expected_size_bytes": case.geometry_size_bytes,
            "media_type": "model/stl",
        },
    ]
    assert result.resource_profile.executor_class == "physicsnemo-cfd-gpu"
    assert result.resource_profile.gpus_required == 1
    assert result.resource_profile.memory_mb == 65000
    assert result.resource_profile.tags == ["physicsnemo-cfd", "gpu"]


def test_build_resolved_config_uses_only_fixed_models_and_paths(tmp_path: Path):
    inputs = _inputs(
        models=["geotransolver_surface", "xmgn_surface"],
        visual_case_ids=["run_1"],
        save_inference_mesh=True,
    )
    resolved = surface.build_resolved_config(
        inputs,
        _config(),
        dataset_root=tmp_path / "dataset",
        output_dir=tmp_path / "output",
    )

    assert resolved["run"]["device"] == "cuda:0"
    assert resolved["run"]["batch_size"] == 1
    assert resolved["run"]["distributed"] is False
    assert resolved["run"]["output_dir"] == str((tmp_path / "output").resolve())
    assert resolved["benchmark"]["reproducibility"] == {
        "log_env": False,
        "save_artifacts": True,
    }
    assert resolved["benchmark"]["models"] == [
        {
            "name": "geotransolver_surface",
            "inference_domain": "surface",
            "package": "hf://nvidia/geotransolver_drivaerml@626c1158e14f6994382924055aa871f863ff8a8c",
            "kwargs": {"batch_resolution": 60000, "geometry_sampling": 300000},
        },
        {
            "name": "xmgn_surface",
            "inference_domain": "surface",
            "package": "hf://nvidia/xmgn_drivaerml_surface@33909568711c0f60bd5fa6f8809e6d51c117f821",
            "kwargs": {"max_points": 250000, "interpolation_k": 4},
        },
    ]
    assert resolved["benchmark"]["datasets"][0]["case_ids"] == ["run_1"]
    assert resolved["reports"]["enabled"] is True
    assert resolved["reports"]["visual_case_ids"] == ["run_1"]
    serialized = json.dumps(resolved)
    assert "checkpoint" not in serialized
    assert 'log_env": true' not in serialized.lower()


def test_materialize_drivaerml_layout_uses_verified_case_name(tmp_path: Path):
    source = tmp_path / "downloaded-object"
    source.write_bytes(FIXTURE_PATH.read_bytes())
    inputs = _inputs()
    dataset_root = tmp_path / "dataset"

    result = surface.materialize_drivaerml_layout(
        inputs,
        _prefetched(source, inputs),
        dataset_root=dataset_root,
    )

    destination = dataset_root / "run_1" / "boundary_1.vtp"
    geometry_destination = dataset_root / "run_1" / "drivaer_1.stl"
    assert result == dataset_root.resolve()
    assert destination.read_bytes() == source.read_bytes()
    assert geometry_destination.read_bytes() == TINY_STL_BYTES
    assert not destination.is_symlink()
    assert not geometry_destination.is_symlink()
    destination.write_bytes(b"provider mutation")
    assert source.read_bytes() == FIXTURE_PATH.read_bytes()


def test_materialize_rejects_missing_digest_and_symlink_escape(tmp_path: Path):
    source = tmp_path / "downloaded-object"
    source.write_bytes(FIXTURE_PATH.read_bytes())
    inputs = _inputs()
    artifacts = _prefetched(source, inputs)
    artifacts[0].pop("sha256")
    with pytest.raises(ValueError, match="digest mismatch"):
        surface.materialize_drivaerml_layout(
            inputs,
            artifacts,
            dataset_root=tmp_path / "missing-digest",
        )

    outside = tmp_path / "outside"
    outside.mkdir()
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "run_1").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="fresh directory already exists"):
        surface.materialize_drivaerml_layout(
            inputs,
            _prefetched(source, inputs),
            dataset_root=dataset_root,
        )


def test_materialize_stream_verifies_content_and_cleans_partial_files(tmp_path: Path):
    expected = FIXTURE_PATH.read_bytes()
    corrupt = bytearray(expected)
    corrupt[-1] = (corrupt[-1] + 1) % 256
    source = tmp_path / "corrupt-cache-object"
    source.write_bytes(corrupt)
    inputs = _inputs()
    dataset_root = tmp_path / "dataset"

    with pytest.raises(ValueError, match="digest mismatch"):
        surface.materialize_drivaerml_layout(
            inputs,
            _prefetched(source, inputs),
            dataset_root=dataset_root,
        )

    case_dir = dataset_root / "run_1"
    assert not (case_dir / "boundary_1.vtp").exists()
    assert list(case_dir.glob("*.tmp")) == []

    source.unlink()
    source.symlink_to(FIXTURE_PATH)
    with pytest.raises(ValueError, match="cannot be opened safely"):
        surface.materialize_drivaerml_layout(
            inputs,
            _prefetched(source, inputs),
            dataset_root=tmp_path / "symlink-source",
        )


def test_materialize_rejects_fifo_without_blocking(tmp_path: Path):
    source = tmp_path / "fifo-cache-object"
    os.mkfifo(source)
    inputs = _inputs()
    errors: list[BaseException] = []

    def materialize() -> None:
        try:
            surface.materialize_drivaerml_layout(
                inputs,
                _prefetched(source, inputs),
                dataset_root=tmp_path / "fifo-dataset",
            )
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=materialize, daemon=True)
    worker.start()
    worker.join(timeout=1)

    assert not worker.is_alive(), "opening an untrusted FIFO blocked mesh staging"
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)
    assert "not a regular file" in str(errors[0])


def test_materialize_long_copy_observes_background_cancellation(tmp_path: Path):
    content = b"mesh-data" * (4 * 1024 * 1024)
    source = tmp_path / "large-cache-object"
    source.write_bytes(content)
    case = surface.SurfaceCase(
        case_id="run_1",
        mesh_uri="https://assets.example.com/run_1/boundary_1.vtp",
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        geometry_uri="https://assets.example.com/run_1/drivaer_1.stl",
        geometry_sha256=hashlib.sha256(TINY_STL_BYTES).hexdigest(),
        geometry_size_bytes=len(TINY_STL_BYTES),
    )
    inputs = surface.SurfaceBenchmarkInput(
        models=["domino_surface"],
        cases=[case],
        metrics=["l2_pressure"],
    )
    callback_called = threading.Event()

    def abort_requested() -> bool:
        callback_called.set()
        return True

    dataset_root = tmp_path / "cancelled-dataset"
    with pytest.raises(PluginCancelledError, match="staging was cancelled"):
        surface.materialize_drivaerml_layout(
            inputs,
            _prefetched(source, inputs),
            dataset_root=dataset_root,
            abort_requested=abort_requested,
        )

    assert callback_called.is_set()
    assert not (dataset_root / "run_1" / "boundary_1.vtp").exists()
    assert list(dataset_root.rglob("*.tmp")) == []


def test_materialize_rejects_symlinked_dataset_root(tmp_path: Path):
    source = tmp_path / "downloaded-object"
    source.write_bytes(FIXTURE_PATH.read_bytes())
    inputs = _inputs()
    real_dataset = tmp_path / "real-dataset"
    real_dataset.mkdir()
    dataset_link = tmp_path / "dataset-link"
    dataset_link.symlink_to(real_dataset, target_is_directory=True)

    with pytest.raises(ValueError, match="dataset root must not be a symlink"):
        surface.materialize_drivaerml_layout(
            inputs,
            _prefetched(source, inputs),
            dataset_root=dataset_link,
        )


def test_benchmark_command_is_exact_and_has_no_override_channel(tmp_path: Path):
    config_path = tmp_path / "resolved.json"
    assert surface.benchmark_command(
        _config(), config_path, python_executable="/opt/cfd/bin/python"
    ) == [
        "/opt/cfd/bin/python",
        "-m",
        "physicsnemo.cfd.evaluation.benchmarks.run",
        "--config",
        str(config_path.resolve()),
    ]


def test_register_known_outputs_rejects_symlink_escape_before_registration(
    tmp_path: Path,
):
    run_dir = tmp_path / "run"
    output_dir = run_dir / "benchmark-output"
    output_dir.mkdir(parents=True)
    outside_report = tmp_path / "outside.json"
    outside_report.write_text("[]", encoding="utf-8")
    (output_dir / "benchmark_results.json").symlink_to(outside_report)
    (output_dir / "benchmark_results.csv").write_text("model\n", encoding="utf-8")
    (output_dir / "benchmark_results.html").write_text(
        "<html></html>", encoding="utf-8"
    )
    outputs = OutputRegistry(run_dir)
    ctx = ExecutionContext(run_id="symlink", run_dir=run_dir, outputs=outputs)

    with pytest.raises(ValueError, match="must not be a symlink"):
        surface.register_known_outputs(
            ctx,
            _config(),
            output_dir=output_dir,
            resolved_config_path=run_dir / "resolved_config.json",
            diagnostics_path=run_dir / "benchmark_diagnostics.json",
            log_path=run_dir / "benchmark.log",
            include_meshes=False,
            include_visuals=False,
        )
    assert outputs.registered_outputs() == []


def test_register_known_outputs_validates_audit_files_before_primary_registration(
    tmp_path: Path,
):
    run_dir = tmp_path / "run"
    output_dir = run_dir / "benchmark-output"
    output_dir.mkdir(parents=True)
    (output_dir / "benchmark_results.json").write_text("[]", encoding="utf-8")
    (output_dir / "benchmark_results.csv").write_text("model\n", encoding="utf-8")
    (output_dir / "benchmark_results.html").write_text(
        "<html></html>", encoding="utf-8"
    )
    resolved_config = run_dir / "resolved_config.json"
    diagnostics = run_dir / "benchmark_diagnostics.json"
    resolved_config.write_text("{}", encoding="utf-8")
    diagnostics.write_text("{}", encoding="utf-8")
    outputs = OutputRegistry(run_dir)
    ctx = ExecutionContext(run_id="missing-log", run_dir=run_dir, outputs=outputs)

    with pytest.raises(RuntimeError, match="required audit artifact is missing"):
        surface.register_known_outputs(
            ctx,
            _config(),
            output_dir=output_dir,
            resolved_config_path=resolved_config,
            diagnostics_path=diagnostics,
            log_path=run_dir / "benchmark.log",
            include_meshes=False,
            include_visuals=False,
        )
    assert outputs.registered_outputs() == []


def test_supervisor_bounds_logs_and_reports_timeout_and_cancellation(tmp_path: Path):
    log_path = tmp_path / "bounded.log"
    result = cfd_supervisor.run_supervised_process(
        [sys.executable, "-c", "print('x' * 20000)"],
        cwd=tmp_path,
        log_path=log_path,
        timeout_seconds=5,
        termination_grace_seconds=0.1,
        max_log_bytes=1024,
        abort_requested=lambda: False,
    )
    assert result.returncode == 0
    assert result.log_truncated is True
    assert log_path.stat().st_size == 1024

    timed_out = cfd_supervisor.run_supervised_process(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
        log_path=tmp_path / "timeout.log",
        timeout_seconds=0.2,
        termination_grace_seconds=0.1,
        max_log_bytes=1024,
        abort_requested=lambda: False,
    )
    assert timed_out.timed_out is True
    assert timed_out.duration_seconds < 3

    started = time.monotonic()
    cancelled = cfd_supervisor.run_supervised_process(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
        log_path=tmp_path / "cancel.log",
        timeout_seconds=5,
        termination_grace_seconds=0.1,
        max_log_bytes=1024,
        abort_requested=lambda: time.monotonic() - started > 0.2,
    )
    assert cancelled.cancelled is True
    assert cancelled.duration_seconds < 3


def test_supervisor_fails_closed_when_abort_callback_exceeds_max_age(tmp_path: Path):
    release_callback = threading.Event()
    callback_entered = threading.Event()
    callback_exited = threading.Event()

    def blocking_abort() -> bool:
        callback_entered.set()
        try:
            release_callback.wait(10)
            return False
        finally:
            callback_exited.set()

    try:
        result = cfd_supervisor.run_supervised_process(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=tmp_path,
            log_path=tmp_path / "blocking-abort.log",
            timeout_seconds=5,
            termination_grace_seconds=0.1,
            max_log_bytes=1024,
            abort_requested=blocking_abort,
        )
    finally:
        release_callback.set()

    assert callback_entered.is_set()
    assert result.cancelled is True
    assert result.timed_out is False
    assert result.duration_seconds < 2
    assert callback_exited.wait(1)


def test_abort_callback_workers_are_globally_bounded(tmp_path: Path):
    release_callback = threading.Event()
    counter_lock = threading.Lock()
    entered_callbacks = 0
    exited_callbacks = 0
    all_workers_entered = threading.Event()

    def blocking_abort() -> bool:
        nonlocal entered_callbacks, exited_callbacks
        with counter_lock:
            entered_callbacks += 1
            if entered_callbacks == cfd_supervisor.ABORT_PROBE_WORKER_COUNT:
                all_workers_entered.set()
        try:
            release_callback.wait(10)
            return False
        finally:
            with counter_lock:
                exited_callbacks += 1

    worker_name_prefix = "cfd-abort-callback-worker-"
    workers_before = {
        thread.ident
        for thread in threading.enumerate()
        if thread.name.startswith(worker_name_prefix)
    }
    try:
        for index in range(cfd_supervisor.ABORT_PROBE_WORKER_COUNT):
            result = cfd_supervisor.run_supervised_process(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                cwd=tmp_path,
                log_path=tmp_path / f"blocked-callback-{index}.log",
                timeout_seconds=0.05,
                termination_grace_seconds=0.01,
                max_log_bytes=1024,
                abort_requested=blocking_abort,
            )
            assert result.timed_out is True
        assert all_workers_entered.wait(1)

        started = time.monotonic()
        fail_closed = cfd_supervisor.run_supervised_process(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=tmp_path,
            log_path=tmp_path / "saturated-pool.log",
            timeout_seconds=5,
            termination_grace_seconds=0.01,
            max_log_bytes=1024,
            abort_requested=lambda: True,
        )
        assert fail_closed.cancelled is True
        assert fail_closed.timed_out is False
        assert time.monotonic() - started < 1

        source = tmp_path / "bounded-pool-source"
        source.write_bytes(FIXTURE_PATH.read_bytes())
        inputs = _inputs()
        with pytest.raises(PluginCancelledError, match="staging was cancelled"):
            surface.materialize_drivaerml_layout(
                inputs,
                _prefetched(source, inputs),
                dataset_root=tmp_path / "bounded-pool-dataset",
                abort_requested=lambda: True,
            )

        workers_after = {
            thread.ident
            for thread in threading.enumerate()
            if thread.name.startswith(worker_name_prefix)
        }
        assert len(workers_after) <= cfd_supervisor.ABORT_PROBE_WORKER_COUNT
        assert (
            len(workers_after - workers_before)
            <= cfd_supervisor.ABORT_PROBE_WORKER_COUNT
        )
        with counter_lock:
            assert entered_callbacks == cfd_supervisor.ABORT_PROBE_WORKER_COUNT
    finally:
        release_callback.set()

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        with counter_lock:
            if exited_callbacks == entered_callbacks:
                break
        time.sleep(0.01)
    with counter_lock:
        assert exited_callbacks == entered_callbacks


def test_supervisor_does_not_wait_for_escaped_stdout_holder(tmp_path: Path):
    holder_pid_path = tmp_path / "escaped-holder.pid"
    holder_code = (
        "import os, pathlib, time; "
        f"pathlib.Path({str(holder_pid_path)!r}).write_text(str(os.getpid())); "
        "time.sleep(30)"
    )
    leader_code = f"""
import pathlib
import subprocess
import sys
import time

pid_path = pathlib.Path({str(holder_pid_path)!r})
subprocess.Popen(
    [sys.executable, "-c", {holder_code!r}],
    start_new_session=True,
)
deadline = time.monotonic() + 5
while not pid_path.exists() and time.monotonic() < deadline:
    time.sleep(0.01)
print("leader complete", flush=True)
"""
    results: list[cfd_supervisor.SupervisedProcessResult] = []
    errors: list[BaseException] = []

    def supervise() -> None:
        try:
            results.append(
                cfd_supervisor.run_supervised_process(
                    [sys.executable, "-c", leader_code],
                    cwd=tmp_path,
                    log_path=tmp_path / "escaped-holder.log",
                    timeout_seconds=5,
                    termination_grace_seconds=0.1,
                    max_log_bytes=1024,
                    abort_requested=lambda: False,
                )
            )
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=supervise, daemon=True)
    worker.start()
    try:
        worker.join(timeout=2)
        assert not worker.is_alive(), "supervisor blocked on inherited stdout"
        assert errors == []
        assert len(results) == 1
        assert results[0].returncode == 0
        assert results[0].duration_seconds < 2
        assert "leader complete" in (tmp_path / "escaped-holder.log").read_text(
            encoding="utf-8"
        )
    finally:
        if holder_pid_path.exists():
            holder_pid = int(holder_pid_path.read_text(encoding="utf-8"))
            try:
                os.kill(holder_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        worker.join(timeout=2)


def test_supervisor_sigterm_cancels_process_group_and_restores_handler(tmp_path: Path):
    child_pid_path = tmp_path / "child.pid"
    child_code = (
        "import pathlib, subprocess, sys, time; "
        "child=subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(30)']); "
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid)); "
        "time.sleep(30)"
    )
    original_handler = signal.getsignal(signal.SIGTERM)
    chained_handler_called = threading.Event()

    def existing_handler(_signum, _frame) -> None:
        chained_handler_called.set()

    signal.signal(signal.SIGTERM, existing_handler)

    def send_sigterm() -> None:
        deadline = time.monotonic() + 5
        while not child_pid_path.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        if child_pid_path.exists():
            os.kill(os.getpid(), signal.SIGTERM)

    try:
        sender = threading.Thread(target=send_sigterm, daemon=True)
        sender.start()
        result = cfd_supervisor.run_supervised_process(
            [sys.executable, "-c", child_code],
            cwd=tmp_path,
            log_path=tmp_path / "signal-cancel.log",
            timeout_seconds=5,
            termination_grace_seconds=0.2,
            max_log_bytes=1024,
            abort_requested=lambda: False,
        )
        sender.join(timeout=1)
        restored_handler = signal.getsignal(signal.SIGTERM)
    finally:
        signal.signal(signal.SIGTERM, original_handler)

    assert result.cancelled is True
    assert chained_handler_called.is_set()
    assert restored_handler is existing_handler
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 3
    child_alive = True
    while time.monotonic() < deadline:
        status = subprocess.run(
            ["ps", "-p", str(child_pid), "-o", "stat="],
            text=True,
            capture_output=True,
            check=False,
        )
        child_alive = status.returncode == 0 and not status.stdout.strip().startswith(
            "Z"
        )
        if not child_alive:
            break
        time.sleep(0.05)
    assert child_alive is False


def test_supervisor_and_json_writer_reject_preexisting_symlink_outputs(tmp_path: Path):
    target = tmp_path / "outside"
    target.write_text("do not overwrite", encoding="utf-8")
    log_link = tmp_path / "benchmark.log"
    log_link.symlink_to(target)
    with pytest.raises(ValueError, match="unsafe output path"):
        cfd_supervisor.run_supervised_process(
            [sys.executable, "-c", "print('never runs')"],
            cwd=tmp_path,
            log_path=log_link,
            timeout_seconds=1,
            termination_grace_seconds=0.1,
            max_log_bytes=1024,
            abort_requested=lambda: False,
        )

    config_link = tmp_path / "resolved_config.json"
    config_link.symlink_to(target)
    with pytest.raises(ValueError, match="unsafe output path"):
        cfd_safe_files.write_json_exclusive(config_link, {"safe": True})
    assert target.read_text(encoding="utf-8") == "do not overwrite"


def test_workflow_execute_registers_only_known_root_contained_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    source = tmp_path / "downloaded-object"
    source.write_bytes(FIXTURE_PATH.read_bytes())
    inputs = _inputs()
    run_dir = tmp_path / "run"
    outputs = OutputRegistry(run_dir)

    def fake_supervisor(argv, **kwargs):
        resolved = json.loads(Path(argv[-1]).read_text(encoding="utf-8"))
        output_dir = Path(resolved["run"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "benchmark_results.json").write_text(
            '[{"model": "domino_surface"}]', encoding="utf-8"
        )
        (output_dir / "benchmark_results.csv").write_text(
            "model\ndomino_surface\n", encoding="utf-8"
        )
        (output_dir / "benchmark_results.html").write_text(
            "<html></html>", encoding="utf-8"
        )
        (output_dir / "untrusted-extra.bin").write_bytes(b"ignore")
        Path(kwargs["log_path"]).write_text("complete\n", encoding="utf-8")
        return cfd_supervisor.SupervisedProcessResult(
            returncode=0,
            duration_seconds=1.25,
            log_bytes=9,
            log_truncated=False,
        )

    monkeypatch.setattr(surface, "run_supervised_process", fake_supervisor)
    workflow = surface.SurfaceBenchmarkWorkflow(MANIFEST_PATH)
    result = workflow.execute(
        {
            "run_id": "surface-1",
            "run_dir": run_dir,
            "outputs": outputs,
            "parameters": dataclasses.asdict(inputs),
            "prefetch_artifacts": _prefetched(source, inputs),
            "resource_profile": {
                "executor_class": "physicsnemo-cfd-gpu",
                "gpus_required": 1,
            },
        }
    )

    assert result["report_path"].endswith("benchmark_results.json")
    assert result["model_names"] == ["domino_surface"]
    assert result["provider"] == _config()["provider"]
    assert result["preset_sha256"] == surface.preset_sha256(_config())
    assert result["selected_metrics"] == ["l2_pressure", "drag"]
    assert result["case_digests"] == [
        {
            "case_id": "run_1",
            "sha256": inputs.cases[0].sha256,
            "size_bytes": inputs.cases[0].size_bytes,
            "geometry_sha256": inputs.cases[0].geometry_sha256,
            "geometry_size_bytes": inputs.cases[0].geometry_size_bytes,
        }
    ]
    registered = outputs.registered_outputs()
    primary = outputs.primary_output()
    assert primary is not None
    assert primary.name == "benchmark_results.json"
    assert Path(primary.path).is_relative_to(run_dir)
    assert "untrusted-extra.bin" not in {item.name for item in registered}
    assert {item.name for item in registered} >= {
        "benchmark_results.json",
        "benchmark_results.csv",
        "benchmark_results.html",
        "resolved_config.json",
        "benchmark_diagnostics.json",
        "benchmark.log",
    }
    assert result["registered_artifact_names"] == [
        "benchmark_results.json",
        "benchmark_results.csv",
        "benchmark_results.html",
        "resolved_config.json",
        "benchmark_diagnostics.json",
        "benchmark.log",
    ]
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    validate(result, manifest["outputs"]["result_schema_inline"])


def test_workflow_cancellation_raises_typed_error_and_preserves_audit_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    source = tmp_path / "downloaded-object"
    source.write_bytes(FIXTURE_PATH.read_bytes())
    inputs = _inputs()
    run_dir = tmp_path / "cancelled-run"
    outputs = OutputRegistry(run_dir)

    def cancelled_supervisor(_argv, **kwargs):
        Path(kwargs["log_path"]).write_text("cancelled\n", encoding="utf-8")
        return cfd_supervisor.SupervisedProcessResult(
            returncode=-signal.SIGTERM,
            duration_seconds=0.2,
            log_bytes=10,
            log_truncated=False,
            cancelled=True,
        )

    monkeypatch.setattr(surface, "run_supervised_process", cancelled_supervisor)
    workflow = surface.SurfaceBenchmarkWorkflow(MANIFEST_PATH)
    with pytest.raises(PluginCancelledError, match="was cancelled"):
        workflow.execute(
            {
                "run_id": "cancelled",
                "run_dir": run_dir,
                "outputs": outputs,
                "parameters": dataclasses.asdict(inputs),
                "prefetch_artifacts": _prefetched(source, inputs),
            }
        )

    assert outputs.primary_output() is None
    assert [artifact.name for artifact in outputs.registered_outputs()] == [
        "resolved_config.json",
        "benchmark_diagnostics.json",
        "benchmark.log",
    ]


def test_workflow_uses_fresh_attempts_and_never_reuses_stale_reports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    source = tmp_path / "downloaded-object"
    source.write_bytes(FIXTURE_PATH.read_bytes())
    inputs = _inputs()
    run_dir = tmp_path / "retry-run"
    calls = 0

    def retry_supervisor(argv, **kwargs):
        nonlocal calls
        calls += 1
        resolved = json.loads(Path(argv[-1]).read_text(encoding="utf-8"))
        output_dir = Path(resolved["run"]["output_dir"])
        if calls == 1:
            (output_dir / "benchmark_results.json").write_text("[]", encoding="utf-8")
            (output_dir / "benchmark_results.csv").write_text(
                "model\n", encoding="utf-8"
            )
            (output_dir / "benchmark_results.html").write_text(
                "<html></html>", encoding="utf-8"
            )
        Path(kwargs["log_path"]).write_text("done\n", encoding="utf-8")
        return cfd_supervisor.SupervisedProcessResult(
            returncode=0,
            duration_seconds=0.1,
            log_bytes=5,
            log_truncated=False,
        )

    monkeypatch.setattr(surface, "run_supervised_process", retry_supervisor)
    workflow = surface.SurfaceBenchmarkWorkflow(MANIFEST_PATH)

    def context(outputs: OutputRegistry) -> dict:
        return {
            "run_id": "retry",
            "run_dir": run_dir,
            "outputs": outputs,
            "parameters": dataclasses.asdict(inputs),
            "prefetch_artifacts": _prefetched(source, inputs),
        }

    workflow.execute(context(OutputRegistry(run_dir)))
    failed_outputs = OutputRegistry(run_dir)
    with pytest.raises(RuntimeError, match="did not produce benchmark_results.json"):
        workflow.execute(context(failed_outputs))

    assert failed_outputs.primary_output() is None
    assert [artifact.name for artifact in failed_outputs.registered_outputs()] == [
        "resolved_config.json",
        "benchmark_diagnostics.json",
        "benchmark.log",
    ]

    attempts = sorted(run_dir.glob("physicsnemo-cfd-surface-attempt-*"))
    assert len(attempts) == 2
    assert attempts[0] != attempts[1]
    reports = [
        attempt / "benchmark-output" / "benchmark_results.json" for attempt in attempts
    ]
    assert sum(report.is_file() for report in reports) == 1


def test_workflow_rejects_symlinked_run_directory_before_writes(tmp_path: Path):
    source = tmp_path / "downloaded-object"
    source.write_bytes(FIXTURE_PATH.read_bytes())
    inputs = _inputs()
    real_run = tmp_path / "real-run"
    real_run.mkdir()
    run_link = tmp_path / "run-link"
    run_link.symlink_to(real_run, target_is_directory=True)
    workflow = surface.SurfaceBenchmarkWorkflow(MANIFEST_PATH)

    with pytest.raises(ValueError, match="run directory must not be a symlink"):
        workflow.execute(
            {
                "run_id": "symlink",
                "run_dir": run_link,
                "parameters": dataclasses.asdict(inputs),
                "prefetch_artifacts": _prefetched(source, inputs),
            }
        )
    assert list(real_run.iterdir()) == []


def test_workflow_fails_on_nonzero_exit_and_missing_required_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    source = tmp_path / "downloaded-object"
    source.write_bytes(FIXTURE_PATH.read_bytes())
    inputs = _inputs()

    def failed_supervisor(_argv, **kwargs):
        Path(kwargs["log_path"]).write_text("provider failed", encoding="utf-8")
        return cfd_supervisor.SupervisedProcessResult(
            returncode=7,
            duration_seconds=0.1,
            log_bytes=15,
            log_truncated=False,
        )

    monkeypatch.setattr(surface, "run_supervised_process", failed_supervisor)
    workflow = surface.SurfaceBenchmarkWorkflow(MANIFEST_PATH)
    ctx = {
        "run_id": "failure",
        "run_dir": tmp_path / "failure",
        "parameters": dataclasses.asdict(inputs),
        "prefetch_artifacts": _prefetched(source, inputs),
    }
    with pytest.raises(RuntimeError, match="exited with code 7"):
        workflow.execute(ctx)
    assert list((tmp_path / "failure").glob("*/benchmark_diagnostics.json"))

    def missing_csv_supervisor(argv, **kwargs):
        resolved = json.loads(Path(argv[-1]).read_text(encoding="utf-8"))
        output_dir = Path(resolved["run"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "benchmark_results.json").write_text("[]", encoding="utf-8")
        (output_dir / "benchmark_results.html").write_text(
            "<html></html>", encoding="utf-8"
        )
        Path(kwargs["log_path"]).write_text("done", encoding="utf-8")
        return cfd_supervisor.SupervisedProcessResult(
            returncode=0,
            duration_seconds=0.1,
            log_bytes=4,
            log_truncated=False,
        )

    monkeypatch.setattr(surface, "run_supervised_process", missing_csv_supervisor)
    ctx["run_id"] = "missing"
    ctx["run_dir"] = tmp_path / "missing"
    missing_outputs = OutputRegistry(ctx["run_dir"])
    ctx["outputs"] = missing_outputs
    with pytest.raises(RuntimeError, match="did not produce benchmark_results.csv"):
        workflow.execute(ctx)
    assert missing_outputs.primary_output() is None
    assert [artifact.name for artifact in missing_outputs.registered_outputs()] == [
        "resolved_config.json",
        "benchmark_diagnostics.json",
        "benchmark.log",
    ]


def test_workflow_preserves_audits_when_report_json_is_invalid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    source = tmp_path / "downloaded-object"
    source.write_bytes(FIXTURE_PATH.read_bytes())
    inputs = _inputs()
    run_dir = tmp_path / "invalid-report"
    outputs = OutputRegistry(run_dir)

    def invalid_report_supervisor(argv, **kwargs):
        resolved = json.loads(Path(argv[-1]).read_text(encoding="utf-8"))
        output_dir = Path(resolved["run"]["output_dir"])
        (output_dir / "benchmark_results.json").write_text(
            "{invalid-json", encoding="utf-8"
        )
        (output_dir / "benchmark_results.csv").write_text("model\n", encoding="utf-8")
        (output_dir / "benchmark_results.html").write_text(
            "<html></html>", encoding="utf-8"
        )
        Path(kwargs["log_path"]).write_text("done", encoding="utf-8")
        return cfd_supervisor.SupervisedProcessResult(
            returncode=0,
            duration_seconds=0.1,
            log_bytes=4,
            log_truncated=False,
        )

    monkeypatch.setattr(surface, "run_supervised_process", invalid_report_supervisor)
    workflow = surface.SurfaceBenchmarkWorkflow(MANIFEST_PATH)
    with pytest.raises(json.JSONDecodeError):
        workflow.execute(
            {
                "run_id": "invalid-report",
                "run_dir": run_dir,
                "outputs": outputs,
                "parameters": dataclasses.asdict(inputs),
                "prefetch_artifacts": _prefetched(source, inputs),
            }
        )

    assert outputs.primary_output() is None
    assert [artifact.name for artifact in outputs.registered_outputs()] == [
        "resolved_config.json",
        "benchmark_diagnostics.json",
        "benchmark.log",
    ]


def test_workflow_preserves_bounded_audits_when_process_launch_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    source = tmp_path / "downloaded-object"
    source.write_bytes(FIXTURE_PATH.read_bytes())
    inputs = _inputs()
    run_dir = tmp_path / "launch-failure"
    outputs = OutputRegistry(run_dir)
    missing_python = tmp_path / "missing-cfd-python"
    monkeypatch.setenv("PHYSICSNEMO_CFD_PYTHON_EXECUTABLE", str(missing_python))

    workflow = surface.SurfaceBenchmarkWorkflow(MANIFEST_PATH)
    with pytest.raises(FileNotFoundError):
        workflow.execute(
            {
                "run_id": "launch-failure",
                "run_dir": run_dir,
                "outputs": outputs,
                "parameters": dataclasses.asdict(inputs),
                "prefetch_artifacts": _prefetched(source, inputs),
            }
        )

    assert outputs.primary_output() is None
    assert [artifact.name for artifact in outputs.registered_outputs()] == [
        "resolved_config.json",
        "benchmark_diagnostics.json",
        "benchmark.log",
    ]
    attempt = next(run_dir.glob("physicsnemo-cfd-surface-attempt-*"))
    diagnostics_path = attempt / "benchmark_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["supervisor_error"]["type"] == "FileNotFoundError"
    assert len(diagnostics["supervisor_error"]["message"]) <= 2048
    assert "traceback" not in diagnostics
    assert diagnostics["log_bytes"] == 0
    assert (attempt / "benchmark.log").is_file()

    surface.register_audit_outputs(
        ExecutionContext(run_id="launch-failure", run_dir=run_dir, outputs=outputs),
        resolved_config_path=attempt / "resolved_config.json",
        diagnostics_path=diagnostics_path,
        log_path=attempt / "benchmark.log",
    )
    assert len(outputs.registered_outputs()) == 3


def test_plugin_manifest_and_example_validate():
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "plugin_dev.py"),
            "validate",
            str(PLUGIN_ROOT),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["status"] == "valid"
