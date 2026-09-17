"""Integrated producer workflows and framework-free completion verification."""

import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import struct

from pnmir_build import worker
from pnmir_build.inputs import input_identities, resolve_inputs
from pnmir_build.tensors import tensor_contracts


MODEL = "geotransolver-surface-core"
PRODUCER_FILES = ("prepare.py", "adapter.py")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _path(root, relative):
    _require(isinstance(relative, str) and bool(relative), "artifact path is missing")
    path = Path(relative)
    _require(
        not path.is_absolute() and ".." not in path.parts,
        "artifact path must remain within its root",
    )
    selected = root
    for part in path.parts:
        selected /= part
        _require(not selected.is_symlink(), "artifact path must not contain symlinks")
    return selected


def _file(root, relative):
    selected = _path(root, relative)
    _require(selected.is_file(), f"artifact file is missing: {relative}")
    return selected


def _directory(root, relative):
    selected = _path(root, relative)
    _require(selected.is_dir(), f"artifact directory is missing: {relative}")
    return selected


def _verify(root, identity):
    _require(isinstance(identity, dict), "artifact identity must be an object")
    path = _file(root, identity.get("path"))
    actual = worker._file_identity(path, root)
    _require(
        all(identity.get(key) == value for key, value in actual.items()),
        f"artifact identity mismatch: {identity.get('path')}",
    )
    return path


def _inventory(root, records):
    _require(isinstance(records, list) and bool(records), "artifact inventory is empty")
    paths = set()
    for record in records:
        _verify(root, record)
        _require(record["path"] not in paths, "duplicate artifact inventory path")
        paths.add(record["path"])
    return paths


def _content(identity):
    return {key: identity[key] for key in ("sha256", "size_bytes")}


def _request(
    operation,
    model_name,
    checkpoint_sha256,
    producer_hashes,
    device,
    points,
    geometry_points,
    backends,
):
    _require(
        operation in ("prepare", "build") and model_name == MODEL,
        "unsupported model workflow",
    )
    _require(
        isinstance(checkpoint_sha256, str)
        and re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256),
        "checkpoint SHA-256 must be an explicit lowercase digest",
    )
    _require(
        isinstance(producer_hashes, dict)
        and set(producer_hashes) == set(PRODUCER_FILES),
        "producer hashes must identify prepare.py and adapter.py",
    )
    _require(
        all(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
            for value in producer_hashes.values()
        ),
        "producer SHA-256 is invalid",
    )
    worker._device_spec(device)
    _require(
        type(points) is int
        and points >= 2
        and type(geometry_points) is int
        and geometry_points >= 2,
        "point counts must be at least two",
    )
    _require(
        isinstance(backends, list)
        and (
            backends == []
            if operation == "prepare"
            else bool(backends)
            and all(backend in ("aoti", "tensorrt") for backend in backends)
            and len(set(backends)) == len(backends)
        ),
        "GeoTransolver build requires unique aoti/tensorrt backends; prepare accepts none",
    )
    _require(
        "tensorrt" not in backends or device.startswith("cuda"),
        "TensorRT requires CUDA",
    )
    return dict(
        operation=operation,
        model_name=model_name,
        checkpoint_sha256=checkpoint_sha256,
        producer_hashes=producer_hashes,
        device=device,
        points=points,
        geometry_points=geometry_points,
        backends=backends,
    )


def _load_producer(directory):
    path = directory / "prepare.py"
    source = path.read_bytes()
    spec = importlib.util.spec_from_file_location("_pnmir_workflow_producer", path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def _preparation(root, request):
    producer = _directory(root, "producer")
    for name, digest in request["producer_hashes"].items():
        _require(
            worker._sha256(_file(producer, name)) == digest,
            "retained producer source differs from request",
        )
    archive = _file(root, "source/checkpoint.mdlus")
    _require(
        worker._sha256(archive) == request["checkpoint_sha256"],
        "retained checkpoint differs from requested SHA-256",
    )
    preparation = _directory(root, "preparation")
    report = json.loads(_file(preparation, "preparation.json").read_text())
    _require(
        report.get("status") == "complete" and report.get("format_version") == 1,
        "preparation is not complete",
    )
    _require(
        report.get("point_count") == request["points"]
        and report.get("geometry_point_count") == request["geometry_points"],
        "prepared point counts differ from request",
    )
    _require(
        report.get("environment", {}).get("device") == request["device"],
        "prepared device differs from request",
    )
    _require(
        _content(report["checkpoint"]) == _content(worker._file_identity(archive)),
        "prepared checkpoint identity differs",
    )
    for key, name in (("preparation", "prepare.py"), ("adapter", "adapter.py")):
        _require(
            _content(report["source"][key])
            == _content(worker._file_identity(root / "producer" / name)),
            "prepared source identity differs",
        )
    paths = _inventory(preparation, report.get("files"))
    reference_path = _verify(preparation, report.get("reference_manifest"))
    _require(
        reference_path == preparation / "references/manifest.json",
        "unexpected reference manifest path",
    )
    manifest = json.loads(reference_path.read_text())
    _require(
        manifest.get("format_version") == 1
        and manifest.get("reference_kind") == "upstream-full-model"
        and manifest.get("byte_order") == "little",
        "unsupported full-model reference format",
    )
    for key in ("checkpoint", "source", "model_state_sha256", "comparisons"):
        _require(
            manifest.get(key) == report.get(key),
            f"reference {key} binding differs from preparation",
        )
    prepared = _inventory(preparation, manifest.get("prepared_files"))
    _require(
        prepared
        == {"recipe.json", "adapter.py", "config.json", "checkpoint.pt", "fixtures.pt"},
        "prepared source inventory is incomplete",
    )
    _require(
        worker._sha256(preparation / "adapter.py")
        == request["producer_hashes"]["adapter.py"],
        "emitted adapter differs from verified producer",
    )
    recipe = json.loads((preparation / "recipe.json").read_text())
    _require(
        recipe.get("name") == request["model_name"], "prepared recipe model differs"
    )
    model_inputs = resolve_inputs(recipe, preparation / "recipe.json")
    _require(
        model_inputs is not None, "workflow requires prepared model input identities"
    )
    cases, comparisons = manifest.get("cases"), manifest.get("comparisons")
    _require(
        isinstance(cases, list)
        and len(cases) == 3
        and isinstance(comparisons, list)
        and len(comparisons) == 3,
        "workflow requires three full-model cases",
    )
    input_specs, output_specs = tensor_contracts(recipe)
    for index, (case, comparison) in enumerate(zip(cases, comparisons, strict=True)):
        _require(
            case.get("case") == index
            and comparison.get("case") == index
            and comparison.get("passed") is True
            and comparison.get("bitwise_equal") is True
            and comparison.get("max_abs") == 0.0,
            "full-model/cached-core preparation parity is not exact",
        )
        for kind, specs in (("inputs", input_specs), ("outputs", output_specs)):
            values = case.get(kind)
            _require(
                isinstance(values, list) and len(values) == len(specs),
                "reference tensor coverage differs from recipe",
            )
            for descriptor, spec in zip(values, specs, strict=True):
                _require(
                    all(
                        descriptor.get(key) == spec[key]
                        for key in ("name", "dtype", "shape")
                    ),
                    "reference tensor contract differs from recipe",
                )
                path = _verify(preparation, descriptor)
                _require(
                    descriptor["path"] in paths, "reference tensor is not inventoried"
                )
                count = math.prod(spec["shape"])
                _require(
                    descriptor["size_bytes"] == count * 4,
                    "reference tensor byte count differs from shape",
                )
                _require(
                    all(
                        math.isfinite(value[0])
                        for value in struct.iter_unpack("<f", path.read_bytes())
                    ),
                    "reference tensor contains nonfinite values",
                )
    return report, manifest, recipe, model_inputs


def _qualification(root, receipt, request, report):
    preparation, manifest, recipe, inputs = _preparation(root, request)
    build = _directory(root, "build")
    model = _directory(build, "model")
    _require(
        receipt.get("requested_backends") == request["backends"]
        and receipt.get("device") == request["device"],
        "native build request differs from workflow",
    )
    _require(
        receipt.get("model_inputs") == input_identities(inputs),
        "native build model input identities differ from preparation",
    )
    _require(
        receipt.get("weights", {}).get("state_sha256")
        == preparation["model_state_sha256"],
        "native build state differs from full-model state",
    )
    _require(
        receipt.get("case_count") == 3
        and set(receipt.get("variants", {})) == set(request["backends"]),
        "native build coverage differs from full-model cases",
    )
    _inventory(build, receipt["source"]["files"])
    report.update(
        format_version=1,
        passed=False,
        scope="Full upstream model to cached core to native inference on three deterministic model-space cases; not raw-mesh or scientific CFD qualification",
        limits=dict(worker.PARITY_LIMITS),
        model_state_sha256=preparation["model_state_sha256"],
        checkpoint=_content(preparation["checkpoint"]),
        reference_manifest=_content(preparation["reference_manifest"]),
        prepared_files=manifest["prepared_files"],
        runtime=_content(receipt["runtime"]),
        variants={},
    )
    for backend in request["backends"]:
        variant = receipt["variants"][backend]
        _require(
            variant.get("status") == "complete",
            "native backend verification is incomplete",
        )
        _inventory(model, variant["files"])
        check_path = _verify(build, variant["checks"])
        check = json.loads(check_path.read_text())
        _require(
            check.get("passed") is True
            and check.get("runtime") == receipt["runtime"]
            and check.get("backend") == backend
            and len(check.get("cases", [])) == 3,
            "native parity receipt is incomplete or stale",
        )
        result = {"package": variant["package"], "files": variant["files"], "cases": []}
        report["variants"][backend] = result
        for index, (reference, native) in enumerate(
            zip(manifest["cases"], check["cases"], strict=True)
        ):
            directory = _directory(build, f"checks/{backend}/case-{index}")
            _require(
                native.get("passed") is True
                and len(native.get("inputs", [])) == len(reference["inputs"]),
                "native input coverage differs",
            )
            for number, (descriptor, claimed) in enumerate(
                zip(reference["inputs"], native["inputs"], strict=True)
            ):
                _require(
                    all(
                        claimed.get(key) == descriptor[key]
                        for key in ("name", "dtype", "shape", "sha256")
                    ),
                    "native input differs from full-model cached input",
                )
                _require(
                    _content(
                        worker._file_identity(_file(directory, f"input-{number}.bin"))
                    )
                    == _content(descriptor),
                    "native input bytes differ from full-model cached input",
                )
            metadata_path = _file(directory, "native-metadata.json")
            metadata = json.loads(metadata_path.read_text())
            _require(
                metadata == native.get("metadata")
                and metadata.get("schema_version") == 1
                and metadata.get("completed") is True
                and metadata.get("backend") == backend
                and metadata.get("execution_device")
                == worker._device_spec(request["device"]),
                "native execution metadata differs or is incomplete",
            )
            _require(
                len(metadata.get("outputs", []))
                == len(reference["outputs"])
                == len(native.get("outputs", [])),
                "native output coverage differs",
            )
            metrics = []
            for number, (descriptor, actual_metadata, previous) in enumerate(
                zip(
                    reference["outputs"],
                    metadata["outputs"],
                    native["outputs"],
                    strict=True,
                )
            ):
                expected = {key: descriptor[key] for key in ("name", "dtype", "shape")}
                expected["data"] = _verify(
                    root / "preparation", descriptor
                ).read_bytes()
                actual = _file(directory, f"output-{number}.bin").read_bytes()
                metric = worker._compare_output(actual, expected, actual_metadata)
                _require(
                    previous.get("actual_sha256") == metric["actual_sha256"]
                    and previous.get("reference_sha256") == metric["reference_sha256"],
                    "native parity reference differs from upstream full-model output",
                )
                metrics.append(metric)
            result["cases"].append(
                {
                    "case": index,
                    "passed": True,
                    "native_metadata": _content(worker._file_identity(metadata_path)),
                    "outputs": metrics,
                }
            )
    report["passed"] = True
    return report


def _pre_release(root, request):
    def check(output, receipt):
        report = {"format_version": 1, "passed": False}
        path = output / "model/qualification/full-model.json"
        try:
            _qualification(root, receipt, request, report)
        except BaseException as error:
            report["error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            worker._write_json(path, report)
        return {"passed": True, "report": worker._file_identity(path, output / "model")}

    return check


def _run(
    operation,
    model_name,
    producer_dir,
    checkpoint,
    output,
    *,
    checkpoint_sha256,
    device,
    points,
    geometry_points,
    runtime=None,
    backends=None,
    expected_identity=None,
    required_gpu_arch=None,
):
    from pnmir_build.targets import validate_target

    validate_target(device, required_gpu_arch)
    producer_dir = Path(producer_dir).expanduser().resolve(strict=True)
    hashes = {
        name: worker._sha256(_file(producer_dir, name)) for name in PRODUCER_FILES
    }
    if expected_identity is not None:
        _require(
            type(expected_identity) is dict
            and not set(expected_identity) - {"producer", "runtime"},
            "unsupported expected workflow identity fields",
        )
        if "producer" in expected_identity:
            _require(
                expected_identity["producer"] == hashes,
                "project producer identity differs from selected workflow inputs",
            )
        _require(
            operation == "build" or "runtime" not in expected_identity,
            "runtime identity requires a build workflow",
        )
    request = _request(
        operation,
        model_name,
        checkpoint_sha256,
        hashes,
        device,
        points,
        geometry_points,
        backends,
    )
    checkpoint = Path(checkpoint).expanduser().absolute()
    _require(
        not checkpoint.is_symlink()
        and checkpoint.is_file()
        and checkpoint.suffix == ".mdlus",
        "checkpoint must select a regular .mdlus file",
    )
    _require(
        worker._sha256(checkpoint) == checkpoint_sha256, "checkpoint SHA-256 mismatch"
    )
    root = Path(output).expanduser().absolute()
    root.mkdir(parents=True, exist_ok=False)
    receipt = {
        "format_version": 1,
        "operation": operation,
        "status": "preparing",
        "request": request,
    }
    worker._write_json(root / "workflow.json", receipt)
    try:
        (root / "producer").mkdir()
        for name, digest in hashes.items():
            shutil.copyfile(producer_dir / name, root / "producer" / name)
            _require(
                worker._sha256(root / "producer" / name) == digest,
                "producer source changed while retaining it",
            )
        (root / "source").mkdir()
        retained_checkpoint = root / "source/checkpoint.mdlus"
        shutil.copyfile(checkpoint, retained_checkpoint)
        _require(
            worker._sha256(retained_checkpoint) == checkpoint_sha256,
            "checkpoint changed while retaining it",
        )
        producer = _load_producer(root / "producer")
        producer.prepare(
            retained_checkpoint,
            root / "preparation",
            device=device,
            points=points,
            geometry_points=geometry_points,
            adapter_path=root / "producer/adapter.py",
        )
        _preparation(root, request)
        receipt.update(
            status="prepared",
            preparation=worker._file_identity(
                root / "preparation/preparation.json", root
            ),
            recipe=worker._file_identity(root / "preparation/recipe.json", root),
        )
        worker._write_json(root / "workflow.json", receipt)
        if operation == "build":
            receipt["status"] = "building"
            worker._write_json(root / "workflow.json", receipt)
            built = worker.execute_build(
                root / "preparation/recipe.json",
                root / "build",
                backends,
                device,
                Path(runtime),
                pre_release_check=_pre_release(root, request),
                **(
                    {"expected_identity": {"runtime": expected_identity["runtime"]}}
                    if expected_identity is not None and "runtime" in expected_identity
                    else {}
                ),
                **(
                    {"required_gpu_arch": required_gpu_arch}
                    if required_gpu_arch is not None
                    else {}
                ),
            )
            receipt.update(
                status="complete",
                build=worker._file_identity(root / "build/build.json", root),
                release=worker._file_identity(
                    root / "build/model/model-release.json", root
                ),
                qualification=built["qualification"],
            )
        worker._write_json(root / "workflow.json", receipt)
        validate_workflow_completion(root, **request)
        return receipt
    except BaseException as error:
        receipt.update(
            status="failed", error={"type": type(error).__name__, "message": str(error)}
        )
        worker._write_json(root / "workflow.json", receipt)
        raise


def prepare_workflow(
    model_name,
    producer_dir,
    checkpoint,
    output,
    *,
    checkpoint_sha256,
    device="cuda",
    points=32,
    geometry_points=64,
    expected_identity=None,
    required_gpu_arch=None,
):
    return _run(
        "prepare",
        model_name,
        producer_dir,
        checkpoint,
        output,
        checkpoint_sha256=checkpoint_sha256,
        device=device,
        points=points,
        geometry_points=geometry_points,
        backends=[],
        expected_identity=expected_identity,
        required_gpu_arch=required_gpu_arch,
    )


def build_workflow(
    model_name,
    producer_dir,
    checkpoint,
    output,
    *,
    checkpoint_sha256,
    runtime,
    backends=None,
    device="cuda",
    points=32,
    geometry_points=64,
    expected_identity=None,
    required_gpu_arch=None,
):
    return _run(
        "build",
        model_name,
        producer_dir,
        checkpoint,
        output,
        checkpoint_sha256=checkpoint_sha256,
        runtime=runtime,
        backends=["aoti"] if backends is None else backends,
        device=device,
        points=points,
        geometry_points=geometry_points,
        expected_identity=expected_identity,
        required_gpu_arch=required_gpu_arch,
    )


def validate_workflow_completion(output, **expected):
    """Verify selected inputs, retained bytes and qualification without Torch."""
    request = _request(**expected)
    root = Path(output).expanduser().absolute()
    _require(
        root.is_dir() and not root.is_symlink(),
        "workflow output directory is missing or a symlink",
    )
    receipt = json.loads(_file(root, "workflow.json").read_text())
    _require(
        receipt.get("format_version") == 1
        and receipt.get("request") == request
        and receipt.get("operation") == request["operation"],
        "workflow request identity differs from the expected operation or inputs",
    )
    status = "prepared" if request["operation"] == "prepare" else "complete"
    _require(receipt.get("status") == status, f"workflow is not {status}")
    _require(
        _verify(root, receipt.get("preparation"))
        == root / "preparation/preparation.json",
        "unexpected preparation receipt path",
    )
    _require(
        _verify(root, receipt.get("recipe")) == root / "preparation/recipe.json",
        "unexpected prepared recipe path",
    )
    _, _, recipe, model_inputs = _preparation(root, request)
    if request["operation"] == "prepare":
        _require(
            not (root / "build").exists(),
            "prepare-only workflow must not contain a candidate build",
        )
        return receipt

    _require(
        _verify(root, receipt.get("build")) == root / "build/build.json",
        "unexpected build receipt path",
    )
    _require(
        _verify(root, receipt.get("release"))
        == root / "build/model/model-release.json",
        "unexpected model release path",
    )
    from pnmir_build.cli import _validate_container_completion

    _validate_container_completion(
        {
            "output": root / "build",
            "recipe": recipe,
            "model_inputs": model_inputs,
            "device": request["device"],
            "backends": request["backends"],
        }
    )
    build = json.loads((root / "build/build.json").read_text())
    release = json.loads((root / "build/model/model-release.json").read_text())
    qualification = build.get("qualification")
    _require(
        isinstance(qualification, dict)
        and qualification.get("passed") is True
        and qualification
        == receipt.get("qualification")
        == release.get("qualification"),
        "workflow/build/release qualification identities differ",
    )
    path = _verify(root / "build/model", qualification.get("report"))
    _require(
        path == root / "build/model/qualification/full-model.json",
        "unexpected full-model qualification report path",
    )
    recorded = json.loads(path.read_text())
    recomputed = _qualification(root, build, request, {})
    _require(
        recorded == recomputed,
        "full-model qualification report differs from the current source, reference, native output or artifact bytes",
    )
    return receipt
