"""Local adoption of the existing compiled Transolver schema-one runner.

This adapter imports compiled packages; it does not submit managed build jobs or
reconstruct the Python model. Its service uses the existing dedicated C++ Runner.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid

from .common import (
    contained,
    digest,
    file_identity,
    inventory,
    read_json,
    require,
    write_json,
)

WORKFLOW = "physicsnemo-cfd-transolver-native"
POINTS = {"full": 60000, "tail": 8095}


def resolve_project(project, repository, *, require_sources=True):
    """Normalize the domain contract without opening a model or loading CUDA."""
    project = dict(project)
    build = project["build"]
    require(
        project["runtime"] == "cpp" and project["name"] == WORKFLOW,
        "CFD requires its existing native workflow identifier and cpp runtime",
    )
    require(
        build.get("environment", "local") == "local",
        "CFD compiled import builds require build.environment: local",
    )
    require(
        build.get("target", "current-gpu") == "current-gpu",
        "CFD build.target must be current-gpu",
    )
    backend = build.get("backend", "aoti")
    require(backend in ("aoti", "tensorrt"), "CFD backend must be aoti or tensorrt")
    require(
        not project.get("models"),
        "CFD compiled import does not compile authored models; select workflow.deployment",
    )
    workflow = dict(project["workflow"])
    require(set(workflow) <= {"template", "deployment"}, "unknown CFD workflow option")
    if require_sources and workflow.get("deployment"):
        selected = Path(workflow["deployment"])
        workflow["deployment"] = str(_regular(Path(project["root"]) / selected))
    project.update(backend=backend, workflow=workflow, models={})
    return project


def native_document(root, repository, manifest, release, *, profile="release.yaml"):
    """Keep the pinned legacy CFD preset; explicitly select a local import."""
    require(
        release.get("provider_profile") == "physicsnemo-cfd-surface-transolver-run1-v1"
        and release.get("models")
        == {
            "full": {"preset": "transolver-surface", "points": 60000},
            "tail": {"preset": "transolver-surface", "points": 8095},
        },
        "CFD adoption requires the pinned full60000/tail8095 Transolver preset",
    )
    backend = release.get("backend", "aoti")
    require(backend in ("aoti", "tensorrt"), "CFD backend must be aoti or tensorrt")
    images = release.get("images", {})
    deployment = release.get("local", {}).get("deployment") or os.environ.get(
        "NATIVE_DEPLOYMENT_CONFIG"
    )
    return {
        "schema_version": 1,
        "name": WORKFLOW,
        "runtime": "cpp",
        "models": {},
        "workflow": {
            "template": "transolver_surface",
            **({"deployment": deployment} if deployment else {}),
        },
        "build": {"backend": backend, "environment": "local", "target": "current-gpu"},
        "serve": {
            "memory_mb": manifest["resources"]["defaults"]["memory_mb"],
            **({"image": images["server"]} if images.get("server") else {}),
        },
        "_legacy_native": {"manifest": "plugin.yaml", "profile": profile},
    }


def _regular(path):
    path = Path(path)
    require(
        path.is_file() and not path.is_symlink(), f"CFD requires a regular file: {path}"
    )
    return path.resolve()


def _startup_files(path):
    fixture = read_json(path)
    models = fixture.get("models")
    require(
        fixture.get("schema_version") == 1
        and isinstance(models, list)
        and len(models) == 2
        and {item.get("point_count") for item in models} == set(POINTS.values()),
        "CFD startup goldens must cover full60000 and tail8095",
    )
    files = {path}
    for model in models:
        points = model["point_count"]
        require(
            all(
                type(model.get(key)) in (int, float)
                and math.isfinite(model[key])
                and model[key] >= 0
                for key in ("rtol", "atol")
            ),
            "CFD startup tolerances must be finite and nonnegative",
        )
        for tensor, width in (
            (model["inputs"]["fx"], 2),
            (model["inputs"]["embedding"], 6),
            (model["expected_output"], 4),
        ):
            selected = _regular(contained(path.parent, tensor["path"]))
            require(
                tensor.get("shape") == [1, points, width]
                and selected.stat().st_size == points * width * 4,
                "CFD startup tensor shape or byte count differs",
            )
            files.add(selected)
    return files


def _driver_version(value):
    require(
        isinstance(value, str) and re.fullmatch(r"[0-9]+(?:\.[0-9]+){0,3}", value),
        "CFD driver version must be numeric",
    )
    parts = tuple(int(part) for part in value.split("."))
    require(all(part <= 0xFFFFFFFF for part in parts), "CFD driver version is invalid")
    return parts + (0,) * (4 - len(parts))


def _deployment_contract(config):
    """Reject imports the existing deny-unknown-fields schema-one worker rejects."""
    require(
        set(config)
        == {
            "schema_version",
            "workflow_id",
            "executor_class",
            "backend",
            "device_index",
            "compute_capability",
            "target",
            "packages",
            "model_build_sources",
            "stats_path",
            "startup_fixture_path",
            "model_assets",
            "block_size",
            "artifact_root",
            "input_cache",
            "case",
            "request_timeout_seconds",
            "drain_timeout_seconds",
        },
        "CFD import requires the complete schema-one deployment contract",
    )
    require(
        type(config["device_index"]) is int
        and 0 <= config["device_index"] <= 0xFFFFFFFF,
        "CFD device index must be a nonnegative uint32",
    )
    for key in ("request_timeout_seconds", "drain_timeout_seconds"):
        require(
            type(config[key]) is int and 0 < config[key] <= 0xFFFFFFFFFFFFFFFF,
            "CFD deployment timeouts must be positive uint64 values",
        )
    case = config["case"]
    require(
        isinstance(case, dict)
        and set(case) == {"case_id", "point_count", "mesh", "geometry"},
        "CFD import requires the pinned case contract",
    )
    for key in ("mesh", "geometry"):
        item = case[key]
        require(
            isinstance(item, dict)
            and set(item) == {"url", "sha256", "size_bytes"}
            and isinstance(item["url"], str)
            and item["url"].startswith("https://")
            and isinstance(item["sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
            and type(item["size_bytes"]) is int
            and item["size_bytes"] > 0,
            "CFD case inputs require HTTPS and explicit byte identities",
        )
    target = config["target"]
    require(
        isinstance(target, dict)
        and set(target) == {"profile", "compilation", "deployment"}
        and target["profile"] == "h100-sxm-80gb-linux-amd64-v1",
        "CFD requires its explicit qualified H100 target policy",
    )
    require(
        target["compilation"]
        == {
            "platform": "linux/amd64",
            "gpu_product": "NVIDIA H100 80GB HBM3",
            "compute_capability": [9, 0],
            "cuda_architecture": "sm90",
            "precision": "fp32",
            "mig": "disabled",
            "hardware_compatibility": "qualified-product-only",
        },
        "CFD compilation target differs from its qualified H100 policy",
    )
    deployment = target["deployment"]
    require(
        isinstance(deployment, dict)
        and set(deployment)
        == {"minimum_available_memory_mb", "mig", "driver_requirements"}
        and deployment["mig"] == "disabled"
        and type(deployment["minimum_available_memory_mb"]) is int
        and deployment["minimum_available_memory_mb"] > 0,
        "CFD deployment requires explicit available memory and disabled MIG",
    )
    drivers = deployment["driver_requirements"]
    require(
        isinstance(drivers, dict)
        and set(drivers)
        in (
            {"minimum_driver_version"},
            {"minimum_driver_version", "maximum_driver_version"},
        ),
        "CFD requires explicit driver version requirements",
    )

    minimum = _driver_version(drivers["minimum_driver_version"])
    if drivers.get("maximum_driver_version") is not None:
        require(
            _driver_version(drivers["maximum_driver_version"]) >= minimum,
            "CFD driver version range is empty",
        )


def _admit_import_environment(source, environment):
    target = source["configuration"]["target"]
    gpu = environment["gpu"]
    require(
        gpu.get("name") == target["compilation"]["gpu_product"]
        and gpu.get("architecture") == target["compilation"]["cuda_architecture"],
        "CFD selected GPU differs from the imported target",
    )
    require(
        str(gpu.get("mig_mode", "")).lower() == "disabled",
        "CFD target requires MIG disabled",
    )
    drivers = target["deployment"]["driver_requirements"]
    actual = _driver_version(gpu.get("driver_version"))
    require(
        actual >= _driver_version(drivers["minimum_driver_version"])
        and (
            not drivers.get("maximum_driver_version")
            or actual <= _driver_version(drivers["maximum_driver_version"])
        ),
        "CFD selected driver differs from the imported target policy",
    )


def import_deployment(project):
    """Verify local packages and every declared byte without importing frameworks."""
    selection = project.get("workflow", {}).get("deployment") or os.environ.get(
        "NATIVE_DEPLOYMENT_CONFIG"
    )
    require(
        selection,
        "CFD build requires an explicit compiled deployment import; new model compilation is not performed by this import adapter",
    )
    path = Path(selection)
    if not path.is_absolute():
        path = Path(project["root"]) / path
    path = _regular(path)
    config = read_json(path)
    _deployment_contract(config)
    backend = project["backend"]
    require(
        config.get("schema_version") == 1
        and config.get("workflow_id") == WORKFLOW
        and config.get("backend", "aoti") == backend
        and config.get("executor_class") == f"native.transolver.{backend}",
        "CFD deployment/workflow/backend differs from the application",
    )
    require(
        config.get("block_size") == 60000
        and config.get("case", {}).get("case_id") == "run_1"
        and config["case"].get("point_count") == 8828095,
        "CFD import requires the complete pinned run_1 block contract",
    )
    require(
        config.get("compute_capability") == [9, 0]
        and isinstance(config.get("target"), dict),
        "CFD import requires its explicit qualified H100 target policy",
    )
    sources = config.get("model_build_sources")
    require(
        isinstance(sources, dict)
        and set(sources) == set(POINTS)
        and all(
            isinstance(value, str) and re.fullmatch(r"[a-f0-9]{40}", value)
            for value in sources.values()
        ),
        "CFD import requires original per-alias compilation sources",
    )
    packages, required = {}, set()
    require(
        isinstance(config.get("packages"), list) and len(config["packages"]) == 2,
        "CFD import requires exactly two compiled packages",
    )
    for value in config["packages"]:
        package = Path(value)
        require(
            package.is_absolute() and package.is_dir() and not package.is_symlink(),
            "CFD package must be a local directory",
        )
        package = package.resolve()
        manifest = read_json(_regular(package / "model.json"))
        inputs = {item.get("name"): item for item in manifest.get("inputs", [])}
        fx = inputs.get("fx", {})
        shape = fx.get("shape", [])
        require(
            len(shape) == 3
            and shape[0] == 1
            and shape[1] in POINTS.values()
            and shape[2] == 2,
            "CFD fx requires the full60000 or tail8095 static shape",
        )
        points = shape[1]
        outputs = manifest.get("outputs", [])
        require(
            set(inputs) == {"fx", "embedding"}
            and inputs["embedding"].get("shape") == [1, points, 6]
            and len(outputs) == 1
            and outputs[0].get("shape") == [1, points, 4]
            and all(
                item.get("dtype") == "float32" for item in (*inputs.values(), *outputs)
            ),
            "CFD package requires the original FP32 tensor contract",
        )
        artifacts = manifest.get("artifacts", [])
        require(
            len(artifacts) == 1
            and all(
                artifacts[0].get(key) == expected
                for key, expected in (
                    ("backend", backend),
                    ("target", "cuda"),
                    ("precision", "fp32"),
                )
            ),
            "CFD package backend/precision differs",
        )
        require(
            artifacts[0].get("correctness_profile", {}).get("name")
            == (
                "aten-boundary-exact-v2"
                if backend == "aoti"
                else "provider-default-exact-v1"
            ),
            "CFD requires its original exact math profile",
        )
        _regular(contained(package, artifacts[0]["path"]))
        alias = next(name for name, count in POINTS.items() if count == points)
        require(alias not in packages, "CFD package shape is duplicated")
        packages[alias] = package
        for item in package.rglob("*"):
            require(not item.is_symlink(), "CFD package links are unsupported")
            if item.is_file():
                required.add(item.resolve())
    require(set(packages) == set(POINTS), "CFD requires both full and tail packages")
    stats = _regular(config["stats_path"])
    require(
        config.get("startup_fixture_path"),
        "CFD import requires independently authored startup goldens",
    )
    startup = _regular(config["startup_fixture_path"])
    startup_files = _startup_files(startup)
    required |= {stats} | startup_files
    declared = {}
    for item in config.get("model_assets", []):
        require(
            set(item) == {"path", "sha256", "size_bytes"}
            and type(item["size_bytes"]) is int
            and item["size_bytes"] > 0,
            "CFD model assets require explicit nonempty byte identities",
        )
        selected = _regular(item["path"])
        require(
            selected not in declared
            and file_identity(selected)
            == {key: item[key] for key in ("sha256", "size_bytes")},
            "CFD asset bytes changed or inventory contains a duplicate",
        )
        declared[selected] = item
    require(
        set(declared) == required,
        "CFD asset inventory must cover every package/statistics/startup file exactly",
    )
    return {
        "configuration": config,
        "deployment": path,
        "packages": packages,
        "stats": stats,
        "startup": startup,
        "startup_files": startup_files,
        "identities": {
            selected: {key: item[key] for key in ("sha256", "size_bytes")}
            for selected, item in declared.items()
        },
        "model_build_sources": sources,
        "recompiled": False,
    }


def assemble(project, source, workflow, output, *, repository, environment, source_sha):
    """Freeze an imported CFD payload into the common immutable bundle format."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    mapping = {}
    for alias, package in source["packages"].items():
        for original in package.rglob("*"):
            if original.is_file():
                mapping[original] = (
                    Path("models") / alias / original.relative_to(package)
                )
    mapping[source["stats"]] = Path("stats/global_stats.json")
    for original in source["startup_files"]:
        mapping[original] = Path("startup") / original.relative_to(
            source["startup"].parent
        )
    for original, relative in mapping.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected = source["identities"][original]
        require(
            file_identity(_regular(original)) == expected,
            "CFD payload changed after import verification",
        )
        shutil.copy2(original, destination)
        require(
            file_identity(destination) == expected,
            "CFD payload changed during bundle capture",
        )
    config = copy.deepcopy(source["configuration"])
    config["packages"] = [f"models/{alias}" for alias in POINTS]
    config["stats_path"] = "stats/global_stats.json"
    config["startup_fixture_path"] = "startup/" + source["startup"].name
    config["model_assets"] = [
        {
            "path": mapping[Path(item["path"]).resolve()].as_posix(),
            "sha256": item["sha256"],
            "size_bytes": item["size_bytes"],
        }
        for item in config["model_assets"]
    ]
    write_json(output / "configuration/deployment.json", config)
    shutil.copytree(workflow["install_root"], output / "workflow", symlinks=True)
    plugin = output / "configuration/plugin" / WORKFLOW
    plugin.mkdir(parents=True)
    import yaml

    manifest = yaml.safe_load(
        (Path(repository) / "plugins" / WORKFLOW / "plugin.yaml").read_text()
    )
    manifest["runtime"]["executor_class"] = config["executor_class"]
    for stage in manifest["pipeline"]["stages"]:
        if stage["phase"] == "execute":
            stage["queue"] = "execute." + config["executor_class"]
    (plugin / "plugin.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    shutil.copy2(
        Path(repository) / "native-inference/deploy/worker_runtime_config.json",
        output / "configuration/worker_runtime_config.json",
    )
    descriptor = {
        "schema_version": 1,
        "name": WORKFLOW,
        "runtime": "cpp",
        "adapter": "transolver_surface",
        "backend": project["backend"],
        "source_sha": source_sha,
        "model_build_sources": source["model_build_sources"],
        "models": {alias: f"models/{alias}" for alias in POINTS},
        "executor_class": config["executor_class"],
        "workflow": {
            "config": "configuration/deployment.json",
            "runner": "workflow/bin/physicsnemo-transolver",
        },
        "environment": {
            key: value for key, value in environment.items() if key != "process_env"
        },
    }
    write_json(output / "bundle.json", descriptor)
    files = inventory(output)
    seal = {"schema_version": 1, "files": files, "bundle_digest": digest(files)}
    write_json(output / "inventory.json", seal)
    return seal


def materialize_deployment(bundle, output, *, artifact_root, input_cache):
    from .bundle import verify

    bundle = Path(bundle).resolve()
    verify(bundle)
    descriptor = read_json(bundle / "bundle.json")
    config = read_json(contained(bundle, descriptor["workflow"]["config"]))
    for name in ("stats_path", "startup_fixture_path"):
        config[name] = str(contained(bundle, config[name]))
    config["packages"] = [str(contained(bundle, name)) for name in config["packages"]]
    for item in config["model_assets"]:
        item["path"] = str(contained(bundle, item["path"]))
    config.update(
        artifact_root=str(Path(artifact_root).resolve()),
        input_cache=str(Path(input_cache).resolve()),
        device_index=0,
    )
    write_json(Path(output) / "deployment.json", config)
    write_json(
        Path(output) / "release-inventory.json",
        {"bundle_digest": read_json(bundle / "inventory.json")["bundle_digest"]},
    )
    return config


def check_project(project, *, repository, config_only=False, **options):
    if not project.get("workflow", {}).get("deployment") and not os.environ.get(
        "NATIVE_DEPLOYMENT_CONFIG"
    ):
        require(
            config_only,
            "CFD check requires a compiled deployment import; model compilation has not been performed",
        )
        return {
            "status": "configuration-ok",
            "runtime": "cpp",
            "adapter": "transolver_surface",
            "model_execution": "not_run",
            "recompiled": False,
        }
    source = import_deployment(project)
    return {
        "status": "configuration-ok" if config_only else "checked",
        "runtime": "cpp",
        "adapter": "transolver_surface",
        "model_build_sources": source["model_build_sources"],
        "model_execution": "not_run",
        "recompiled": False,
    }


def build_workflow(project, output, *, repository, environment):
    """Compile the dedicated domain runner; generic host linkage loads no models."""
    from . import compatibility, native

    python = Path(environment["python"])
    selected = environment.get("native_environment", {})
    tensorrt = project.get("build", {}).get("tensorrt_root")
    generic = native.build_workflow(
        {**project, "workflow": {"template": "tensor"}},
        Path(output) / "generic",
        repository=Path(repository),
        sdk_prefix=Path(environment["sdk_prefix"]),
        python=python,
        backend=project["backend"],
        environment=selected,
        tensorrt_root=Path(tensorrt) if tensorrt else None,
        build_environment=environment,
    )
    probe = subprocess.run(
        [
            str(python),
            "-c",
            "import json,pathlib,torch,warp; print(json.dumps([torch.utils.cmake_prefix_path,str(pathlib.Path(warp.__file__).parent)]))",
        ],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, **selected},
    )
    torch_prefix, warp = json.loads(probe.stdout)
    build = Path(output) / "transolver"
    installed, log = (
        Path(generic["install_root"]),
        Path(output) / "transolver-build.log",
    )
    args = [
        "cmake",
        "-S",
        str(Path(repository) / "native-inference/workflows/transolver"),
        "-B",
        str(build),
        f"-DCMAKE_PREFIX_PATH={environment['sdk_prefix']};{torch_prefix}",
        f"-DPNMIR_WARP_ROOT={warp}",
        f"-DCMAKE_INSTALL_PREFIX={installed}",
        "-DCMAKE_INSTALL_LIBDIR=lib",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_TESTING=OFF",
        f"-DPNMIR_ENABLE_AOTI={'ON' if project['backend'] == 'aoti' else 'OFF'}",
        f"-DPNMIR_ENABLE_TENSORRT={'ON' if project['backend'] == 'tensorrt' else 'OFF'}",
    ]
    args.append("-DCMAKE_CUDA_ARCHITECTURES=90")
    toolkit = selected.get("CUDAToolkit_ROOT") or selected.get("CUDA_HOME")
    if toolkit:
        args.append(f"-DCUDA_TOOLKIT_ROOT_DIR={Path(toolkit).expanduser().absolute()}")
    if tensorrt:
        args.append(f"-DPNMIR_TENSORRT_ROOT={Path(tensorrt).resolve()}")
    native._execute(args, log=log, environment={**os.environ, **selected})
    native._execute(
        ["cmake", "--build", str(build), "--parallel", "4"],
        log=log,
        environment={**os.environ, **selected},
    )
    native._execute(
        ["cmake", "--install", str(build)],
        log=log,
        environment={**os.environ, **selected},
    )
    library = installed / "lib/libtransolver_workflow.so"
    require(
        library.is_file() and (installed / "bin/physicsnemo-transolver").is_file(),
        "Transolver install omitted its dedicated runner/library",
    )
    observed = compatibility.capture(
        python,
        library,
        environment={**os.environ, **selected},
        output=Path(output) / "transolver-compatibility.json",
    )
    return {
        **generic,
        "runner": str(installed / "bin/physicsnemo-transolver"),
        "library": str(library),
        "library_name": library.name,
        "compatibility": observed,
        "reused": False,
    }


def _runner_arguments(bundle, config):
    descriptor = read_json(bundle / "bundle.json")
    values = [
        str(contained(bundle, descriptor["workflow"]["runner"])),
        "--backend",
        descriptor["backend"],
        "--domain",
        "surface",
        "--block-size",
        "60000",
        "--stats",
        config["stats_path"],
        "--math-profile",
        "provider-default-v1",
        "--startup-fixture",
        config["startup_fixture_path"],
    ]
    for package in config["packages"]:
        values += ["--package", package]
    return values


def _runner_environment(bundle, environment=None):
    selected = {**os.environ, **(environment or {})}
    loader = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    workflow = Path(bundle).resolve() / "workflow"
    # LD_LIBRARY_PATH precedes a runner's RUNPATH. An image can also contain an
    # older backend-specific Transolver library with the same shared name.
    selected[loader] = os.pathsep.join(
        dict.fromkeys(
            filter(
                None,
                [
                    str(workflow / "lib"),
                    str(workflow / "lib64"),
                    *selected.get(loader, "").split(os.pathsep),
                ],
            )
        )
    )
    return selected


def _startup(report):
    require(
        report.get("status") == "passed"
        and report.get("policy") == "expected-fp32-v1"
        and report.get("session_load_count") == 2
        and {model.get("point_count") for model in report.get("models", [])}
        == set(POINTS.values()),
        "CFD startup did not verify both resident sessions against independent goldens",
    )


def run_bundle(
    bundle, input, output, *, inputs=None, environment=None, request_id=None
):
    """Run the dedicated C++ consumer once; no Python inference or model rebuild."""
    from . import native
    from .bundle import verify

    bundle = Path(bundle).resolve()
    verify(bundle)
    descriptor = read_json(bundle / "bundle.json")
    parameters = read_json(input)
    require(
        set(parameters) <= {"case_id", "seed", "capture_inputs"}
        and parameters.get("case_id") == "run_1",
        "CFD request must select the pinned run_1 case",
    )
    seed, capture = parameters.get("seed", 42), parameters.get("capture_inputs", False)
    require(
        type(seed) is int and 0 <= seed <= 0xFFFFFFFF and type(capture) is bool,
        "CFD seed/capture_inputs types are invalid",
    )
    portable = read_json(contained(bundle, descriptor["workflow"]["config"]))
    supplied = inputs or {}
    require(
        set(supplied) == {"mesh", "geometry"},
        "CFD direct execution requires --input-file mesh=PATH and geometry=PATH for the pinned case bytes",
    )
    admitted = {}
    for name, raw in supplied.items():
        path = _regular(raw)
        require(
            file_identity(path)
            == {key: portable["case"][name][key] for key in ("sha256", "size_bytes")},
            "CFD input bytes differ from the pinned case inventory: " + name,
        )
        admitted[name] = path
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = materialize_deployment(
        bundle,
        output / ".pnms-run",
        artifact_root=output,
        input_cache=output / ".inputs",
    )
    arguments = _runner_arguments(bundle, config)
    physical, standardized, metadata, startup = [
        output / name
        for name in (
            "physical.bin",
            "standardized.bin",
            "metadata.json",
            "startup.json",
        )
    ]
    effective_seed = (
        int.from_bytes(hashlib.sha256(f"{seed}\0run_1".encode()).digest()[:8], "little")
        & 0x7FFFFFFFFFFFFFFF
    )
    arguments += [
        "--mesh",
        str(admitted["mesh"]),
        "--stl",
        str(admitted["geometry"]),
        "--seed",
        str(effective_seed),
        "--physical-output",
        str(physical),
        "--standardized-output",
        str(standardized),
        "--metadata",
        str(metadata),
        "--startup-report",
        str(startup),
    ]
    if capture:
        arguments += ["--dump-input-dir", str(output / "inputs")]
    log = output / "native.log"
    native._execute(
        arguments, log=log, environment=_runner_environment(bundle, environment)
    )
    _startup(read_json(startup))
    observed = read_json(metadata)
    require(
        observed.get("point_count") == 8828095
        and observed.get("session_load_count") == 2
        and observed.get("math_profile") == "provider-default-v1"
        and observed.get("permutation_seed") == effective_seed,
        "CFD execution point count, resident sessions or scientific math profile differs",
    )
    artifacts = []
    for path, name, media in (
        (physical, "physical_fields", "application/octet-stream"),
        (standardized, "standardized_fields", "application/octet-stream"),
        (metadata, "native_metadata", "application/json"),
    ):
        _regular(path)
        if path != metadata:
            require(
                path.stat().st_size == 8828095 * 4 * 4,
                "CFD output byte count differs from the source-order FP32 fields",
            )
        artifacts.append(
            {
                "name": name,
                "path": str(path),
                "media_type": media,
                "primary": path == physical,
                **file_identity(path),
            }
        )
    if capture:
        for name, width in (
            ("fx.f32", 8),
            ("embedding.f32", 24),
            ("permutation.i64", 8),
            ("source_cell_ids.i64", 8),
        ):
            path = _regular(output / "inputs" / name)
            require(
                path.stat().st_size == 8828095 * width,
                "CFD captured input byte count differs: " + name,
            )
            artifacts.append(
                {
                    "name": name,
                    "path": str(path),
                    "media_type": "application/octet-stream",
                    "primary": False,
                    **file_identity(path),
                }
            )
    payload = {
        "output_path": str(physical),
        "artifacts": artifacts,
        "native": {
            "schema_version": 1,
            "runtime": "cpp",
            "case_id": "run_1",
            "run_seed": seed,
            "permutation_seed": effective_seed,
            "backend": descriptor["backend"],
            "block_size": 60000,
            "mesh_sha256": portable["case"]["mesh"]["sha256"],
            "geometry_sha256": portable["case"]["geometry"]["sha256"],
            "source_sha": descriptor["source_sha"],
            "model_build_sources": descriptor["model_build_sources"],
            "metadata": observed,
        },
    }
    result = {"request_id": request_id or str(uuid.uuid4()), "payload": payload}
    result_path = output / "result.json"
    write_json(result_path, result)
    return {
        "output_dir": str(output),
        "result": result,
        "result_path": str(result_path),
        "startup_path": str(startup),
        "log": str(log),
    }


def build_application(
    project,
    *,
    repository,
    image=None,
    push=False,
    publish_result=True,
    refresh_lock=False,
):
    from . import build, environment_lock, model, native

    require(
        project.get("build", {}).get("environment", "local") == "local",
        "CFD compiled import builds use the local environment",
    )
    source = import_deployment(project)
    previous_lock = environment_lock.capture(project)
    root = Path(project["root"]) / ".pnms/builds" / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    revision = model._source_revision(repository)
    result = {
        "schema_version": 1,
        "build_id": root.name,
        "name": WORKFLOW,
        "runtime": "cpp",
        "adapter": "transolver_surface",
        "source_sha": revision,
        "application": copy.deepcopy(project.get("document", {})),
        "status": "building",
        "recompiled": False,
        "model_build_sources": source["model_build_sources"],
        "checks": {
            "model": "not_run",
            "application": "not_run",
            "http": "not_run",
            "scientific": "not_run",
        },
    }
    try:
        options = {
            key: value
            for key, value in project.get("build", {}).items()
            if key in ("python", "runtime", "sdk_prefix", "gpu")
        }
        environment = model.prepare_environment(
            project, root / "environment", repository=repository, **options
        )
        _admit_import_environment(source, environment)
        result["environment"] = {
            key: value for key, value in environment.items() if key != "process_env"
        }
        result["environment_lock"] = environment_lock.check(
            project, result["environment"], refresh=refresh_lock
        )
        workflow = build_workflow(
            project,
            root / "workflow-build",
            repository=repository,
            environment=environment,
        )
        result["workflow"] = {
            key: value for key, value in workflow.items() if key != "loader_environment"
        }
        result["loader_environment"] = workflow["loader_environment"]
        seal = assemble(
            project,
            source,
            workflow,
            root / "bundle",
            repository=repository,
            environment=environment,
            source_sha=revision,
        )
        result.update(
            bundle="bundle",
            bundle_digest=seal["bundle_digest"],
            models={
                alias: {
                    "package": f"bundle/models/{alias}",
                    "output_root": f"bundle/models/{alias}",
                    "compilation_source_sha": source["model_build_sources"][alias],
                    "reused": True,
                    "recompiled": False,
                }
                for alias in POINTS
            },
        )
        materialized = materialize_deployment(
            root / "bundle",
            root / "checks",
            artifact_root=root / "checks/artifacts",
            input_cache=root / "checks/inputs",
        )
        startup = root / "checks/startup.json"
        native._execute(
            _runner_arguments(root / "bundle", materialized)
            + ["--startup-only", "true", "--startup-report", str(startup)],
            log=root / "checks/native.log",
            environment=_runner_environment(
                root / "bundle",
                {
                    **workflow["loader_environment"],
                    "CUDA_VISIBLE_DEVICES": environment["gpu"].get("binding", "0"),
                },
            ),
        )
        _startup(read_json(startup))
        result["checks"].update(model="passed", application="passed")
        result["application_check"] = (
            "compiled-domain-runner-and-independent-model-startup"
        )
        if image:
            from .images import build_image

            result["image"] = build_image(
                project,
                result,
                root=root,
                repository=repository,
                image=image,
                push=push,
            )
        result["status"] = "complete"
        write_json(root / "build.json", result)
        if publish_result:
            build.publish_success(project, root, result, previous_lock=previous_lock)
        return {**result, "root": str(root)}
    except BaseException as error:
        result.update(status="failed", error=str(error))
        write_json(root / "build.json", result)
        raise
