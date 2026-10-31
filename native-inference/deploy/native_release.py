"""Verified model artifact handoff and native workflow release orchestration.

The controller runs without importing a model or a tensor framework. Compilation
is delegated to a finite Lepton job; Docker on the controller assembles CPU images.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import sys
import uuid


def canonical(value) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def content_id(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def compilation_key(resolved: dict, alias: str) -> str:
    return content_id(compilation_identity(resolved, alias))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def image_digest(value):
    require(
        isinstance(value, str) and re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", value),
        "an immutable registry image digest is required",
    )
    return value


def relative(value):
    require(
        isinstance(value, str) and value and "\\" not in value,
        "contained relative path required",
    )
    path = PurePosixPath(value)
    require(
        not path.is_absolute()
        and ".." not in path.parts
        and path.parts
        and ":" not in value,
        "path escapes its artifact root",
    )
    return path


def contained(root, value, *, file=True):
    root = Path(root)
    require(not root.is_symlink(), "artifact root is a symlink")
    path = root
    for part in relative(value).parts:
        path /= part
        require(not path.is_symlink(), f"artifact path contains a symlink: {value}")
    if file:
        require(path.is_file(), f"missing regular file: {value}")
    return path


def file_identity(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"regular file required: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def records(records):
    require(
        isinstance(records, list) and records, "a nonempty file inventory is required"
    )
    result = []
    names = set()
    for item in records:
        path = str(relative(item.get("path")))
        require(path not in names, "duplicate file inventory path")
        require(
            isinstance(item.get("sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]),
            "invalid file SHA256",
        )
        require(
            type(item.get("size_bytes")) is int and item["size_bytes"] >= 0,
            "invalid file size",
        )
        names.add(path)
        result.append(
            {"path": path, "sha256": item["sha256"], "size_bytes": item["size_bytes"]}
        )
    return sorted(result, key=lambda item: item["path"])


def inventory(root):
    root = Path(root)
    require(
        root.is_dir() and not root.is_symlink(), "regular artifact directory required"
    )
    result = []
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"artifact symlinks are unsupported: {path}")
        if path.is_dir():
            continue
        result.append(
            {"path": path.relative_to(root).as_posix(), **file_identity(path)}
        )
    return records(result)


def verify_inventory(root, expected, *, exact=False):
    expected = records(expected)
    for item in expected:
        require(
            file_identity(contained(root, item["path"]))
            == {key: item[key] for key in ("sha256", "size_bytes")},
            f"artifact bytes changed: {item['path']}",
        )
    if exact:
        require(inventory(root) == expected, "artifact file inventory changed")
    return expected


def read_json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, f"duplicate JSON key: {key}")
            value[key] = item
        return value

    value = json.loads(Path(path).read_text(), object_pairs_hook=pairs)
    require(isinstance(value, dict), f"JSON object required: {path}")
    canonical(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_bytes(canonical(value))
    os.replace(temporary, path)


def compilation_identity(resolved, alias):
    model = resolved["model_aliases"][alias]
    document = {
        key: value
        for key, value in model["project_document"].items()
        if key
        not in {
            "executor",
            "builder_image",
            "runtime",
            "output_root",
            "profiles",
            "default_profile",
        }
    }
    identity = {
        "schema_version": 1,
        "model": document,
        "files": records(model["project_files"]),
        "backend": resolved["backend"],
        "builder_image": image_digest(resolved["images"]["builder"]),
        "target": resolved["target"]["compilation"],
    }
    repository = resolved.get("paths", {}).get("repository_root")
    if repository:
        identity["default_toolchain_lock"] = file_identity(
            contained(
                Path(repository), "native-inference/model-builder/toolchain.lock.json"
            )
        )
        root = Path(repository) / "native-inference/model-builder/src"
        identity["builder_sources"] = [
            {**item, "path": "native-inference/model-builder/src/" + item["path"]}
            for item in inventory(root)
            if "__pycache__" not in PurePosixPath(item["path"]).parts
            and not item["path"].endswith(".pyc")
        ]
    if resolved.get("adapter") == "transolver_surface":
        identity["checkpoint"] = resolved["cfd"]["checkpoint"]
    return identity


def _builder():
    # The source checkout's framework-free verifier also ships in the installed
    # builder. Neither this import nor verification imports Torch or user code.
    try:
        from model_builder.build import (
            authoring,
            authoring_config,
            cli,
            completion,
            inputs,
        )
    except ModuleNotFoundError as error:
        if error.name != "model_builder":
            raise
        sys.path.insert(
            0, str(Path(__file__).resolve().parents[1] / "model-builder/src")
        )
        from model_builder.build import (
            authoring,
            authoring_config,
            cli,
            completion,
            inputs,
        )
    return authoring, authoring_config, cli, completion, inputs


def model_input_identity(resolved, alias):
    """The portable model portion of the Builder's captured authoring identity."""
    if resolved.get("adapter") == "transolver_surface":
        return None  # The joint full-field receipt binds the preset preparation.
    from types import SimpleNamespace

    authoring, configuration, _, _, inputs = _builder()
    model = resolved["model_aliases"][alias]
    verify_inventory(Path(model["project"]).parent, model["project_files"])
    project = configuration.load(
        Path(model["project"]), SimpleNamespace(backend=[resolved["backend"]])
    )
    source, selected, config = authoring._selected(project)
    effective = project["effective"]
    result = {
        "format_version": 2,
        "name": effective["name"],
        "version": effective["version"],
        "adapter": Path(effective["adapter"])
        .relative_to(Path(project["path"]).parent)
        .as_posix(),
        "source": {
            name: {key: item[key] for key in ("sha256", "size_bytes")}
            for name, item in source.items()
        },
        "inputs": {
            name: {key: item[key] for key in ("sha256", "size_bytes")}
            for name, item in selected.items()
        },
        "configuration_sha256": hashlib.sha256(inputs._canonical(config)).hexdigest(),
        "backends": [resolved["backend"]],
        "device": "cuda",
        "required_gpu_arch": resolved["target"]["compilation"]["cuda_architecture"],
        "input_names": effective.get("input_names"),
        "output_names": effective.get("output_names"),
    }
    for key in ("aoti_profile", "aoti_options", "tensorrt_profile"):
        if key in effective:
            result[key] = effective[key]
    return result


def verify_completed_build(output, *, backend, expected_identity=None, target=None):
    output = Path(output)
    inventory(output)  # Reject every symlink, including uninventoried additions.
    _, _, cli, completion, inputs = _builder()
    recipe_path = output / "source/effective-recipe.json"
    if not recipe_path.is_file():
        recipe_path = output / "source/recipe.json"
    recipe = cli.read_recipe(recipe_path)
    built = read_json(contained(output, "build.json"))
    plan = {
        "output": output,
        "backends": [backend],
        "device": built.get("device"),
        "recipe": recipe,
        "model_inputs": inputs.resolve_inputs(recipe, recipe_path),
    }
    completion.validate_container_completion(plan, read_recipe=cli.read_recipe)
    published = read_json(contained(output, "model/model-release.json"))
    require(
        published.get("environment") == built.get("environment"),
        "model release environment differs from build receipt",
    )
    from model_builder.build import worker
    from model_builder.build.tensors import tensor_contracts

    input_contract, output_contract = tensor_contracts(recipe)
    manifest = read_json(
        contained(
            output / "model", published["variants"][backend]["package"] + "/model.json"
        )
    )
    require(
        manifest["inputs"] == input_contract,
        "package input tensor contract differs from build recipe",
    )
    for declared, compiled in zip(output_contract, manifest["outputs"], strict=True):
        require(
            all(
                compiled.get(key) == value
                for key, value in declared.items()
                if value is not None
            ),
            "package output tensor contract differs from build recipe",
        )
    output_contract = manifest["outputs"]
    checks = read_json(
        completion._verify_completion_file(output, built["variants"][backend]["checks"])
    )
    for index, case in enumerate(checks["cases"]):
        directory = output / "checks" / backend / f"case-{index}"
        require(
            len(case.get("inputs", [])) == len(input_contract),
            "native parity input coverage differs",
        )
        for number, (contract, captured) in enumerate(
            zip(input_contract, case["inputs"], strict=True)
        ):
            data = contained(directory, f"input-{number}.bin").read_bytes()
            require(
                all(captured.get(key) == value for key, value in contract.items()),
                "native parity input tensor contract differs",
            )
            require(
                hashlib.sha256(data).hexdigest() == captured.get("sha256"),
                "native parity input bytes differ",
            )
            count = 1
            for dimension in contract["shape"]:
                count *= dimension
            require(
                len(data) == count * 4
                and all(
                    math.isfinite(value) for value in struct.unpack(f"<{count}f", data)
                ),
                "native parity input bytes or finite values differ",
            )
        for number, (contract, metadata, reported) in enumerate(
            zip(
                output_contract,
                case["metadata"]["outputs"],
                case["outputs"],
                strict=True,
            )
        ):
            expected = {
                **contract,
                "data": contained(directory, f"reference-{number}.bin").read_bytes(),
            }
            metrics = worker._compare_output(
                contained(directory, f"output-{number}.bin").read_bytes(),
                expected,
                metadata,
                require_byte_identical=checks.get("require_byte_identical", False),
            )
            require(
                metrics == reported,
                "native parity metrics differ from their actual output bytes",
            )
    if expected_identity is not None:
        check = read_json(contained(output, "check.json"))
        actual = check.get("input_identity", {})
        require(
            check.get("status") == "complete" and check.get("operation") == "build",
            "authoring build did not complete",
        )
        require(
            {key: actual.get(key) for key in expected_identity} == expected_identity,
            "compiled model inputs differ from selected source, weights or export settings",
        )
        require(
            check.get("environment") == built.get("environment")
            and check.get("tensor_contract")
            == {"inputs": recipe["inputs"], "outputs": recipe["outputs"]},
            "authoring summary differs from verified build",
        )
        proof = built.get("qualification", {})
        require(
            proof.get("passed") is True and proof == published.get("qualification"),
            "source integrity qualification is missing",
        )
        proof_file = completion._verify_completion_file(
            output / "model", proof["report"]
        )
        proof_json = read_json(proof_file)
        require(
            proof_json.get("passed") is True
            and proof_json.get("source") == expected_identity["source"],
            "source integrity proof differs from selected project",
        )
    if target is not None:
        environment = built.get("environment", {})
        require(
            environment.get("gpu_name") == target["gpu_product"]
            and environment.get("compute_capability") == target["compute_capability"],
            "actual build GPU differs from requested product or compute capability",
        )
    variant = published["variants"][backend]
    package = contained(output / "model", variant["package"], file=False)
    package_files = records(variant["files"])
    require(
        {item["path"] for item in package_files}
        == {
            str(PurePosixPath(variant["package"]) / item["path"])
            for item in inventory(package)
        },
        "compiled package contains extra or missing files",
    )
    for item in inventory(output / "model"):
        require(
            Path(item["path"]).suffix not in {".py", ".pyc", ".pt", ".pth", ".ckpt"},
            "deployable model contains build source or checkpoint",
        )
    source = None
    if (output / "release-provenance.json").is_file():
        provenance = read_json(contained(output, "release-provenance.json"))
        source = provenance.get("source_sha")
        require(
            isinstance(source, str) and re.fullmatch(r"[0-9a-f]{40}", source),
            "compiled alias has invalid source provenance",
        )
    return {
        "package": variant["package"],
        "model_files": inventory(output / "model"),
        "build": file_identity(output / "build.json"),
        "model_release": file_identity(output / "model/model-release.json"),
        "environment": built["environment"],
        "compilation_source_sha": source,
    }


def cache_lookup(cache_dir, key, *, backend, expected_identity=None, target=None):
    entry_root = Path(cache_dir) / key
    if not entry_root.exists() and not entry_root.is_symlink():
        return {"status": "miss", "key": key}
    try:
        entry = read_json(contained(entry_root, "entry.json"))
        require(
            entry.get("schema_version") == 1
            and entry.get("status") == "complete"
            and entry.get("compilation_key") == key,
            "cache entry identity is invalid",
        )
        require(
            content_id(entry["files"]) == entry.get("artifact_digest"),
            "cache artifact digest differs",
        )
        payload = entry_root / "candidate"
        verify_inventory(payload, entry["files"], exact=True)
        verified = verify_completed_build(
            payload, backend=backend, expected_identity=expected_identity, target=target
        )
        require(
            entry.get("verified") == verified, "cached verification metadata differs"
        )
        provenance = read_json(contained(payload, "release-provenance.json"))
        require(
            provenance.get("compilation_key") == key,
            "cached source proof has a different compilation identity",
        )
        return {
            "status": "hit",
            "key": key,
            "path": str(payload),
            "artifact_digest": entry["artifact_digest"],
            "entry": entry,
        }
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        return {"status": "invalid", "key": key, "reason": str(error)}


def cache_publish(
    cache_dir, key, candidate, *, backend, expected_identity=None, target=None
):
    import fcntl

    require(
        isinstance(key, str) and re.fullmatch(r"[0-9a-f]{64}", key),
        "invalid compilation key",
    )
    verified = verify_completed_build(
        candidate, backend=backend, expected_identity=expected_identity, target=target
    )
    provenance = read_json(contained(candidate, "release-provenance.json"))
    require(
        provenance.get("compilation_key") == key,
        "build provenance has a different compilation identity",
    )
    files = inventory(candidate)
    cache_dir = Path(cache_dir)
    require(not cache_dir.is_symlink(), "cache root cannot be a symlink")
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = cache_dir / (".publish-" + uuid.uuid4().hex)
    temporary.mkdir()
    try:
        shutil.copytree(candidate, temporary / "candidate")
        verify_inventory(temporary / "candidate", files, exact=True)
        require(
            verify_completed_build(
                temporary / "candidate",
                backend=backend,
                expected_identity=expected_identity,
                target=target,
            )
            == verified,
            "build changed while copying into cache",
        )
        write_json(
            temporary / "entry.json",
            {
                "schema_version": 1,
                "status": "complete",
                "compilation_key": key,
                "artifact_digest": content_id(files),
                "files": files,
                "verified": verified,
            },
        )
        locks = cache_dir / ".locks"
        locks.mkdir(exist_ok=True)
        require(not locks.is_symlink(), "cache lock directory is a symlink")
        lock_path = locks / key
        require(not lock_path.is_symlink(), "cache lock is a symlink")
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            existing = cache_lookup(
                cache_dir,
                key,
                backend=backend,
                expected_identity=expected_identity,
                target=target,
            )
            if existing["status"] == "hit":
                return existing  # First valid publisher owns the immutable key.
            destination = cache_dir / key
            if destination.exists() or destination.is_symlink():
                require(not destination.is_symlink(), "cache entry root is a symlink")
                os.rename(
                    destination,
                    cache_dir / (".invalid-" + key + "-" + uuid.uuid4().hex),
                )
            os.rename(temporary, destination)
        return cache_lookup(
            cache_dir,
            key,
            backend=backend,
            expected_identity=expected_identity,
            target=target,
        )
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


INSTALL_ROOT = PurePosixPath("/opt/physicsnemo-release")
MANAGEMENT_CAPABILITIES = ["live-native-readiness-v1", "fresh-deployment-v1"]
SOURCE_ROOTS = (
    "Cargo.toml",
    "Cargo.lock",
    ".cargo",
    "crates",
    "scripts/entrypoint.sh",
    "scripts/health_stub.py",
    "native-inference/deploy",
    "native-inference/workflows/runtime",
    "native-inference/workflows/transolver",
    "native-inference/cpp-runtime/include/physicsnemo/inference/v1/self_test.hpp",
    "native-inference/model-builder/src",
    "native-inference/model-builder/models",
    "native-inference/model-builder/toolchain.lock.json",
    "native-inference/examples",
    "qa/native_inference",
    "qa/scripts",
    "qa/inference/cfd_parity_surface_run1_transolver.json",
    "plugins/physicsnemo-cfd-surface-benchmark/examples/public_run_1_transolver_request.json",
)
EXCLUDED_DIRECTORIES = {
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    "target",
    "builds",
    "dist",
    ".venv",
    "venv",
}


def source_inventory(repository):
    repository = Path(repository)
    selected = []
    for name in SOURCE_ROOTS:
        path = contained(repository, name, file=False)
        require(path.exists(), f"release source is missing: {name}")
        paths = [path] if path.is_file() else path.rglob("*")
        for item in paths:
            if any(
                part in EXCLUDED_DIRECTORIES
                for part in item.relative_to(repository).parts
            ):
                continue
            require(not item.is_symlink(), f"release source contains a symlink: {item}")
            if item.is_file():
                selected.append(
                    {
                        "path": item.relative_to(repository).as_posix(),
                        **file_identity(item),
                    }
                )
    return records(selected)


def workflow_binary_key(resolved, sources):
    custom = resolved.get("native_workflow")
    custom = (
        {"cmake_target": custom["cmake_target"], "files": custom["files"]}
        if custom
        else None
    )
    return content_id(
        {
            "schema_version": 1,
            "builder_image": image_digest(resolved["images"]["builder"]),
            "server_image": image_digest(resolved["images"]["server"]),
            "source": sources,
            "custom": custom,
            "platform": resolved["target"]["compilation"]["platform"],
            **({"backend": "tensorrt"} if resolved["backend"] == "tensorrt" else {}),
        }
    )


def _verify_inputs(resolved):
    require(resolved.get("schema_version") == 1, "unsupported resolved release schema")
    require(
        resolved.get("adapter") in {"tensor.v1", "custom.v1", "transolver_surface"},
        "unsupported native workflow adapter",
    )
    require(
        resolved.get("backend") in {"aoti", "tensorrt"}, "unsupported release backend"
    )
    require(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", resolved.get("workflow", "")),
        "invalid workflow identifier",
    )
    for image in resolved["images"].values():
        image_digest(image)
    require(
        re.fullmatch(r"[0-9a-f]{40}", resolved.get("source_sha", "")),
        "full source revision required",
    )
    aliases = resolved["model_aliases"]
    require(
        isinstance(aliases, dict) and 0 < len(aliases) <= 16,
        "one to sixteen model aliases required",
    )
    for alias, model in aliases.items():
        require(
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", alias),
            "invalid fixed model alias",
        )
        project = Path(model["project"])
        require(project.is_absolute(), "resolved model project must be absolute")
        if resolved["adapter"] != "transolver_surface":
            verify_inventory(project.parent, model["project_files"])
    for asset in resolved.get("assets", []):
        relative(asset["destination"])
        require(
            PurePosixPath(asset["destination"]).parts[0] == "assets",
            "domain assets must be under assets/",
        )
        require(
            file_identity(asset["source"])
            == {key: asset[key] for key in ("sha256", "size_bytes")},
            "domain asset bytes changed",
        )
    custom = resolved.get("native_workflow")
    if resolved["adapter"] == "custom.v1" and not resolved.get("model_only"):
        require(
            isinstance(custom, dict)
            and re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_.-]*", custom.get("cmake_target", "")
            ),
            "custom workflow requires a valid CMake target",
        )
        verify_inventory(custom["source_dir"], custom["files"], exact=True)
    if not resolved.get("model_only"):
        policy = resolved["qualification"]
        required = policy.get("required_cases")
        require(
            isinstance(required, list)
            and required
            and len(set(required)) == len(required)
            and all(isinstance(name, str) and name for name in required),
            "qualification requires unique ordered named cases",
        )
        fixtures = policy.get("fixtures", [])
        if fixtures:
            verify_inventory(resolved["paths"]["workflow_root"], fixtures)
        require(
            isinstance(policy.get("policy"), str) and policy["policy"],
            "versioned qualification policy required",
        )
    target = resolved["target"]
    compilation = target["compilation"]
    require(
        compilation["platform"] == "linux/amd64" and compilation["precision"] == "fp32",
        "initial workflow releases require linux/amd64 FP32",
    )
    require(
        compilation["hardware_compatibility"] == "qualified-product-only"
        and compilation["mig"] == "disabled",
        "initial release requires exact GPU product with MIG disabled",
    )
    capability = compilation["compute_capability"]
    require(
        isinstance(capability, list)
        and len(capability) == 2
        and all(type(n) is int and n >= 0 for n in capability)
        and compilation["cuda_architecture"] == f"sm{capability[0]}{capability[1]}",
        "target architecture and compute capability disagree",
    )
    require(
        target["deployment"]["mig"] == "disabled"
        and type(target["deployment"]["minimum_available_memory_mb"]) is int
        and target["deployment"]["minimum_available_memory_mb"] > 0,
        "explicit deployment MIG and memory policy required",
    )
    require(
        re.fullmatch(
            r"[0-9]+(?:\.[0-9]+)+",
            target["deployment"]["driver_requirements"]["minimum_driver_version"],
        ),
        "qualified driver version bound required",
    )


def plan_release(resolved, cache_dir):
    """Return byte-verified hit/miss decisions without running Docker or a job."""
    require(
        not (
            resolved.get("model_only")
            and resolved.get("adapter") == "transolver_surface"
        ),
        "the CFD preset requires its complete workflow check; select a standalone model project",
    )
    if resolved.get("adapter") == "transolver_surface":
        resolved = resolve_cfd_profile(resolved)
    _verify_inputs(resolved)
    sources = source_inventory(resolved["paths"]["repository_root"])
    models = {}
    for alias in sorted(resolved["model_aliases"]):
        key = compilation_key(resolved, alias)
        identity = model_input_identity(resolved, alias)
        cached = cache_lookup(
            cache_dir,
            key,
            backend=resolved["backend"],
            expected_identity=identity,
            target=resolved["target"]["compilation"],
        )
        models[alias] = {
            "compilation_key": key,
            "action": "reuse" if cached["status"] == "hit" else "compile",
            "cache_status": cached["status"],
            "cache_reason": cached.get("reason"),
            "artifact_digest": cached.get("artifact_digest"),
            "candidate": cached.get("path"),
            "input_identity": identity,
        }
    result = {
        "schema_version": 1,
        "status": "planned",
        "workflow": resolved["workflow"],
        "source_sha": resolved["source_sha"],
        "models": models,
        "source_files": sources,
        "final_image_qualification_required": True,
        "gpu_compilation_executor": "lepton",
    }
    if resolved.get("model_only"):
        result.update(
            scope="model",
            model_verified=False,
            workflow_incomplete=True,
            qualified=False,
        )
        return result
    return {
        **result,
        "workflow_binary_key": workflow_binary_key(resolved, sources),
        "qualification_policy_key": content_id(
            {"target": resolved["target"], "qualification": resolved["qualification"]}
        ),
    }


def check_models(resolved, *, cache_dir, workspace=None, process_runner=None):
    """Verify/build selected models without assembling or qualifying a workflow."""
    require(resolved.get("model_only") is True, "model-only resolution is required")
    require(
        resolved.get("adapter") != "transolver_surface",
        "the CFD preset requires its complete workflow check; select a standalone model project",
    )
    plan = plan_release(resolved, cache_dir)
    missing = {
        alias: value
        for alias, value in plan["models"].items()
        if value["action"] == "compile"
    }
    if missing:
        require(
            resolved.get("output", {}).get("repository"),
            "output.repository is required to publish model compiler inputs on a cache miss",
        )
    workspace = Path(workspace or resolved["paths"]["workspace"])
    require(
        not workspace.exists() and not workspace.is_symlink(),
        "model-check workspace already exists; choose a fresh directory",
    )
    workspace.mkdir(parents=True)
    report = {
        "schema_version": 1,
        "kind": "native-model-check",
        "scope": "model",
        "status": "running",
        "model_verified": False,
        "workflow_incomplete": True,
        "qualified": False,
        "workflow": resolved["workflow"],
        "backend": resolved["backend"],
        "target": resolved["target"],
        "source_sha": resolved["source_sha"],
        "plan": plan,
    }
    write_json(workspace / "resolved.json", resolved)
    try:
        candidates = {
            alias: Path(value["candidate"])
            for alias, value in plan["models"].items()
            if value["action"] == "reuse"
        }
        if missing:
            import native_release_lepton as lepton

            controller = lepton.validate_controller_config(resolved)
            resolved = {**resolved, "lepton": controller}
            runner = process_runner or ProcessRunner(workspace / "commands")
            stack = verify_stack_images(resolved, runner, workspace)
            source, custom, inputs, empty = (
                workspace / name
                for name in ("source", "custom", "compile-input", "empty")
            )
            _copy_inventory(
                resolved["paths"]["repository_root"], plan["source_files"], source
            )
            for directory in (custom, inputs, empty):
                directory.mkdir()
            candidates, compilation, _ = _compile_handoff(
                resolved,
                plan,
                candidates,
                missing,
                source=source,
                custom=custom,
                inputs=inputs,
                empty=empty,
                workspace=workspace,
                artifact_workspace=workspace,
                runner=runner,
                lepton=lepton,
                cache_dir=cache_dir,
                stack=stack,
                process_runner=process_runner,
            )
            report["compilation"] = compilation
        models = {}
        for alias, candidate in candidates.items():
            decision = plan["models"][alias]
            proof = verify_completed_build(
                candidate,
                backend=resolved["backend"],
                expected_identity=decision["input_identity"],
                target=resolved["target"]["compilation"],
            )
            provenance = read_json(contained(candidate, "release-provenance.json"))
            require(
                provenance.get("compilation_key") == decision["compilation_key"],
                "verified model compilation key differs",
            )
            models[alias] = {
                "candidate": str(candidate.resolve()),
                "compilation_key": decision["compilation_key"],
                "artifact_digest": content_id(inventory(candidate)),
                "verified": proof,
                "recompiled": alias in missing,
            }
        require(
            set(models) == set(resolved["model_aliases"]),
            "model check alias coverage differs",
        )
        report.update(status="model_verified", model_verified=True, models=models)
        return report
    except BaseException as error:
        report.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        write_json(workspace / "model-check.json", report)


def _validate_self_test(manifest, fixture):
    require(
        isinstance(fixture, dict),
        "each model alias requires an authored startup self-test",
    )
    for direction, key in (("inputs", "inputs"), ("outputs", "expected_outputs")):
        tensors = manifest[direction]
        supplied = fixture.get(key)
        require(
            isinstance(supplied, dict)
            and set(supplied) == {tensor["name"] for tensor in tensors},
            "startup fixture tensor names differ from compiled package",
        )
        for tensor in tensors:
            shape = tensor["shape"]
            require(
                tensor["dtype"] == "float32"
                and all(type(n) is int and n > 0 for n in shape),
                "startup fixture requires static FP32 tensors",
            )
            values = supplied[tensor["name"]]
            require(
                isinstance(values, list) and len(values) == math.prod(shape),
                "startup fixture element count differs from package",
            )
            require(
                all(
                    type(n) in (int, float)
                    and math.isfinite(n)
                    and abs(n) <= 3.4028234663852886e38
                    for n in values
                ),
                "startup fixture must contain finite FP32 values",
            )
    for key in ("rtol", "atol"):
        value = fixture.get(key, 0.0)
        require(
            type(value) in (int, float) and math.isfinite(value) and value >= 0,
            "startup fixture tolerances must be finite and nonnegative",
        )


def stage_release(resolved, candidates, output, *, sources=None):
    """Stage complete deployable bundles; build sources and weights stay outside."""
    if resolved.get("adapter") == "transolver_surface":
        return stage_cfd_release(
            resolve_cfd_profile(resolved), candidates, output, sources=sources
        )
    _verify_inputs(resolved)
    require(
        set(candidates) == set(resolved["model_aliases"]),
        "all fixed model aliases must be present before staging",
    )
    output = Path(output)
    require(
        not output.exists() and not output.is_symlink(), "release output already exists"
    )
    verified = {}
    models = {}
    for alias, path in candidates.items():
        proof = verify_completed_build(
            path,
            backend=resolved["backend"],
            expected_identity=model_input_identity(resolved, alias),
            target=resolved["target"]["compilation"],
        )
        manifest = read_json(
            contained(Path(path) / "model", proof["package"] + "/model.json")
        )
        fixture = resolved.get("self_tests", {}).get(alias)
        _validate_self_test(manifest, fixture)
        verified[alias] = proof
        models[alias] = {
            "package_path": str(INSTALL_ROOT / "model" / alias / proof["package"]),
            "backend": resolved["backend"],
            "device": "cuda",
            "device_index": 0,
            "self_test": fixture,
        }
    require(
        set(resolved.get("self_tests", {})) == set(models),
        "startup fixture alias coverage differs",
    )
    assets = {}
    for asset in resolved.get("assets", []):
        name = asset.get("name", Path(asset["destination"]).stem)
        require(name not in assets, "domain asset names must be unique")
        assets[name] = str(INSTALL_ROOT / asset["destination"])
    managed_contract = "manifest_version" in resolved
    workflow = {
        "schema_version": 2 if managed_contract else 1,
        "adapter": resolved["adapter"],
        "models": models,
        "assets": assets,
    }
    if resolved["adapter"] == "tensor.v1":
        workflow["tensor_adapter"] = resolved["adapter_config"]
    elif resolved["adapter"] == "custom.v1":
        workflow["custom"] = resolved.get("adapter_config", {})
    runtime = resolved.get("runtime", {})
    deployment = {
        "schema_version": 3 if managed_contract else 2,
        "workflow_id": resolved["workflow"],
        "executor_class": runtime.get("executor_class", "cpp.gpu.sdk"),
        "device_index": 0,
        "target": resolved["target"],
        "workflow": workflow,
        "artifact_root": runtime.get("artifact_root", "/outputs/native-workflow"),
        "input_cache": runtime.get("input_cache", "/outputs/.cache/native-workflow"),
        "request_timeout_seconds": runtime.get("request_timeout_seconds", 300),
        "drain_timeout_seconds": runtime.get("drain_timeout_seconds", 60),
        "startup_timeout_seconds": runtime.get("startup_timeout_seconds", 300),
        "request_schema": resolved["request_schema"],
        "response_schema": resolved.get("response_schema", True),
        "input_artifacts": {},
    }
    if managed_contract:
        require(
            re.fullmatch(r"[0-9a-f]{64}", resolved.get("public_contract_digest", "")),
            "public contract digest is required",
        )
        deployment.update(
            manifest_version=resolved["manifest_version"],
            public_contract_digest=resolved["public_contract_digest"],
            ingress_root=deployment["artifact_root"],
            input_files=resolved.get("input_files", {}),
            startup_cases=[
                {key: value for key, value in case.items() if key != "files"}
                for case in resolved.get("startup_cases", [])
            ],
        )
    require(
        all(
            isinstance(proof.get("compilation_source_sha"), str)
            for proof in verified.values()
        ),
        "every model alias requires its verified compilation source",
    )
    deployment["model_build_sources"] = {
        alias: proof["compilation_source_sha"] for alias, proof in verified.items()
    }
    try:
        output.mkdir(parents=True, exist_ok=False)
        for alias, candidate in candidates.items():
            shutil.copytree(Path(candidate) / "model", output / "model" / alias)
            verify_inventory(
                output / "model" / alias, verified[alias]["model_files"], exact=True
            )
        for asset in resolved.get("assets", []):
            destination = output / asset["destination"]
            require(not destination.exists(), "duplicate domain asset destination")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(asset["source"], destination)
            require(
                file_identity(destination)
                == {key: asset[key] for key in ("sha256", "size_bytes")},
                "domain asset changed while staging",
            )
        for name, descriptor in resolved.get("input_artifacts", {}).items():
            require(
                re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", name),
                "invalid immutable input alias",
            )
            if "url" in descriptor:
                require(
                    descriptor["url"].startswith("https://"),
                    "immutable input URL requires HTTPS",
                )
                deployment["input_artifacts"][name] = descriptor
            else:
                selected = Path(descriptor.get("source", descriptor.get("path", "")))
                destination = output / "inputs" / name
                destination.parent.mkdir(exist_ok=True)
                shutil.copyfile(selected, destination)
                identity = file_identity(destination)
                require(
                    identity
                    == {key: descriptor[key] for key in ("sha256", "size_bytes")},
                    "immutable input bytes changed",
                )
                deployment["input_artifacts"][name] = {
                    "path": str(INSTALL_ROOT / "inputs" / name),
                    **identity,
                }
        for case in deployment.get("startup_cases", []):
            require(
                isinstance(case.get("name"), str)
                and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", case["name"]),
                "invalid startup case name",
            )
            files = {}
            for name, descriptor in case.get("input_artifacts", {}).items():
                require(
                    re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", name),
                    "invalid startup file name",
                )
                source = Path(descriptor["path"])
                destination = output / "startup" / case["name"] / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                identity = file_identity(destination)
                require(
                    identity
                    == {key: descriptor[key] for key in ("sha256", "size_bytes")},
                    "startup fixture file bytes changed",
                )
                files[name] = {
                    **descriptor,
                    "path": str(INSTALL_ROOT / "startup" / case["name"] / name),
                }
            if files:
                case["input_artifacts"] = files
        assets_inventory = []
        for item in inventory(output):
            assets_inventory.append({**item, "path": str(INSTALL_ROOT / item["path"])})
        deployment["model_assets"] = assets_inventory
        write_json(output / "deployment.json", deployment)
        plugin = resolved["plugin"]["document"]
        write_json(output / "plugin" / resolved["workflow"] / "plugin.yaml", plugin)
        shutil.copyfile(
            Path(__file__).parent / "worker_runtime_config.json",
            output / "worker_runtime_config.json",
        )
        fixtures = resolved["qualification"].get("fixtures", [])
        for item in fixtures:
            src = contained(resolved["paths"]["workflow_root"], item["path"])
            dest = output / "qualification" / item["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            require(
                file_identity(dest)
                == {key: item[key] for key in ("sha256", "size_bytes")},
                "qualification fixture changed while staging",
            )
        manifest = {
            "schema_version": 1,
            "workflow": resolved["workflow"],
            "source_sha": resolved["source_sha"],
            "images": resolved["images"],
            "target": resolved["target"],
            "adapter": resolved["adapter"],
            "backend": resolved["backend"],
            "models": {
                alias: {
                    "compilation_key": compilation_key(resolved, alias),
                    **verified[alias],
                }
                for alias in models
            },
            "workflow_binary_key": workflow_binary_key(
                resolved,
                sources or source_inventory(resolved["paths"]["repository_root"]),
            ),
            "model_build_sources": deployment["model_build_sources"],
            "management_capabilities": MANAGEMENT_CAPABILITIES,
            "qualification": resolved["qualification"],
            "files": inventory(output),
        }
        manifest["bundle_digest"] = content_id(manifest["files"])
        write_json(output / "release-inventory.json", manifest)
        verify_bundle(output)
        return manifest
    except BaseException:
        shutil.rmtree(output)
        raise


def verify_bundle(output):
    output = Path(output)
    record = read_json(contained(output, "release-inventory.json"))
    expected = records(record["files"])
    require(
        content_id(expected) == record.get("bundle_digest"),
        "release bundle identity differs",
    )
    verify_inventory(output, expected)
    require(
        {item["path"] for item in inventory(output)}
        == {item["path"] for item in expected} | {"release-inventory.json"},
        "release bundle has unrecorded files",
    )
    return record


def resolve_cfd_profile(resolved):
    """Expand the fixed full/tail preset without local Torch or model downloads."""
    import copy

    result = copy.deepcopy(resolved)
    if "cfd" in result and result.get("model_aliases"):
        return result
    require(
        result.get("provider_profile") == "physicsnemo-cfd-surface-transolver-run1-v1",
        "unsupported CFD provider profile",
    )
    definitions = result.get("cfd_models", {})
    require(
        isinstance(definitions, dict)
        and {item.get("points") for item in definitions.values()} == {60000, 8095}
        and len(definitions) == 2
        and all(
            item.get("preset") == "transolver-surface" for item in definitions.values()
        ),
        "CFD preset requires full60000 and exact8095 tail",
    )
    repository = Path(result["paths"]["repository_root"])
    template = repository / "native-inference/examples/transolver-surface"
    document = read_json(template / "model-build.json")
    backend = result["backend"]
    require(backend in ("aoti", "tensorrt"), "backend must be aoti or tensorrt")
    document["backends"] = [backend]
    if backend == "aoti":
        document["assets"] = {}
        document.pop("tensorrt_profile", None)
    else:
        document["tensorrt_profile"] = "provider-default-exact-v1"
    aliases = {}
    for alias, definition in definitions.items():
        adapter, count = re.subn(
            r"(?m)^POINT_COUNT = 75$",
            "POINT_COUNT = " + str(definition["points"]),
            (template / "adapter.py").read_text(),
        )
        require(
            count == 1, "Transolver preset source changed its point-count declaration"
        )
        aliases[alias] = {
            "project": str(template / "model-build.json"),
            "project_document": {**document, "points": definition["points"]},
            "project_files": [
                {
                    "path": "adapter.py",
                    "sha256": hashlib.sha256(adapter.encode()).hexdigest(),
                    "size_bytes": len(adapter.encode()),
                }
            ],
            "preset": definition,
        }
    profile = repository / "qa/inference/cfd_parity_surface_run1_transolver.json"
    request = (
        repository
        / "plugins/physicsnemo-cfd-surface-benchmark/examples/public_run_1_transolver_request.json"
    )
    result.update(
        model_aliases=aliases,
        cfd={
            "profile_path": str(profile),
            "request_path": str(request),
            "profile": file_identity(profile),
            "request": file_identity(request),
            "checkpoint": {
                "sha256": "eb98f399a050a8f8a24919335c61642e4a835bd4044f7e21abec231aa31fd82c",
                "size_bytes": 39188235,
            },
            "statistics": {
                "sha256": "c4b19eaca6158219570232ce094e6ef48b8705eb727b335441a310c7479b6b25",
                "size_bytes": 1375,
            },
        },
    )
    result.setdefault("self_tests", {})
    result["qualification"] = {
        "policy": "native-cfd-full-service-v1",
        "required_cases": ["run_1"],
        "fixtures": [],
        "provider_profile": result["provider_profile"],
    }
    return result


def reference_storage_identity(controller):
    """Bind oracle placement to the effective, validated controller location."""
    return {
        name: controller[name]
        for name in (
            "workspace_id",
            "workspace_url",
            "node_group",
            "nfs_mount_base",
            "lustre_dir",
            "lustre_storage",
            "mount_target",
        )
    }


def cfd_reference_contract(resolved):
    return {
        "provider_profile": resolved["provider_profile"],
        **{
            name: resolved["cfd"][name]
            for name in ("profile", "request", "checkpoint", "statistics")
        },
    }


def _cfd_package_info(resolved, candidates):
    packages, identities = [], []
    for candidate in candidates.values():
        verified = verify_completed_build(
            candidate,
            backend=resolved["backend"],
            target=resolved["target"]["compilation"],
        )
        package = Path(candidate) / "model" / verified["package"]
        packages.append(package)
        identities.append(
            {
                "manifest_sha256": file_identity(package / "model.json")["sha256"],
                "files": inventory(package),
            }
        )
    return packages, sorted(identities, key=lambda item: item["manifest_sha256"])


def _cfd_joint_identity(resolved, candidates):
    _, packages = _cfd_package_info(resolved, candidates)
    return {
        "schema_version": 1,
        "qualification": "full-run1-fields-and-metrics-v1",
        "compilation_keys": sorted(
            compilation_key(resolved, alias) for alias in resolved["model_aliases"]
        ),
        "packages": packages,
        "reference_contract": cfd_reference_contract(resolved),
        "model_build_sources": {
            compilation_key(resolved, alias): read_json(
                Path(candidate) / "release-provenance.json"
            )["source_sha"]
            for alias, candidate in candidates.items()
        },
        "builder_stack_image": resolved["images"]["builder"],
        "native_stack_fingerprint": resolved["_native_stack_fingerprint"],
        "reference_storage": resolved.get("_reference_storage", {}),
        **({"backend": "tensorrt"} if resolved["backend"] == "tensorrt" else {}),
    }


def _validate_cfd_joint(proof, identity):
    proof = Path(proof)
    inventory(proof)
    receipt = read_json(contained(proof, "build-receipt.json"))
    require(
        receipt.get("backend", "aoti") == identity.get("backend", "aoti"),
        "CFD joint proof backend differs",
    )
    require(
        receipt.get("status") == "passed"
        and receipt.get("qualification") == identity["qualification"]
        and receipt.get("summary_sha256")
        == file_identity(contained(proof, "summary.json"))["sha256"],
        "CFD joint qualification summary is missing or changed",
    )
    require(
        sorted(receipt.get("packages", []), key=lambda item: item["manifest_sha256"])
        == identity["packages"],
        "CFD joint proof does not cover selected package bytes",
    )
    keys = receipt.get("compilation_keys")
    require(
        isinstance(keys, dict)
        and sorted(keys.values()) == identity["compilation_keys"],
        "CFD joint proof model compilation identities differ",
    )
    sources = receipt.get("model_build_sources") or {
        alias: receipt.get("source_sha") for alias in keys
    }
    require(
        set(sources) == set(keys)
        and {keys[alias]: value for alias, value in sources.items()}
        == identity["model_build_sources"],
        "CFD joint proof actual compilation sources differ",
    )
    for name in (
        "reference_contract",
        "builder_stack_image",
        "native_stack_fingerprint",
        "reference_storage",
    ):
        require(
            receipt.get(name, {} if name == "reference_storage" else None)
            == identity[name],
            "CFD joint proof " + name + " differs",
        )
    stats_identity = file_identity(contained(proof, "global_stats.json"))
    require(
        stats_identity == identity["reference_contract"]["statistics"]
        and receipt.get("stats_sha256") == stats_identity["sha256"],
        "CFD joint statistics differ from current reference contract",
    )
    startup = receipt.get("startup_fixture", {})
    require(
        startup.get("path") == "startup/startup.json",
        "CFD needs qualified startup fixtures",
    )
    for item in records(startup.get("files")):
        require(
            PurePosixPath(item["path"]).parts[0] == "startup",
            "startup proof path escapes",
        )
        verify_inventory(proof, [item])
    reference = receipt.get("reference", {})
    require(
        isinstance(reference.get("path"), str)
        and Path(reference["path"]).is_absolute()
        and re.fullmatch(r"[0-9a-f]{64}", reference.get("sha256", ""))
        and type(reference.get("size_bytes")) is int
        and reference["size_bytes"] > 0,
        "CFD reference location or byte identity is invalid",
    )
    return receipt


def cfd_joint_lookup(cache_dir, resolved, candidates):
    identity = _cfd_joint_identity(resolved, candidates)
    key = content_id(identity)
    root = Path(cache_dir) / ".cfd-qualified" / key
    if not root.exists() and not root.is_symlink():
        return {"status": "miss", "key": key, "identity": identity}
    try:
        entry = read_json(contained(root, "entry.json"))
        require(
            entry.get("status") == "passed" and entry.get("identity") == identity,
            "CFD joint cache identity differs",
        )
        proof = root / "proof"
        verify_inventory(proof, entry["files"], exact=True)
        require(
            content_id(entry["files"]) == entry["artifact_digest"],
            "CFD joint proof inventory changed",
        )
        _validate_cfd_joint(proof, identity)
        require(
            isinstance(entry.get("node_id"), str) and entry["node_id"],
            "CFD reference storage node missing",
        )
        return {
            "status": "hit",
            "key": key,
            "path": str(proof),
            "node_id": entry["node_id"],
            "identity": identity,
        }
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        return {
            "status": "invalid",
            "key": key,
            "identity": identity,
            "reason": str(error),
        }


def cfd_joint_publish(cache_dir, resolved, candidates, proof, *, node_id):
    import fcntl

    identity = _cfd_joint_identity(resolved, candidates)
    _validate_cfd_joint(proof, identity)
    require(isinstance(node_id, str) and node_id, "CFD reference storage node missing")
    key = content_id(identity)
    root = Path(cache_dir) / ".cfd-qualified"
    require(
        not Path(cache_dir).is_symlink() and not root.is_symlink(),
        "CFD cache root is a symlink",
    )
    root.mkdir(parents=True, exist_ok=True)
    locks = root / ".locks"
    locks.mkdir(exist_ok=True)
    require(
        not locks.is_symlink() and not (locks / key).is_symlink(),
        "CFD cache lock is a symlink",
    )
    temporary = root / (".publish-" + uuid.uuid4().hex)
    temporary.mkdir()
    try:
        shutil.copytree(proof, temporary / "proof")
        _validate_cfd_joint(temporary / "proof", identity)
        files = inventory(temporary / "proof")
        write_json(
            temporary / "entry.json",
            {
                "schema_version": 1,
                "status": "passed",
                "identity": identity,
                "files": files,
                "artifact_digest": content_id(files),
                "node_id": node_id,
            },
        )
        with (locks / key).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            existing = cfd_joint_lookup(cache_dir, resolved, candidates)
            if existing["status"] == "hit":
                return existing
            destination = root / key
            if destination.exists() or destination.is_symlink():
                os.replace(
                    destination, root / (".invalid-" + key + "-" + uuid.uuid4().hex)
                )
            os.replace(temporary, destination)
        return cfd_joint_lookup(cache_dir, resolved, candidates)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def configure_cfd_runtime(config, resolved):
    """Keep v1 output/time bounds aligned with the selected image environment."""
    result = dict(config)
    runtime = resolved.get("runtime", {})
    for name in (
        "artifact_root",
        "input_cache",
        "request_timeout_seconds",
        "drain_timeout_seconds",
    ):
        if name in runtime:
            result[name] = runtime[name]
    return result


def stage_cfd_release(resolved, candidates, output, *, sources=None, joint_proof=None):
    """Bridge a joint full-CFD receipt into the same automatic image pipeline."""
    from prepare_release import prepare_release

    _verify_inputs(resolved)
    require(
        set(candidates) == set(resolved["model_aliases"]),
        "all CFD full/tail aliases are required",
    )
    packages, _ = _cfd_package_info(resolved, candidates)
    identity = _cfd_joint_identity(resolved, candidates)
    selected = None
    node_id = None
    if joint_proof is not None:
        require(joint_proof.get("status") == "hit", "verified CFD joint proof required")
        selected = Path(joint_proof["path"])
        node_id = joint_proof["node_id"]
        _validate_cfd_joint(selected, identity)
    else:
        for candidate in candidates.values():
            proof = Path(candidate) / "cfd-release"
            if not proof.is_dir():
                continue
            try:
                _validate_cfd_joint(proof, identity)
            except (OSError, ValueError, KeyError, TypeError):
                continue
            selected = proof
            node_id = read_json(Path(candidate) / "release-provenance.json")[
                "placement"
            ]["node_id"]
            break
    require(
        selected is not None,
        "no joint full-field qualification covers these exact CFD packages and reference contract",
    )
    receipt = read_json(selected / "build-receipt.json")
    inventory_record = prepare_release(
        backend=resolved["backend"],
        builder_image=receipt["builder_image"],
        packages=packages,
        stats=selected / "global_stats.json",
        case_config=Path(resolved["cfd"]["request_path"]),
        compile_evidence=selected / "build-receipt.json",
        source_sha=resolved["source_sha"],
        output=Path(output),
    )
    output = Path(output)
    try:
        startup = receipt.get("startup_fixture")
        require(
            isinstance(startup, dict) and startup.get("path") == "startup/startup.json",
            "CFD needs qualified Python-reference startup fixtures",
        )
        for item in records(startup["files"]):
            require(
                PurePosixPath(item["path"]).parts[0] == "startup",
                "startup fixture path escapes its prefix",
            )
            src = contained(selected, item["path"])
            require(
                file_identity(src)
                == {key: item[key] for key in ("sha256", "size_bytes")},
                "trusted CFD startup fixture changed",
            )
            dest = output / item["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
        config = configure_cfd_runtime(read_json(output / "deployment.json"), resolved)
        config.update(
            backend=resolved["backend"],
            workflow_id=resolved["workflow"],
            executor_class=resolved.get("runtime", {}).get(
                "executor_class", "native.transolver.aoti"
            ),
            startup_fixture_path="/opt/physicsnemo-native/startup/startup.json",
            target=resolved["target"],
            model_build_sources={
                alias: read_json(Path(candidate) / "release-provenance.json")[
                    "source_sha"
                ]
                for alias, candidate in candidates.items()
            },
        )
        for item in startup["files"]:
            config["model_assets"].append(
                {**item, "path": "/opt/physicsnemo-native/" + item["path"]}
            )
        write_json(output / "deployment.json", config)
        write_json(
            output / "plugin" / resolved["workflow"] / "plugin.yaml",
            resolved["plugin"]["document"],
        )
        write_json(
            output / "orchestration.json",
            {
                "cfd": resolved["cfd"],
                "reference": receipt["reference"],
                "input_root": receipt["input_root"],
                "node_id": node_id,
            },
        )
        files = [
            item
            for item in inventory(output)
            if item["path"] != "release-inventory.json"
        ]
        manifest = {
            **inventory_record,
            "workflow": resolved["workflow"],
            "source_sha": resolved["source_sha"],
            "images": resolved["images"],
            "target": resolved["target"],
            "adapter": "transolver_surface",
            "backend": resolved["backend"],
            "models": {
                alias: {
                    "compilation_key": compilation_key(resolved, alias),
                    "compilation_source_sha": read_json(
                        Path(candidate) / "release-provenance.json"
                    )["source_sha"],
                }
                for alias, candidate in candidates.items()
            },
            "workflow_binary_key": workflow_binary_key(
                resolved,
                sources or source_inventory(resolved["paths"]["repository_root"]),
            ),
            "model_build_sources": config["model_build_sources"],
            "management_capabilities": MANAGEMENT_CAPABILITIES,
            "qualification": resolved["qualification"],
            "files": files,
            "bundle_digest": content_id(files),
        }
        write_json(output / "release-inventory.json", manifest)
        verify_bundle(output)
        return manifest
    except BaseException:
        shutil.rmtree(output)
        raise


class ProcessRunner:
    """Record bounded commands; secret-bearing calls deliberately omit details."""

    def __init__(self, evidence):
        self.evidence = Path(evidence)
        self.evidence.mkdir(parents=True, exist_ok=True)
        self.count = 0

    def run(
        self, argv, *, cwd=None, env=None, timeout=7200, sensitive=False, check=True
    ):
        import subprocess

        self.count += 1
        name = f"{self.count:04d}"
        command = [str(value) for value in argv]
        record = {"sensitive": sensitive, "timeout_seconds": timeout}
        if not sensitive:
            record.update(argv=command, cwd=str(cwd) if cwd else None)
        try:
            try:
                result = subprocess.run(
                    command,
                    cwd=cwd,
                    env=env,
                    timeout=timeout,
                    text=True,
                    capture_output=True,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                if sensitive:
                    raise RuntimeError("sensitive release command timed out") from None
                raise
            except OSError:
                if sensitive:
                    raise RuntimeError(
                        "sensitive release command failed to start"
                    ) from None
                raise
            record["exit_code"] = result.returncode
            if not sensitive:
                (self.evidence / (name + ".stdout.log")).write_text(result.stdout)
                (self.evidence / (name + ".stderr.log")).write_text(result.stderr)
            require(
                not check or result.returncode == 0,
                "release command failed"
                if sensitive
                else f"release command failed ({result.returncode}): {command[:4]}; evidence {self.evidence / name}",
            )
            return result
        finally:
            write_json(self.evidence / (name + ".command.json"), record)


def _copy_inventory(source, selected, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for item in records(selected):
        destination = output / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(contained(source, item["path"]), destination)
    verify_inventory(output, selected, exact=True)


def _metadata_digest(path, repository):
    metadata = read_json(path)
    digest = metadata.get("containerimage.digest")
    require(
        isinstance(digest, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", digest),
        "BuildKit did not return an immutable image digest",
    )
    return image_digest(repository + "@" + digest)


def build_image(resolved, *, source, custom, inputs, bundle, workspace, runner, target):
    workspace = Path(workspace)
    repository = resolved["output"]["repository"]
    require(
        isinstance(repository, str)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]*", repository)
        and "@" not in repository
        and not repository.endswith(":latest"),
        "release output requires a registry repository",
    )
    tag = (
        repository
        + ":"
        + ("build-input" if target == "compiler" else "candidate")
        + "-"
        + resolved["source_sha"][:8]
        + "-"
        + uuid.uuid4().hex[:10]
    )
    metadata = workspace / (target + "-image.json")
    custom_config = resolved.get("native_workflow")
    model_build_source = ""
    if target == "release":
        deployment = read_json(Path(bundle) / "deployment.json")
        model_sources = deployment.get("model_build_sources")
        if model_sources:
            require(
                isinstance(model_sources, dict)
                and all(
                    isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value)
                    for value in model_sources.values()
                ),
                "image requires actual alias compilation source identities",
            )
            actual = set(model_sources.values())
            model_build_source = next(iter(actual)) if len(actual) == 1 else ""
        elif resolved.get("adapter") == "transolver_surface":
            model_build_source = read_json(Path(bundle) / "build-receipt.json")[
                "source_sha"
            ]
        else:
            raise ValueError("generic image requires actual alias compilation sources")
    command = [
        "docker",
        "buildx",
        "build",
        "--platform",
        "linux/amd64",
        "--provenance=false",
        "--file",
        str(Path(source) / "native-inference/deploy/Dockerfile.workflow"),
        "--target",
        target,
        "--build-arg",
        "NATIVE_BUILDER_IMAGE=" + resolved["images"]["builder"],
        "--build-arg",
        "NATIVE_SERVER_IMAGE=" + resolved["images"]["server"],
        "--build-arg",
        "SOURCE_SHA=" + resolved["source_sha"],
        "--build-arg",
        "WORKFLOW_ID=" + resolved["workflow"],
        "--build-arg",
        "RELEASE_ROOT="
        + (
            "/opt/physicsnemo-native"
            if resolved.get("adapter") == "transolver_surface"
            else "/opt/physicsnemo-release"
        ),
        "--build-arg",
        "ARTIFACT_ROOT="
        + resolved.get("runtime", {}).get("artifact_root", "/outputs/native-workflow"),
        "--build-arg",
        "MODEL_BUILD_SOURCE_SHA=" + model_build_source,
        "--build-arg",
        "ENABLE_AOTI=" + ("ON" if resolved["backend"] == "aoti" else "OFF"),
        "--build-arg",
        "ENABLE_TENSORRT=" + ("ON" if resolved["backend"] == "tensorrt" else "OFF"),
        "--build-arg",
        "CUSTOM_WORKFLOW_SOURCE_DIR=" + ("/workflow-custom" if custom_config else ""),
        "--build-arg",
        "CUSTOM_WORKFLOW_TARGET="
        + (custom_config["cmake_target"] if custom_config else ""),
        "--build-context",
        "release-source=" + str(source),
        "--build-context",
        "workflow-project=" + str(custom),
        "--build-context",
        "compile-input=" + str(inputs),
        "--build-context",
        "compiled-release=" + str(bundle),
        "--metadata-file",
        str(metadata),
        "--tag",
        tag,
        "--push",
        str(source),
    ]
    # No --gpus or Docker GPU entitlement: these compile ordinary CPU binaries.
    runner.run(command, timeout=7200, env={**os.environ, "DOCKER_BUILDKIT": "1"})
    result = {
        "tag": tag,
        "image": _metadata_digest(metadata, repository),
        "metadata": file_identity(metadata),
    }
    write_json(workspace / (target + "-image-binding.json"), result)
    return result


STACK_PROBE = r"""
import hashlib,json,os,pathlib,subprocess
roots=[pathlib.Path('/opt/physicsnemo-inference/lib')]
import torch
roots.append(pathlib.Path(torch.__file__).parent/'lib')
try:
 import warp
 roots.append(pathlib.Path(warp.__file__).parent/'bin')
except ImportError:pass
extra={}
native=set()
if os.environ.get('PNMS_RELEASE_BACKEND','aoti')=='tensorrt':
 import tensorrt
 extra['tensorrt_version']=tensorrt.__version__
 trt_root=pathlib.Path(tensorrt.__file__).parent
 roots.append(trt_root)
 roots.extend(p for p in trt_root.parent.glob('tensorrt*') if p.is_dir())
 libraries=subprocess.run(['ldconfig','-p'],capture_output=True,text=True,check=True,timeout=30)
 for line in libraries.stdout.splitlines():
  if '=>' in line and any(name in line.split('=>')[0] for name in ('libnvinfer','libnvonnxparser')):
   native.add(pathlib.Path(line.split('=>')[1].strip()))
files={}
for root in roots:
 assert root.is_dir(),str(root)
 for p in sorted(root.rglob('*')):
  if p.is_file() and ('.so' in p.name or p.suffix in ('.dll','.dylib') or (extra and p.suffix=='.a')):
   q=p.resolve()
   with q.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
   files[str(p)]={'sha256':h,'size_bytes':q.stat().st_size}
if extra:
 assert any('libnvinfer.so' in str(p) for p in native) or any('libnvinfer.so' in p for p in files),'TensorRT runtime library is absent'
 native={p.resolve() for p in native}
 native.update(pathlib.Path(p).resolve() for p in files)
 visited=set()
 def driver_library(name):
  return name.startswith(('libcuda.so','libnvidia-'))
 while native:
  q=min(native);native.remove(q)
  if q in visited:continue
  visited.add(q)
  assert q.is_file(),str(q)
  with q.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
  files[str(q)]={'sha256':h,'size_bytes':q.stat().st_size}
  if q.suffix=='.a':continue
  dependencies=subprocess.run(['ldd',str(q)],capture_output=True,text=True,timeout=30)
  assert dependencies.returncode==0,(str(q),dependencies.stdout,dependencies.stderr)
  for line in dependencies.stdout.splitlines():
   # GPU drivers are injected by the deployment runtime and checked by the target policy.
   tokens=line.strip().split()
   if not tokens or driver_library(tokens[0]):continue
   assert 'not found' not in line,(str(q),line)
   values=line.split('=>')[-1].strip().split()
   if values and values[0].startswith('/') and not driver_library(pathlib.Path(values[0]).name):
    native.add(pathlib.Path(values[0]).resolve())
print(json.dumps({'schema_version':1,'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'files':files,**extra},sort_keys=True))
"""


def verify_stack_images(resolved, runner, workspace):
    observed = {}
    for kind in ("builder", "server"):
        image = resolved["images"][kind]
        if image in observed:
            continue
        result = runner.run(
            [
                "docker",
                "run",
                "--rm",
                "--runtime",
                "runc",
                "--env",
                "NVIDIA_VISIBLE_DEVICES=void",
                "--env",
                "PNMS_RELEASE_BACKEND=" + resolved["backend"],
                "--entrypoint",
                "python3",
                image,
                "-c",
                STACK_PROBE,
            ],
            timeout=1200,
        )
        # NVIDIA may emit informational messages before the final JSON record.
        lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
        require(lines, "native stack probe produced no evidence")
        observed[image] = json.loads(lines[-1])
        require(observed[image].get("files"), "native stack closure inventory is empty")
    builder, server = (
        observed[resolved["images"][kind]] for kind in ("builder", "server")
    )
    require(
        builder == server,
        "builder and server native stack fingerprints differ; select matched immutable images",
    )
    proof = {
        "schema_version": 1,
        "images": resolved["images"],
        "fingerprint": content_id(builder),
        "observed": builder,
    }
    write_json(Path(workspace) / "native-stack.json", proof)
    return proof


def inspect_qa_sources(root="/opt/native-release/source/qa"):
    """Fail the CPU image gate if subprocess imports select inherited base QA."""
    import importlib
    import importlib.util

    root = Path(root).resolve(strict=True)
    modules = {}
    for name in (
        "qa.scripts.run_native_cfd_reference",
        "qa.scripts.run_native_cfd_job",
        "qa.scripts.run_native_cfd_http_job",
        "qa.scripts.cfd_parity_contract",
        "qa.native_inference.cfd_reference",
        "qa.native_inference.cfd_startup",
        "qa.native_inference.cfd_native",
        "qa.native_inference.cfd_evaluate",
        "qa.native_inference.cfd_contract",
    ):
        spec = importlib.util.find_spec(name)
        require(
            spec is not None and spec.origin is not None, f"missing QA module: {name}"
        )
        path = Path(spec.origin).resolve(strict=True)
        require(
            path.is_relative_to(root),
            f"QA module is outside sealed source: {name}: {path}",
        )
        modules[name] = {"path": str(path), **file_identity(path)}
    reference = importlib.import_module("qa.scripts.run_native_cfd_reference")
    for command in ("capture", "evaluate"):
        argv = [
            command,
            "--profile",
            "profile.json",
            "--request",
            "request.json",
            "--input-root",
            "inputs",
            "--image",
            "image",
            "--source-sha",
            "source",
            "--output",
            "out",
            "--checkpoint",
            "Transolver.0.501.mdlus",
            "--stats",
            "global_stats.json",
        ]
        if command == "evaluate":
            argv += ["--fields", "fields.json"]
        parsed = reference.parser().parse_args(argv)
        require(
            parsed.checkpoint == Path("Transolver.0.501.mdlus")
            and parsed.stats == Path("global_stats.json"),
            "sealed provider QA CLI does not accept verified local model assets",
        )
    return {"modules": modules, "local_model_asset_flags": True}


IMAGE_PROBE = r"""
import json,os,pathlib,subprocess,sys
sys.path.insert(0,'/opt/native-release')
from native_release import verify_bundle,file_identity,inspect_qa_sources
root=pathlib.Path(os.environ['NATIVE_DEPLOYMENT_CONFIG']).parent
record=verify_bundle(root)
deployment=json.loads((root/'deployment.json').read_text())
backends={model['backend'] for model in deployment.get('workflow',{}).get('models',{}).values()} or {deployment.get('backend','aoti')}
assert len(backends)==1 and backends<={'aoti','tensorrt'},backends
backend=next(iter(backends))
assert backend==record.get('backend',backend)
qa=inspect_qa_sources()
assert not list(pathlib.Path('/dev').glob('nvidia*'))
assert os.environ['SOURCE_SHA']==record['source_sha']
assert os.environ['PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID']==record['workflow']
paths=['/app/bin/native-worker','/app/bin/inference-server','/app/bin/worker-runtime','/opt/physicsnemo-native-runner/lib/libtransolver_workflow.so','/opt/physicsnemo-workflow/lib/libphysicsnemo_workflow.so']
evidence={}
for raw in paths:
 p=pathlib.Path(raw);assert p.is_file(),raw
 q=subprocess.run(['ldd',raw],capture_output=True,text=True,timeout=30)
 assert q.returncode==0 and 'not found' not in q.stdout+q.stderr,(raw,q.stdout,q.stderr)
 evidence[raw]={'file':file_identity(p),'dependencies':q.stdout}
extra={}
if record.get('management_capabilities'):
 extra['management_files']={'startup_guard':file_identity('/opt/native-release/native_deploy_start.py')}
if backend=='tensorrt':
 import tensorrt
 assert any('libnvinfer.so' in item['dependencies'] for item in evidence.values()),'selected TensorRT backend is absent from native dependency closure'
 extra['tensorrt_version']=tensorrt.__version__
print(json.dumps({'schema_version':1,'source_sha':record['source_sha'],'workflow':record['workflow'],'backend':backend,'bundle_digest':record['bundle_digest'],'deployment_sha256':file_identity(root/'deployment.json')['sha256'],'management_capabilities':record.get('management_capabilities',[]),'native_binaries':evidence,'qa_sources':qa,'gpu_devices':[],**extra},sort_keys=True))
"""


def inspect_candidate(image, *, runner, workspace, expected):
    result = runner.run(
        [
            "docker",
            "run",
            "--rm",
            "--runtime",
            "runc",
            "--env",
            "NVIDIA_VISIBLE_DEVICES=void",
            "--entrypoint",
            "python3",
            image_digest(image),
            "-c",
            IMAGE_PROBE,
        ],
        timeout=1200,
    )
    lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
    require(lines, "final image inspection produced no evidence")
    proof = json.loads(lines[-1])
    require(
        "backend" not in expected or proof.get("backend") == expected["backend"],
        "final image backend differs from assembled bundle",
    )
    require(
        all(
            proof.get(key) == expected[key]
            for key in ("source_sha", "workflow", "bundle_digest")
        ),
        "final image content identity differs from assembled bundle",
    )
    proof["image_digest"] = image
    if "management_capabilities" in expected:
        require(
            proof.get("management_capabilities") == expected["management_capabilities"],
            "final image management capabilities differ",
        )
    write_json(Path(workspace) / "candidate-inspection.json", proof)
    return proof


def validate_qualification(report, *, image, resolved, bundle):
    require(
        report.get("status") == "passed"
        and report.get("image_digest") == image
        and report.get("source_sha") == resolved["source_sha"],
        "qualification is not bound to the exact candidate image/source",
    )
    require(
        report.get("workflow") == resolved["workflow"]
        and report.get("bundle_digest") == bundle["bundle_digest"]
        and report.get("target") == resolved["target"],
        "qualification workflow, bundle or full target policy differs",
    )
    expected = required_http_cases(resolved)
    require(
        report.get("expected_cases") == expected
        and [row.get("name") for row in report.get("cases", [])] == expected,
        "qualification required case inventory differs",
    )
    require(
        all(row.get("status") == "passed" for row in report["cases"]),
        "every required image qualification case must pass",
    )
    return report


def required_http_cases(resolved):
    if resolved.get("adapter") == "transolver_surface":
        return [
            "preflight",
            "start_stack",
            "readiness",
            "first_request",
            "second_request",
            "redis_completion",
            "artifact_downloads",
            "native_fields",
            "shared_evaluator",
            "parity",
            "malformed_request",
            "overlap",
            "worker_restart",
            "active_recovery",
            "final_completion",
            "admission_rejection",
            "tampered_model",
            "shutdown",
        ]
    return [
        "preflight",
        "startup",
        *["fixture." + name for name in resolved["qualification"]["required_cases"]],
        "malformed_request",
        "overlap",
        "worker_restart",
        "redis_completion",
        "artifacts",
        "admission_rejection",
        "tampered_model",
        "shutdown",
    ]


def _compile_handoff(
    resolved,
    plan,
    candidates,
    missing,
    *,
    source,
    custom,
    inputs,
    empty,
    workspace,
    artifact_workspace,
    runner,
    lepton,
    cache_dir,
    stack,
    process_runner,
):
    is_cfd = resolved.get("adapter") == "transolver_surface"
    artifact_workspace.mkdir(parents=True, exist_ok=True)
    if is_cfd:
        for alias, candidate_path in candidates.items():
            _copy_inventory(
                candidate_path,
                inventory(candidate_path),
                inputs / "reuse" / alias,
            )
    else:
        for alias in missing:
            model = resolved["model_aliases"][alias]
            _copy_inventory(
                Path(model["project"]).parent,
                model["project_files"],
                inputs / "projects" / alias,
            )
    build_aliases = (
        {
            alias: {**value, "action": "compile" if alias in missing else "reuse"}
            for alias, value in plan["models"].items()
        }
        if is_cfd
        else missing
    )
    write_json(
        inputs / "request.json",
        {
            "schema_version": 1,
            "resolved": resolved,
            "build_aliases": build_aliases,
            "builder_image": resolved["images"]["builder"],
            "stack_fingerprint": stack["fingerprint"],
        },
    )
    built = build_image(
        resolved,
        source=source,
        custom=custom,
        inputs=inputs,
        bundle=empty,
        workspace=artifact_workspace,
        runner=runner,
        target="compiler",
    )
    run_id = "release-build-" + uuid.uuid4().hex[:16]
    if is_cfd:
        from native_release_cfd_job import CASES as cfd_cases

        cases = cfd_cases
    else:
        cases = [
            "preflight",
            *["compile." + alias for alias in sorted(missing)],
            "seal",
        ]
    job = lepton.run_job(
        image=built["image"],
        source_sha=resolved["source_sha"],
        run_id=run_id,
        mode="cfd_compile" if is_cfd else "compile",
        required_cases=cases,
        artifact_dir=workspace / "jobs",
        config=resolved.get("lepton"),
        process_runner=process_runner,
    )
    transfer = lepton.fetch_artifact(
        image=built["image"],
        job=job,
        expected={
            "run_id": run_id,
            "image_digest": built["image"],
            "source_sha": resolved["source_sha"],
            "builder_image": resolved["images"]["builder"],
            "compilation_keys": {
                alias: plan["models"][alias]["compilation_key"]
                for alias in build_aliases
            },
        },
        output=artifact_workspace / "handoff",
        runner=runner,
    )
    for alias in missing:
        received = Path(transfer["extracted"]) / "aliases" / alias
        if is_cfd:
            provenance = read_json(received / "release-provenance.json")
            provenance["placement"] = {
                "node_id": job["controller"]["node_id"],
                "job_id": job["controller"]["job_id"],
            }
            write_json(received / "release-provenance.json", provenance)
        cached = cache_publish(
            cache_dir,
            plan["models"][alias]["compilation_key"],
            received,
            backend=resolved["backend"],
            expected_identity=plan["models"][alias]["input_identity"],
            target=resolved["target"]["compilation"],
        )
        require(
            cached["status"] == "hit",
            "compiled artifact could not be verified in cache",
        )
        candidates[alias] = Path(cached["path"])
    return (
        candidates,
        {"image": built, "job": job["controller"], "transfer": transfer},
        Path(transfer["extracted"])
        / "aliases"
        / next(iter(build_aliases))
        / "cfd-release"
        if is_cfd
        else None,
    )


def release(resolved, *, cache_dir, workspace=None, promote=None, process_runner=None):
    """Build/reuse, assemble, inspect and qualify; promote only the tested digest."""
    require(
        not resolved.get("model_only"), "model-only resolution must use check_models"
    )
    import native_release_lepton as lepton

    if promote is not None:
        require(
            isinstance(promote, str)
            and re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", promote),
            "promotion requires a registry tag",
        )
    if resolved.get("adapter") == "transolver_surface":
        resolved = resolve_cfd_profile(resolved)
    controller = lepton.validate_controller_config(resolved)
    resolved = {
        **resolved,
        "lepton": controller,
        "_reference_storage": reference_storage_identity(controller),
    }
    plan = plan_release(resolved, cache_dir)
    workspace = Path(workspace or resolved["paths"]["workspace"])
    require(
        not workspace.exists() and not workspace.is_symlink(),
        "release workspace already exists; choose a fresh directory",
    )
    workspace.mkdir(parents=True, exist_ok=False)
    runner = process_runner or ProcessRunner(workspace / "commands")
    report = {
        "schema_version": 1,
        "status": "running",
        "workflow": resolved["workflow"],
        "source_sha": resolved["source_sha"],
        "plan": plan,
        "promoted": False,
    }
    write_json(workspace / "resolved.json", resolved)
    write_json(workspace / "release.json", report)
    try:
        stack = verify_stack_images(resolved, runner, workspace)
        resolved = {**resolved, "_native_stack_fingerprint": stack["fingerprint"]}
        is_cfd = resolved.get("adapter") == "transolver_surface"
        source = workspace / "source"
        _copy_inventory(
            resolved["paths"]["repository_root"], plan["source_files"], source
        )
        custom = workspace / "custom"
        if resolved.get("native_workflow"):
            _copy_inventory(
                resolved["native_workflow"]["source_dir"],
                resolved["native_workflow"]["files"],
                custom,
            )
        else:
            custom.mkdir()
        inputs = workspace / "compile-input"
        inputs.mkdir()
        empty = workspace / "empty"
        empty.mkdir()
        missing = {
            alias: value
            for alias, value in plan["models"].items()
            if value["action"] == "compile"
        }
        candidates = {
            alias: Path(value["candidate"])
            for alias, value in plan["models"].items()
            if value["action"] == "reuse"
        }
        recompiled_aliases = set(missing)
        joint = (
            cfd_joint_lookup(cache_dir, resolved, candidates)
            if is_cfd and not missing
            else {"status": "miss"}
        )
        attempts = []
        for attempt in range(2 if is_cfd else 1):
            if not missing and (not is_cfd or joint["status"] == "hit"):
                break
            if attempt:
                inputs = workspace / ("compile-input-" + str(attempt))
                inputs.mkdir()
            attempt_workspace = (
                workspace / ("compilation-attempt-" + str(attempt))
                if is_cfd
                else workspace
            )
            candidates, compiled, proof = _compile_handoff(
                resolved,
                plan,
                candidates,
                missing,
                source=source,
                custom=custom,
                inputs=inputs,
                empty=empty,
                workspace=workspace,
                artifact_workspace=attempt_workspace,
                runner=runner,
                lepton=lepton,
                cache_dir=cache_dir,
                stack=stack,
                process_runner=process_runner,
            )
            attempts.append(compiled)
            report["compilation"] = {**compiled, "attempts": attempts}
            if is_cfd:
                try:
                    joint = cfd_joint_publish(
                        cache_dir,
                        resolved,
                        candidates,
                        proof,
                        node_id=compiled["job"]["node_id"],
                    )
                except ValueError as error:
                    if (
                        attempt == 0
                        and str(error)
                        == "CFD joint proof does not cover selected package bytes"
                    ):
                        # A concurrent first publisher won one model key with
                        # different valid bytes. Requalify the actual winners,
                        # using both verified packages; do not compile again.
                        missing = {}
                        continue
                    raise
            break
        if is_cfd:
            require(
                joint["status"] == "hit",
                "CFD joint requalification did not produce a verified proof",
            )
            report["joint_qualification"] = {
                key: value for key, value in joint.items() if key != "identity"
            }
        bundle = workspace / "bundle"
        staged = (
            stage_cfd_release(
                resolved,
                candidates,
                bundle,
                sources=plan["source_files"],
                joint_proof=joint,
            )
            if is_cfd
            else stage_release(
                resolved, candidates, bundle, sources=plan["source_files"]
            )
        )
        candidate = build_image(
            resolved,
            source=source,
            custom=custom,
            inputs=empty,
            bundle=bundle,
            workspace=workspace,
            runner=runner,
            target="release",
        )
        inspect_candidate(
            candidate["image"], runner=runner, workspace=workspace, expected=staged
        )
        run_id = "release-http-" + uuid.uuid4().hex[:16]
        job = lepton.run_job(
            image=candidate["image"],
            source_sha=resolved["source_sha"],
            run_id=run_id,
            mode="cfd_http" if is_cfd else "http",
            required_cases=required_http_cases(resolved),
            artifact_dir=workspace / "jobs",
            config=resolved.get("lepton"),
            process_runner=process_runner,
            node_id=read_json(bundle / "orchestration.json")["node_id"]
            if is_cfd
            else None,
        )
        qualified = validate_qualification(
            job["summary"], image=candidate["image"], resolved=resolved, bundle=staged
        )
        write_json(workspace / "qualification.json", qualified)
        evidence = lepton.fetch_artifact(
            image=candidate["image"],
            job=job,
            expected={
                "run_id": run_id,
                "source_sha": resolved["source_sha"],
                "image_digest": candidate["image"],
                "qualification": "exact-image-service-evidence-v1",
            },
            output=workspace / "qualification-evidence",
            runner=runner,
        )
        report["qualification_evidence"] = evidence
        report.update(
            status="passed",
            candidate=candidate,
            bundle_digest=staged["bundle_digest"],
            stack_fingerprint=stack["fingerprint"],
            management_capabilities=staged.get("management_capabilities", []),
            deployment_sha256=file_identity(bundle / "deployment.json")["sha256"],
            qualification=file_identity(workspace / "qualification.json"),
            models={
                alias: {
                    "compilation_key": plan["models"][alias]["compilation_key"],
                    "recompiled": alias in recompiled_aliases,
                }
                for alias in candidates
            },
        )
        write_json(workspace / "release.json", report)
        if promote:
            require(
                isinstance(promote, str)
                and re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", promote),
                "promotion requires a registry tag",
            )
            destination = resolved["output"]["repository"] + ":" + promote
            runner.run(
                [
                    "docker",
                    "buildx",
                    "imagetools",
                    "create",
                    "--prefer-index=false",
                    "--tag",
                    destination,
                    candidate["image"],
                ],
                timeout=300,
            )
            actual = (
                runner.run(
                    [
                        "docker",
                        "buildx",
                        "imagetools",
                        "inspect",
                        "--format",
                        "{{json .Manifest.Digest}}",
                        destination,
                    ],
                    timeout=120,
                )
                .stdout.strip()
                .strip('"')
            )
            require(
                actual == candidate["image"].rsplit("@", 1)[1],
                "promotion changed the tested image digest",
            )
            report.update(promoted=True, promoted_tag=destination)
        return report
    except BaseException as error:
        report.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        write_json(workspace / "release.json", report)
