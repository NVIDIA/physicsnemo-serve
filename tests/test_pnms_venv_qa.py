from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qa.native_inference import pnms_venv_job as job  # noqa: E402

IDENTITY = {
    "run_id": "venv-unit",
    "source_sha": "a" * 40,
    "image_digest": "registry.example/qa@sha256:" + "b" * 64,
    "profile": "fresh-venv",
}


def passed_report():
    cases = []
    for name in job.expected_cases():
        details = {}
        if name == "venv.setup":
            details = {
                "fresh": True,
                "system_site_packages": False,
                "sdk_built": True,
                "requirements": [
                    {"path": "/outputs/requirements.txt", "sha256": "0" * 64}
                ],
            }
        elif name == "venv.verify":
            details = {
                "isolated": True,
                "pip_check": "passed",
                "visible_gpu_count": 1,
                "module_origins_verified": True,
                "sdk_provider_paths_verified": True,
                "python": "/outputs/venv/bin/python",
                "prefix": "/outputs/venv",
                "base_prefix": "/usr",
                "torch": "2.10.0+cu130",
                "cuda": "13.0",
                "tensorrt": "10.14.1.48",
                "gpu": {"uuid": "GPU-allocated"},
            }
        elif name.endswith(".build"):
            details = {
                "build_id": "c" * 32,
                "bundle_digest": "d" * 64,
                "model_identity": "e" * 64,
                "native_checks": {"verified": True, "case_count": 3},
            }
        elif name.endswith(".held_out"):
            details = {"case_count": 3, "parity": "passed"}
        elif name.endswith(".cache_reuse") or name == "environment_lock.refresh":
            details = {
                "model_reused": True,
                "workflow_reused": True,
                "environment_lock_digest": "1" * 64,
                "build_id": "f" * 32,
                "bundle_digest": "d" * 64,
            }
            if name == "environment_lock.refresh":
                details.update(
                    drift_rejected=True,
                    prior_success_preserved=True,
                    refresh_passed=True,
                )
        cases.append({"name": name, "status": "passed", "details": details})
    return {
        "schema_version": 1,
        **IDENTITY,
        "status": "passed",
        "qa_scope": "fresh-venv-workflow",
        "expected_cases": job.expected_cases(),
        "cases": cases,
        "http_qa": "not_run",
        "release_candidates": [
            {
                "backend": backend,
                "template": template,
                "ingress": "application/json",
                "build_id": "f" * 32,
                "bundle_digest": "d" * 64,
            }
            for backend, template in job.NATIVE_TARGETS
        ],
    }


def test_inventory_requires_fresh_isolated_venv_and_both_native_backends():
    assert len(job.expected_cases()) == 18
    assert job.expected_cases()[:3] == ["environment", "venv.setup", "venv.verify"]
    assert "native.aoti.tensor.build" in job.expected_cases()
    assert "native.tensorrt.custom.build" in job.expected_cases()
    assert "environment_lock.refresh" in job.expected_cases()
    job.validate_summary(passed_report(), **IDENTITY)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "foreign",
        "global-torch",
        "shared-sdk",
        "no-pip-check",
        "workflow-miss",
        "no-lock",
        "stale-candidate",
        "no-parity",
    ],
)
def test_summary_rejects_incomplete_or_unisolated_evidence(mutation):
    report = copy.deepcopy(passed_report())
    by_name = {case["name"]: case["details"] for case in report["cases"]}
    if mutation == "missing":
        report["cases"].pop()
    elif mutation == "duplicate":
        report["cases"].append(report["cases"][0])
    elif mutation == "foreign":
        report["image_digest"] = "other"
    elif mutation == "global-torch":
        by_name["venv.verify"]["module_origins_verified"] = False
    elif mutation == "shared-sdk":
        by_name["venv.setup"]["sdk_built"] = False
    elif mutation == "no-pip-check":
        by_name["venv.verify"]["pip_check"] = "not_run"
    elif mutation == "workflow-miss":
        by_name["native.tensorrt.custom.cache_reuse"]["workflow_reused"] = False
    elif mutation == "no-lock":
        by_name["native.aoti.tensor.cache_reuse"].pop("environment_lock_digest")
    elif mutation == "stale-candidate":
        report["release_candidates"][0]["build_id"] = "c" * 32
    else:
        by_name["native.aoti.tensor.held_out"]["parity"] = "not_run"
    with pytest.raises(ValueError):
        job.validate_summary(report, **IDENTITY)


def test_environment_removes_inherited_python_and_pip_configuration():
    result = job.isolated_environment(
        {
            "PATH": "/usr/bin",
            "PYTHONPATH": "/image/packages",
            "PYTHONHOME": "/image/python",
            "VIRTUAL_ENV": "/old",
            "PIP_EXTRA_INDEX_URL": "https://wrong.example",
            "LD_LIBRARY_PATH": os.pathsep.join(
                [
                    "/usr/local/cuda/compat",
                    "/image/site-packages/torch/lib",
                    "/opt/physicsnemo-inference/lib",
                ]
            ),
        }
    )
    assert "PYTHONPATH" not in result and "PYTHONHOME" not in result
    assert "PIP_EXTRA_INDEX_URL" not in result and "VIRTUAL_ENV" not in result
    assert result["PYTHONNOUSERSITE"] == "1"
    assert result["PIP_CONFIG_FILE"] == os.devnull
    assert result["LD_LIBRARY_PATH"] == "/usr/local/cuda/compat"


def test_setup_command_builds_sdk_from_source_and_never_enables_system_packages(
    tmp_path,
):
    from types import SimpleNamespace

    args = SimpleNamespace(
        python=Path(sys.executable),
        repository=ROOT,
        requirements=[tmp_path / "requirements.txt"],
        tensorrt_root=Path("/usr"),
        tensorrt_cuda_major="13",
    )
    command = job.setup_command(args, tmp_path / "fresh")
    assert "setup-env" in command and command.count("--backend") == 2
    assert "--runtime" not in command and "--system-site-packages" not in command
    assert str(ROOT / "native-inference/cpp-runtime") in command
    assert str(args.requirements[0]) in command


def test_requirement_digest_mismatch_fails_before_setup(tmp_path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("torch==2.10.0+cu130\n")
    with pytest.raises(ValueError, match="digest"):
        job.requirement_inventory([requirements], ["0" * 64])
    assert job.requirement_inventory([requirements], [job.sha256(requirements)])[0][
        "path"
    ] == str(requirements)


def test_actual_fresh_venv_has_no_image_site_packages_even_with_host_pythonpath(
    tmp_path,
):
    venv = tmp_path / "fresh"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True
    )
    trap = tmp_path / "image-packages"
    trap.mkdir()
    (trap / "image_only.py").write_text(
        "raise AssertionError('image package leaked')\n"
    )
    environment = job.isolated_environment(
        {"PATH": str(Path(sys.executable).parent), "PYTHONPATH": str(trap)}
    )
    result = subprocess.run(
        [
            str(
                venv
                / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
            ),
            "-c",
            job.ISOLATION_PROBE,
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    details = json.loads(result.stdout)
    job.verify_isolation(details, venv)
    assert details["prefix"] != details["base_prefix"]
    assert str(trap) not in details["sys_path"]


def test_isolation_rejects_global_dist_packages_even_with_venv_prefix(tmp_path):
    details = {
        "python": str(tmp_path / "bin/python"),
        "prefix": str(tmp_path),
        "base_prefix": "/usr",
        "sys_path": [
            str(tmp_path / "lib/python3.12/site-packages"),
            "/image/lib/python3.12/dist-packages",
        ],
        "no_user_site": 1,
    }
    with pytest.raises(ValueError, match="site-packages|dist-packages"):
        job.verify_isolation(details, tmp_path)


def controller_args(tmp_path):
    from scripts import run_lepton_pnms_venv_qa as controller

    args = controller.build_parser().parse_args(
        [
            "--image",
            IDENTITY["image_digest"],
            "--expected-source-sha",
            IDENTITY["source_sha"],
            "--run-id",
            IDENTITY["run_id"],
            "--artifact-dir",
            str(tmp_path),
            "--workspace-id",
            "unit-workspace",
            "--node-group",
            "unit-group",
            "--pull-secret",
            "unit-secret",
            "--resource-shape",
            "gpu.h100-sxm",
            "--nfs-mount-base",
            "/mnt/shared",
            "--requirements",
            "/outputs/qa-inputs/requirements.txt",
            "--requirements-sha256",
            "0" * 64,
        ]
    )
    return controller, args


def test_controller_is_finite_ordinary_job_and_passes_pinned_inputs(tmp_path):
    controller, args = controller_args(tmp_path)
    command = controller.build_job_command(args, "unit-job")
    assert "--privileged" not in command
    assert command[command.index("--num-workers") + 1] == "1"
    assert command[command.index("--max-failure-retry") + 1] == "0"
    remote = shlex.split(command[command.index("--command") + 1])
    assert any(value.endswith("pnms_venv_job.py") for value in remote)
    assert "docker" not in remote and "--sdk-prefix" not in remote
    assert (
        remote[remote.index("--requirements") + 1]
        == "/outputs/qa-inputs/requirements.txt"
    )
    assert remote[remote.index("--requirements-sha256") + 1] == "0" * 64


def test_controller_rejects_unmounted_requirements_before_submission(
    tmp_path, monkeypatch
):
    controller, args = controller_args(tmp_path)
    args.requirements = ["/etc/requirements.txt"]
    monkeypatch.setattr(
        controller.lifecycle, "run", lambda *a, **k: pytest.fail("submitted")
    )
    with pytest.raises(ValueError, match="mount"):
        controller.run(args)


def test_controller_uses_existing_lifecycle_validation_and_cleanup(
    tmp_path, monkeypatch
):
    controller, args = controller_args(tmp_path)
    calls = []

    def fake(command, **kwargs):
        calls.append(command)
        if command[:2] == ["job", "create"]:
            return 0, "ID: venv-unit-job\n"
        if command[:2] == ["job", "get"]:
            return 0, '{"state":"Completed"}'
        if command[:2] == ["log", "get"]:
            return 0, "\n".join(
                "2026-10-29 16:00:00.000001|" + json.dumps(line)
                for line in [
                    job.SUMMARY_BEGIN,
                    json.dumps(passed_report()),
                    job.SUMMARY_END,
                ]
            )
        return 0, ""

    monkeypatch.delenv("LEPTON_WORKSPACE_TOKEN", raising=False)
    monkeypatch.setattr(controller.lifecycle, "lep", fake)
    assert controller.run(args) == 0
    assert calls[-1] == ["job", "remove", "-i", "venv-unit-job"]


def test_stack_verification_rejects_one_global_dependency_before_inference(tmp_path):
    from types import SimpleNamespace

    venv = tmp_path / "venv"
    stack = {
        "python": str(venv / "bin/python"),
        "prefix": str(venv),
        "base_prefix": "/usr",
        "sys_path": [str(venv / "lib/python3.12/site-packages")],
        "no_user_site": 1,
        "visible_gpu_count": 1,
        "origins": {
            name: str(venv / f"lib/python3.12/site-packages/{name}/__init__.py")
            for name in (
                "torch",
                "tensorrt",
                "numpy",
                "onnx",
                "onnxscript",
                "model_builder",
                "pnms_app",
            )
        },
    }
    stack["origins"]["torch"] = "/image/lib/python3.12/site-packages/torch/__init__.py"
    context = SimpleNamespace(
        venv=venv,
        args=SimpleNamespace(python=venv / "bin/python"),
        command=lambda *a, **k: stack,
    )
    with pytest.raises(ValueError, match="torch.*outside"):
        job.verify(context)


def test_setup_uses_fresh_sdk_and_keeps_checkout_unchanged(tmp_path):
    from types import SimpleNamespace

    repository = tmp_path / "repository"
    application = repository / "application-cli"
    application.mkdir(parents=True)
    (application / "pyproject.toml").write_text('[project]\nname="unit-app"\n')
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("unit-dependency==1.0\n")
    root = tmp_path / "output"
    root.mkdir()
    venv = root / "venv"
    calls = []
    args = SimpleNamespace(
        python=Path(sys.executable),
        repository=repository,
        requirements=[requirements],
        requirements_sha256=[job.sha256(requirements)],
        tensorrt_root=Path("/usr"),
        tensorrt_cuda_major="13",
    )

    def command(label, arguments, **kwargs):
        calls.append((label, list(map(str, arguments))))
        if label == "gpu.target":
            result = root / "logs/gpu.target/result.json"
            result.parent.mkdir(parents=True)
            result.write_text('{"architecture":"90"}')
            return {"architecture": "90"}
        if label == "venv.tensorrt":
            return {"python": "10.14.1.48.post1", "cpp": "10.14.1.48"}
        if label == "venv.setup":
            venv.mkdir()
            (venv / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
            return {
                "status": "complete",
                "stage": "environment",
                "python": str(venv / "bin/python"),
                "runtime": str(venv / ".physicsnemo/runtime/physicsnemo-infer"),
                "backends": ["aoti", "tensorrt"],
            }
        return {"exit_code": 0}

    context = SimpleNamespace(
        root=root,
        venv=venv,
        args=args,
        repository=repository,
        environment={"PATH": "/usr/bin"},
        command=command,
    )
    result = job.setup(context)
    assert result["fresh"] and result["sdk_built"]
    assert args.python == venv / "bin/python"
    assert args.sdk_prefix == venv / ".physicsnemo/sdk"
    assert context.environment["CUDAARCHS"] == "90"
    assert context.environment["TORCH_CUDA_ARCH_LIST"] == "9.0"
    install_app = next(command for name, command in calls if name == "venv.application")
    assert str(root / "application-source") in install_app
    assert str(application) not in install_app
    assert sorted(path.name for path in application.iterdir()) == ["pyproject.toml"]
    probe = next(
        command for name, command in calls if name == "venv.tensorrt-dependencies"
    )
    assert "_DEPENDENCY_PROBE" in probe[2]
    assert probe[-3:] == ["13", "10.14.1.48", "aoti"]
    assert not any(name == "venv.sdk-configure" for name, _ in calls)
    setup = next(command for name, command in calls if name == "venv.setup")
    assert [setup[i + 1] for i, value in enumerate(setup) if value == "--backend"] == [
        "aoti",
        "tensorrt",
    ]
    assert result["setup_backends"] == ["aoti", "tensorrt"]
    assert result["sdk_backends"] == ["aoti", "tensorrt"]
    assert result["tensorrt_compatibility"]["compatible"] is True
    assert calls[-1][0] == "venv.sdk-install"
    assert str(args.sdk_prefix) in calls[-1][1]


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Linux QA FindCUDA reproducer uses POSIX nvcc shell programs and tool symlinks",
)
def test_cuda_toolkit_discovery_uses_real_nvcc_prefix_with_isolated_path(tmp_path):
    """Execute CMake FindCUDA, whose nvcc-prefix logic is also used by Torch."""
    import os
    import shutil

    cmake = shutil.which("cmake")
    if not cmake:
        pytest.skip("CMake is unavailable")
    toolkit = tmp_path / "cuda"
    binary = toolkit / "bin/nvcc"
    binary.parent.mkdir(parents=True)
    binary.write_text(
        "#!/bin/sh\necho 'Cuda compilation tools, release 13.0, V13.0.0'\n"
    )
    binary.chmod(0o755)
    isolated = tmp_path / "qa-output/host-tools"
    selected_path = job.local.docker_free_path(str(binary.parent), isolated)
    probe = tmp_path / "toolkit.cmake"
    result = tmp_path / "toolkit.txt"
    probe.write_text(
        'cmake_minimum_required(VERSION 3.20)\ninclude(CMakeInitializeConfigs)\ninclude("${CMAKE_ROOT}/Modules/FindCUDA.cmake")\n'
        f'file(WRITE "{result.as_posix()}" "${{CUDA_TOOLKIT_ROOT_DIR}}")\n'
    )
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("CUDA")
    }
    environment.update(
        PATH=selected_path,
        CUDA_HOME=str(toolkit),
        CUDAToolkit_ROOT=str(toolkit),
        CUDACXX=str(binary),
    )
    subprocess.run(
        [cmake, "-P", str(probe)],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert (
        Path(result.read_text()).resolve().is_relative_to(isolated.parent.resolve())
    ), "Reproducer must expose FindCUDA's lexical symlink-prefix behavior"
    job.configure_cuda_environment(environment)
    subprocess.run(
        [cmake, "-P", str(probe)],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert Path(result.read_text()).resolve() == toolkit.resolve()


def test_qa_children_do_not_write_bytecode_into_staged_source(tmp_path):
    import os

    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "staged_model.py").write_text("value = 7\n")
    script = "import sys; sys.path.insert(0, sys.argv[1]); import staged_model; assert staged_model.value == 7"
    subprocess.run(
        [sys.executable, "-c", script, str(stage)],
        env=job.isolated_environment(os.environ),
        check=True,
    )
    assert not (stage / "__pycache__").exists()


def test_controller_disables_initial_runner_bytecode_before_source_import(tmp_path):
    controller, args = controller_args(tmp_path)
    command = controller.build_job_command(args, "unit-job")
    remote = shlex.split(command[command.index("--command") + 1])
    assert remote[remote.index(args.python) + 1] == "-B"


def test_setup_env_qualifies_the_public_dual_backend_bootstrap(tmp_path):
    from types import SimpleNamespace

    args = SimpleNamespace(
        python=Path(sys.executable),
        repository=ROOT,
        requirements=[tmp_path / "requirements.txt"],
        tensorrt_root=Path("/usr"),
        tensorrt_cuda_major="13",
    )
    command = job.setup_command(args, tmp_path / "fresh")
    backends = [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--backend"
    ]
    assert backends == ["aoti", "tensorrt"]
    assert "--runtime" not in command


def test_python_trt_post_release_matches_cpp_version_but_other_sdk_is_rejected():
    assert job.validate_tensorrt_versions("10.14.1.48.post1", "10.14.1.48") == {
        "python": "10.14.1.48.post1",
        "cpp": "10.14.1.48",
        "compatible": True,
    }
    with pytest.raises(ValueError, match="TensorRT"):
        job.validate_tensorrt_versions("10.15.1.29", "10.14.1.48")


@pytest.mark.parametrize(
    "version",
    [
        "10.14.1.49",
        "10.14.1.48.dev1",
        "10.14.1.48rc1",
        "10.14.1.48+local",
        "10.14.1.48.post",
        "10.14.1.48.post1+local",
        "10.14.1.48.1",
    ],
)
def test_qa_tensorrt_check_rejects_different_native_build_or_nonpost_suffix(version):
    with pytest.raises(ValueError, match="TensorRT"):
        job.validate_tensorrt_versions(version, "10.14.1.48")
