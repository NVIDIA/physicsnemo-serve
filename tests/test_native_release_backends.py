"""Backend selection contracts; no model, Docker daemon or GPU is needed."""

from pathlib import Path
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from test_native_release_orchestration import release, resolved


ROOT = Path(__file__).resolve().parents[1]


def cfd_profile(backend):
    return {
        "backend": backend,
        "provider_profile": "physicsnemo-cfd-surface-transolver-run1-v1",
        "paths": {"repository_root": str(ROOT)},
        "cfd_models": {
            "full": {"preset": "transolver-surface", "points": 60000},
            "tail": {"preset": "transolver-surface", "points": 8095},
        },
    }


def test_cfd_tensorrt_expansion_uses_existing_exact_profile_and_nine_plugins():
    resolved = release.resolve_cfd_profile(cfd_profile("tensorrt"))
    for model in resolved["model_aliases"].values():
        project = model["project_document"]
        assert project["backends"] == ["tensorrt"]
        assert project["tensorrt_profile"] == "provider-default-exact-v1"
        assert len(project["assets"]) == 9
        assert "tensorrt_exact_deslice_bmm_plugin" in project["assets"]


def test_cfd_aoti_expansion_preserves_qualified_project_document():
    resolved = release.resolve_cfd_profile(cfd_profile("aoti"))
    for model in resolved["model_aliases"].values():
        project = model["project_document"]
        assert project["backends"] == ["aoti"]
        assert project["assets"] == {}
        assert "tensorrt_profile" not in project
        assert project["aoti_profile"] == "aten-boundary-exact-v2"


def test_cfd_package_validation_uses_selected_backend(tmp_path, monkeypatch):
    package = tmp_path / "model/backends/tensorrt"
    package.mkdir(parents=True)
    (package / "model.json").write_text("{}")
    selected = []

    def verify(candidate, *, backend, target):
        selected.append(backend)
        return {"package": "backends/tensorrt"}

    monkeypatch.setattr(release, "verify_completed_build", verify)
    release._cfd_package_info(
        {"backend": "tensorrt", "target": {"compilation": {}}},
        {"full": tmp_path},
    )
    assert selected == ["tensorrt"]


def test_tensorrt_stack_fingerprint_binds_runtime_plugins_and_transitive_libraries(
    tmp_path, monkeypatch, capsys
):
    sdk = tmp_path / "sdk"
    torch_root = tmp_path / "torch"
    trt_root = tmp_path / "tensorrt"
    for root in (sdk, torch_root / "lib", trt_root):
        root.mkdir(parents=True)
    plugin = sdk / "libpnmir_tensorrt_exact_linear_plugin.so"
    plugin.write_bytes(b"exact native plugin")
    archive = sdk / "libphysicsnemo_inference_tensorrt.a"
    archive.write_bytes(b"SDK static link input")
    nvinfer = trt_root / "libnvinfer.so.10"
    nvinfer.write_bytes(b"TensorRT runtime")
    dependency = tmp_path / "libcublas.so.13"
    dependency.write_bytes(b"CUDA BLAS dependency")
    # Mock Linux loader output with a rooted path on the fixture's current drive.
    monkeypatch.chdir(tmp_path)
    dependency_loader_path = dependency.as_posix().removeprefix(dependency.drive)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __file__=str(torch_root / "__init__.py"),
            __version__="2.test",
            version=SimpleNamespace(cuda="13.test"),
        ),
    )
    monkeypatch.setitem(sys.modules, "warp", None)
    monkeypatch.setitem(
        sys.modules,
        "tensorrt",
        SimpleNamespace(
            __file__=str(trt_root / "__init__.py"),
            __version__="10.14.1.48",
        ),
    )

    def inspect(argv, **kwargs):
        if argv[0] == "ldconfig":
            return SimpleNamespace(
                returncode=0, stdout=f"libnvinfer.so => {nvinfer}\n", stderr=""
            )
        return SimpleNamespace(
            returncode=0,
            stdout=f"libcublas.so => {dependency_loader_path} (0x00)\nlibcuda.so.1 => not found\n",
            stderr="",
        )

    monkeypatch.setitem(sys.modules, "subprocess", SimpleNamespace(run=inspect))
    probe = release.STACK_PROBE.replace(
        repr("/opt/physicsnemo-inference/lib"), repr(str(sdk))
    )
    monkeypatch.setenv("PNMS_RELEASE_BACKEND", "tensorrt")
    exec(probe, {})
    first = json.loads(capsys.readouterr().out)
    assert first.get("tensorrt_version") == "10.14.1.48"
    assert first["files"][str(plugin)] == release.file_identity(plugin)
    assert first["files"][str(archive)] == release.file_identity(archive)
    assert first["files"][str(nvinfer)] == release.file_identity(nvinfer)
    assert first["files"][str(dependency)] == release.file_identity(dependency)
    dependency.write_bytes(b"changed CUDA BLAS")
    exec(probe, {})
    assert release.content_id(
        json.loads(capsys.readouterr().out)
    ) != release.content_id(first)


def test_cpu_candidate_inspection_rejects_another_backend(tmp_path):
    expected = {
        "source_sha": "a" * 40,
        "workflow": "native",
        "bundle_digest": "b" * 64,
        "backend": "tensorrt",
    }
    runner = SimpleNamespace(
        run=lambda *args, **kwargs: SimpleNamespace(
            stdout=json.dumps({**expected, "backend": "aoti"})
        )
    )
    with pytest.raises(ValueError, match="backend"):
        release.inspect_candidate(
            "registry/native@sha256:" + "c" * 64,
            runner=runner,
            workspace=tmp_path,
            expected=expected,
        )


def test_tensorrt_stack_fingerprint_is_stable_across_python_hash_seeds(tmp_path):
    sdk, torch_root, trt_root = (tmp_path / name for name in ("sdk", "torch", "trt"))
    for root in (sdk, torch_root / "lib", trt_root):
        root.mkdir(parents=True)
    runtime = tmp_path / "libnvinfer.so.10.14.1"
    runtime.write_bytes(b"same TensorRT library behind two loader aliases")
    aliases = [tmp_path / name for name in ("libnvinfer.so", "libnvinfer.so.10")]
    for alias in aliases:
        alias.symlink_to(runtime.name)
    listing = "\n".join(f"{alias.name} => {alias}" for alias in aliases)
    probe = release.STACK_PROBE.replace(
        repr("/opt/physicsnemo-inference/lib"), repr(str(sdk))
    )
    script = f"""
import sys
from types import SimpleNamespace
sys.modules['torch']=SimpleNamespace(__file__={str(torch_root / "__init__.py")!r},__version__='2.test',version=SimpleNamespace(cuda='13.test'))
sys.modules['warp']=None
sys.modules['tensorrt']=SimpleNamespace(__file__={str(trt_root / "__init__.py")!r},__version__='10.14.1.48')
def run(argv,**kwargs):
 return SimpleNamespace(returncode=0,stdout={listing!r} if argv[0]=='ldconfig' else '',stderr='')
sys.modules['subprocess']=SimpleNamespace(run=run)
exec({probe!r})
"""
    fingerprints = set()
    for seed in range(8):
        child = subprocess.run(
            [sys.executable, "-c", script],
            env={
                **os.environ,
                "PYTHONHASHSEED": str(seed),
                "PNMS_RELEASE_BACKEND": "tensorrt",
            },
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        fingerprints.add(release.content_id(json.loads(child.stdout)))
    assert len(fingerprints) == 1, (
        "shared-library aliases must not depend on set iteration order"
    )


def test_workflow_binary_identity_distinguishes_backends_in_the_same_stack():
    profile = resolved()
    sources = [{"path": "workflow.cpp", "sha256": "a" * 64, "size_bytes": 1}]
    aoti = release.workflow_binary_key(profile, sources)
    profile["backend"] = "tensorrt"
    assert release.workflow_binary_key(profile, sources) != aoti


def test_final_http_preflight_records_selected_tensorrt_environment(
    tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(str(ROOT / "native-inference/deploy"))
    import native_release_http_job as http_job

    class PreflightComplete(Exception):
        pass

    selected = []

    def environment(target, backend="aoti"):
        selected.append(backend)
        return {"backend": backend}

    def case(name, function):
        function()
        raise PreflightComplete()

    job = SimpleNamespace(
        summary={},
        stack=None,
        monitor=None,
        root=tmp_path / "out",
        case=case,
        save=lambda: None,
        stack_tail=lambda: "",
    )
    profile = {"source_sha": "a" * 40, "backend": "tensorrt", "target": {}}
    monkeypatch.setenv("SOURCE_SHA", profile["source_sha"])
    monkeypatch.setattr(http_job, "verify_bundle", lambda root: profile)
    monkeypatch.setattr(http_job, "read_json", lambda path: {})
    monkeypatch.setattr(http_job, "ServiceJob", lambda *args: job)
    monkeypatch.setattr(http_job, "HTTP", lambda *args: None)
    monkeypatch.setattr(http_job, "RedisProbe", lambda: None)
    monkeypatch.setattr(http_job, "environment", environment)
    monkeypatch.setattr(http_job, "seal_evidence", lambda *args: None)
    probe = SimpleNamespace(connect_ex=lambda address: 1)

    class Socket:
        def __enter__(self):
            return probe

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(http_job.socket, "socket", Socket)
    args = SimpleNamespace(
        release_root=tmp_path,
        output=job.root,
        image="registry/native@sha256:" + "b" * 64,
    )
    with pytest.raises(PreflightComplete):
        http_job.run(args)
    assert selected == ["tensorrt"]
