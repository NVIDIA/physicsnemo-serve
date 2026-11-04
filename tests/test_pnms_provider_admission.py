"""CPU launch gates for native libraries changed in place after a build."""

from copy import deepcopy
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "application-cli/src"))
from pnms_app import admission, build, cli, compatibility, native, service  # noqa: E402
from pnms_app.common import file_identity, write_json  # noqa: E402
from test_pnms_python_cli import executable as python_executable  # noqa: E402


@pytest.fixture
def providers(tmp_path, monkeypatch):
    app = tmp_path / "app"
    location = app / ".pnms/builds/complete/bundle"
    library = location / "workflow/lib/libphysicsnemo_workflow.so"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"sealed workflow")
    provider = tmp_path / "selected-venv/lib/libtorch.so"
    provider.parent.mkdir(parents=True)
    provider.write_bytes(b"original provider")
    fingerprint = {
        "schema_version": 1,
        "system": "Linux",
        "machine": "x86_64",
        "libc": ["glibc", "2.39"],
        "torch": "2.10.0",
        "cuda": "13.1",
        "cxx11_abi": True,
        "tensorrt": None,
        "torch_libraries": {"lib/libtorch.so": file_identity(provider)},
        "system_libraries": {"libstdc++.so.6": {"sha256": "b" * 64, "size_bytes": 3}},
        "driver_libraries": [],
        "workflow_binary": file_identity(library),
    }
    project = {"root": app, "runtime": "cpp", "build": {}}
    built = {
        "root": str(location.parent),
        "bundle": "bundle",
        "environment": {"kind": "local", "python": "/selected-venv/bin/python"},
        "workflow": {"compatibility": fingerprint},
        "loader_environment": {"LD_LIBRARY_PATH": str(provider.parent)},
    }
    captures = []

    def capture(python, selected, *, environment, output):
        assert Path(python) == Path(built["environment"]["python"])
        assert selected == library
        loader = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
        assert environment[loader].split(os.pathsep)[:2] == [
            str(library.parent),
            str(location / "workflow/lib64"),
        ]
        assert str(provider.parent) in environment["LD_LIBRARY_PATH"].split(os.pathsep)
        assert not Path(output).resolve().is_relative_to(location.resolve())
        observed = deepcopy(fingerprint)
        observed["torch_libraries"]["lib/libtorch.so"] = file_identity(provider)
        write_json(output, observed)
        captures.append(output)
        return observed

    monkeypatch.setattr(compatibility, "capture", capture)
    return project, built, provider, captures


@pytest.mark.parametrize("changed", [False, True])
def test_direct_run_checks_provider_bytes_before_execution(
    providers, tmp_path, monkeypatch, changed
):
    project, built, provider, captures = providers
    if changed:
        provider.write_bytes(b"different provider with the same reported version")
    executed = []
    monkeypatch.setattr(cli, "load_project", lambda *args, **kwargs: project)
    monkeypatch.setattr(build, "last_build", lambda *args, **kwargs: built)
    monkeypatch.setattr(
        admission,
        "local_environment",
        lambda *args: {**built["loader_environment"], "CUDA_VISIBLE_DEVICES": "0"},
    )
    monkeypatch.setattr(
        native,
        "run_bundle",
        lambda *args, **kwargs: executed.append(kwargs) or {"status": "ran"},
    )
    args = cli.parser().parse_args(
        ["run", str(project["root"]), "--input", str(tmp_path / "request.json")]
    )
    if changed:
        with pytest.raises(ValueError, match="provider.*pnms build"):
            cli.execute(args)
        assert not executed
    else:
        assert cli.execute(args)["status"] == "ran"
        assert len(executed) == 1
    assert len(captures) == 1


def host_plan(tmp_path, built, *, in_image=False):
    return {
        "kind": "host",
        "in_image": in_image,
        "output": str(tmp_path / "serve"),
        "environment": {**built["loader_environment"], "CUDA_VISIBLE_DEVICES": "0"},
        "binaries": [],
    }


@pytest.mark.parametrize("changed", [False, True])
def test_host_service_checks_provider_bytes_before_starting_processes(
    providers, tmp_path, monkeypatch, changed
):
    project, built, provider, captures = providers
    if changed:
        provider.write_bytes(b"different provider with the same reported version")
    started = []
    monkeypatch.setattr(
        service, "plan_host", lambda *args, **kwargs: host_plan(tmp_path, built)
    )
    monkeypatch.setattr(service.shutil, "which", lambda name: "/installed/" + name)
    monkeypatch.setattr(
        service,
        "_supervise",
        lambda plan: started.append(plan) or {"status": "started"},
    )
    options = dict(
        repository=ROOT, output=tmp_path / "serve", dev=True, skip_build=True
    )
    if changed:
        with pytest.raises(ValueError, match="provider.*pnms build"):
            service.serve(project, built, **options)
        assert not started
    else:
        assert service.serve(project, built, **options)["status"] == "started"
        assert len(started) == 1
    assert len(captures) == 1


@pytest.mark.parametrize("mode", ["dry_run", "image"])
def test_planning_and_image_startup_do_not_require_a_host_provider_receipt(
    providers, tmp_path, monkeypatch, mode
):
    project, built, _, captures = providers
    del built["workflow"]
    monkeypatch.setattr(
        service,
        "plan_host",
        lambda *args, **kwargs: host_plan(tmp_path, built, in_image=mode == "image"),
    )
    monkeypatch.setattr(service.shutil, "which", lambda name: "/installed/" + name)
    monkeypatch.setattr(service, "_supervise", lambda plan: {"status": "started"})
    result = service.serve(
        project,
        built,
        repository=ROOT,
        output=tmp_path / "serve",
        dev=True,
        skip_build=True,
        dry_run=mode == "dry_run",
    )
    assert result["status"] == ("planned" if mode == "dry_run" else "started")
    assert not captures


def test_host_launch_requires_a_recorded_provider_fingerprint(providers, tmp_path):
    _, built, _, captures = providers
    del built["workflow"]
    with pytest.raises(ValueError, match="provider.*pnms build"):
        admission.verify_providers(
            built,
            environment=built["loader_environment"],
            output=tmp_path / "provider.json",
        )
    assert not captures


def test_capture_honors_selected_loader_order_for_duplicate_library_names(
    tmp_path, monkeypatch
):
    package = tmp_path / "python/torch"
    torch_library = package / "lib/libprovider.so"
    torch_library.parent.mkdir(parents=True)
    torch_library.write_bytes(b"Torch provider")
    (package / "__init__.py").write_text(
        "from types import SimpleNamespace\n"
        "__version__='2.10.0'\n"
        "version=SimpleNamespace(cuda='13.1')\n"
        "_C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True)\n"
    )
    selected_library = tmp_path / "selected-sdk/libprovider.so"
    selected_library.parent.mkdir()
    selected_library.write_bytes(b"selected provider with the same SONAME")
    workflow = tmp_path / "bundle/libworkflow.so"
    workflow.parent.mkdir()
    workflow.write_bytes(b"sealed workflow")
    ldd = tmp_path / "bin/ldd"
    ldd.parent.mkdir()
    ldd = python_executable(
        ldd,
        "import os\nfrom pathlib import Path\n"
        "paths=(Path(part)/'libprovider.so' for part in os.environ['LD_LIBRARY_PATH'].split(os.pathsep) if part)\n"
        "selected=next(path for path in paths if path.is_file())\n"
        "print('libprovider.so => '+str(selected)+' (0x1234)')\n",
    )
    # Exercise the Linux provider closure on CPU-only macOS test hosts too.
    monkeypatch.setattr(
        compatibility,
        "PROBE",
        "import platform; platform.system=lambda: 'Linux'\n" + compatibility.PROBE,
    )
    environment = {
        **os.environ,
        "PYTHONPATH": str(package.parent),
        "PATH": str(ldd.parent) + os.pathsep + os.environ.get("PATH", ""),
        "LD_LIBRARY_PATH": os.pathsep.join(
            [str(selected_library.parent), str(torch_library.parent)]
        ),
    }
    observed = compatibility.capture(
        Path(sys.executable),
        workflow,
        environment=environment,
        output=tmp_path / "providers.json",
    )
    assert observed["system_libraries"]["libprovider.so"] == file_identity(
        selected_library
    )
