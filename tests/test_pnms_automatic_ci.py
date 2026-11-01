"""CPU checks for source-bound automatic native QA; no Docker/Lepton calls."""

from __future__ import annotations
from fnmatch import fnmatchcase
import json
import os
from pathlib import Path
import subprocess
import tarfile
import pytest
import yaml
from scripts import prepare_pnms_qa_context as capture

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def source(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()

    def git(*args):
        return (
            subprocess.check_output(["git", "-C", str(repository), *args])
            .decode()
            .strip()
        )

    git("init", "-q")
    git("config", "user.email", "qa@example.invalid")
    git("config", "user.name", "QA")
    for relative in (
        "pnms",
        "qa/native_inference/Dockerfile.pnms",
        "qa/native_inference/venv-requirements.txt",
        "native-inference/model-builder/src/model_builder/build/cli.py",
        "native-inference/model-builder/images/Dockerfile.builder",
        "native-inference/model-builder/images/Dockerfile.builder.dockerignore",
    ):
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# " + relative + "\n")
    git("add", ".")
    git("commit", "-qm", "fixture")
    return repository, git("rev-parse", "HEAD")


def test_capture_retains_legitimate_model_builder_build_source(source, tmp_path):
    repository, revision = source
    out = tmp_path / "contexts"
    metadata = capture.prepare(repository, out, expected_source_sha=revision)
    assert metadata["source_sha"] == revision and metadata["source_dirty"] is False
    relative = "native-inference/model-builder/src/model_builder/build/cli.py"
    records = json.loads((out / "dev-context/source-inventory.json").read_text())
    assert relative in records
    assert (out / "builder-context" / relative).read_bytes() == (
        repository / relative
    ).read_bytes()
    with tarfile.open(out / "dev-context/source.tar") as archive:
        assert (
            archive.extractfile(relative).read() == (repository / relative).read_bytes()
        )


@pytest.mark.parametrize(
    "relative",
    [
        ".env",
        "deploy/config.yaml",
        ".aws/credentials",
        "native-inference/model-builder/build/cache.py",
        "native-inference/model-builder/src/model_builder/build/__pycache__/cli.pyc",
    ],
)
def test_capture_excludes_secrets_and_generated_source_even_if_git_tracked(
    source, tmp_path, relative
):
    repository, revision = source
    private = repository / relative
    private.parent.mkdir(parents=True, exist_ok=True)
    private.write_text("must-not-leave-checkout")
    subprocess.check_call(["git", "-C", str(repository), "add", "-f", relative])
    out = tmp_path / "contexts"
    capture.prepare(repository, out, expected_source_sha=revision, allow_dirty=True)
    records = json.loads((out / "dev-context/source-inventory.json").read_text())
    assert relative not in records
    with tarfile.open(out / "dev-context/source.tar") as archive:
        assert relative not in archive.getnames()
    assert not (out / "builder-context" / relative).exists()


def test_capture_rejects_symlink_source_instead_of_dereferencing(source, tmp_path):
    repository, revision = source
    (repository / "native-inference/leak").symlink_to("/etc/hosts")
    with pytest.raises(ValueError, match="symlink"):
        capture.prepare(
            repository,
            tmp_path / "contexts",
            expected_source_sha=revision,
            allow_dirty=True,
        )


def test_capture_detects_source_change_during_copy(source, tmp_path, monkeypatch):
    repository, revision = source
    target = (
        repository / "native-inference/model-builder/src/model_builder/build/cli.py"
    )
    original = Path.read_bytes
    calls = 0

    def changed(path):
        nonlocal calls
        data = original(path)
        if path == target:
            calls += 1
            if calls == 2:
                return data + b"changed-during-capture"
        return data

    monkeypatch.setattr(Path, "read_bytes", changed)
    with pytest.raises(ValueError, match="changed|identity"):
        capture.prepare(repository, tmp_path / "contexts", expected_source_sha=revision)


def test_existing_lepton_profiles_are_reusable_with_required_source_bound_images():
    workflow = yaml.load(
        (ROOT / ".github/workflows/lepton_pnms_qa.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    called = workflow["on"].get("workflow_call", {})
    for field in ("image", "source_sha", "builder_image"):
        assert called.get("inputs", {}).get(field, {}).get("required") == "true"
    document = (ROOT / ".github/workflows/lepton_pnms_qa.yml").read_text()
    assert "workflow_dispatch" in document
    assert "run_lepton_model_builder_image_qa.py" in document
    assert "run_lepton_pnms_venv_qa.py" in document


def automatic():
    return yaml.load(
        (ROOT / ".github/workflows/lepton_pnms_native_ci.yml").read_text(),
        Loader=yaml.BaseLoader,
    )


def test_automatic_policy_filters_native_changes_and_reuses_required_cross_backend_profiles():
    flow = automatic()
    assert set(flow["on"]) == {"push"}
    assert flow["on"]["push"]["branches"] == ["main", "pull-request/[0-9]+"]
    for trigger in ("push",):
        paths = flow["on"][trigger]["paths"]
        for path in (
            "native-inference/**",
            "application-cli/**",
            "crates/native-worker/**",
            "crates/worker-runtime/**",
            "qa/native_inference/**",
        ):
            assert path in paths
        assert "plugins/**" not in paths and "python/**" not in paths
    qualify = flow["jobs"]["qualify"]
    assert qualify["uses"] == "./.github/workflows/lepton_pnms_qa.yml"
    for field in ("image", "builder_image", "source_sha"):
        assert qualify["with"][field] == "${{ needs.prepare.outputs." + field + " }}"
    assert qualify["secrets"] == "inherit"
    assert "if" not in qualify and "continue-on-error" not in qualify
    assert flow["concurrency"]["cancel-in-progress"] == "false"


@pytest.mark.parametrize(
    "changed_path",
    [
        "qa/scripts/benchmark_pnms_cfd.py",
        "tests/test_pnms_cfd_benchmark.py",
        "tests/test_pnms_release_fixtures_qa.py",
        "tests/test_model_builder_image_qa.py",
        "scripts/run_lepton_model_builder_image_qa.py",
        "scripts/native_workflow.py",
        "scripts/plugin_runtime.py",
    ],
)
@pytest.mark.parametrize("trigger", ["push"])
def test_qualification_changes_trigger_native_ci(changed_path, trigger):
    patterns = automatic()["on"][trigger]["paths"]
    assert any(fnmatchcase(changed_path, pattern) for pattern in patterns), (
        f"{changed_path} must trigger native acceptance on {trigger}"
    )


def test_automatic_ci_requires_file_fixture_release_images_for_both_backends():
    flow = automatic()
    release = flow["jobs"].get("release", {})
    assert release.get("needs") == "prepare"
    assert release.get("uses") == "./.github/workflows/lepton_pnms_release.yml"
    assert release.get("strategy", {}).get("matrix", {}).get("backend") == [
        "aoti",
        "tensorrt",
    ]
    assert release["strategy"]["fail-fast"] == "false"
    selected = release.get("with", {})
    assert (
        selected.get("image")
        == selected.get("runtime_image")
        == ("${{ needs.prepare.outputs.image }}")
    )
    for field in ("source_sha", "repository"):
        assert selected.get(field) == "${{ needs.prepare.outputs." + field + " }}"
    assert selected.get("backend") == "${{ matrix.backend }}"
    assert selected.get("template") == "tensor"
    assert selected.get("project") == (
        "qa/native_inference/projects/tensor-files-${{ matrix.backend }}"
    )
    assert release.get("secrets") == "inherit"
    assert "if" not in release and "continue-on-error" not in release
    assert flow["jobs"]["prepare"]["outputs"].get("repository") == (
        "${{ steps.images.outputs.repository }}"
    )


def test_automatic_digest_outputs_use_nonsecret_repository_configuration():
    flow = automatic()
    assert flow["jobs"]["prepare"]["env"]["PNMS_QA_REPOSITORY"] == (
        "${{ vars.PNMS_QA_REPOSITORY }}"
    )
    assert "secrets.NGC_ORG" not in json.dumps(flow)
    assert "secrets.NGC_IMAGE" not in json.dumps(flow)


def ci_step(needle):
    return next(
        step
        for step in automatic()["jobs"]["prepare"]["steps"]
        if needle in step.get("name", "")
    )


def invoke_ci_step(tmp_path, needle, *, env=None):
    environment = os.environ | {
        "PNMS_QA_SOURCE_SHA": "a" * 40,
        "PNMS_QA_BASE_IMAGE": "registry/base@sha256:" + "b" * 64,
        "PNMS_QA_REPOSITORY": "registry/org/app",
        "NGC_REGISTRY": "registry",
        "NGC_PASSWORD": "secret",
        "LEPTON_WORKSPACE_TOKEN": "token",
        "LEPTON_WORKSPACE_ID": "workspace",
        "LEPTON_NODE_GROUP": "gpu",
        "LEPTON_PULL_SECRET": "pull",
        "NFS_MOUNT_BASE": "/shared",
        "RUNNER_TEMP": str(tmp_path),
        "PNMS_QA_RUN": "native-ci-1-1",
        "GITHUB_OUTPUT": str(tmp_path / "outputs"),
        **(env or {}),
    }
    return subprocess.run(
        ["bash", "-c", ci_step(needle)["run"]],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "field",
    [
        "PNMS_QA_BASE_IMAGE",
        "PNMS_QA_REPOSITORY",
        "NGC_PASSWORD",
        "LEPTON_WORKSPACE_TOKEN",
        "LEPTON_NODE_GROUP",
        "LEPTON_PULL_SECRET",
    ],
)
def test_missing_native_qa_configuration_fails_instead_of_skipping_gpu_gate(
    tmp_path, field
):
    completed = invoke_ci_step(tmp_path, "Validate required", env={field: ""})
    assert completed.returncode != 0
    assert field in completed.stderr


@pytest.mark.parametrize("trusted", ["true", "false"])
def test_approved_mirror_policy_never_exposes_secrets_on_untrusted_runner(
    tmp_path, trusted
):
    guard = automatic()["jobs"]["trust"]
    assert guard["runs-on"] == "ubuntu-latest"
    assert automatic()["jobs"]["prepare"]["needs"] == "trust"
    assert set(automatic()["on"]) == {"push"}
    step = guard["steps"][0]
    assert step["env"]["TRUSTED_SOURCE"] == (
        "${{ github.event_name == 'push' && (github.ref == 'refs/heads/main' || "
        "startsWith(github.ref, 'refs/heads/pull-request/')) }}"
    )
    completed = subprocess.run(
        ["bash", "-c", step["run"]],
        cwd=tmp_path,
        env=os.environ | {"TRUSTED_SOURCE": trusted},
        capture_output=True,
        text=True,
    )
    assert (completed.returncode == 0) == (trusted == "true")


@pytest.mark.parametrize("bad_metadata", ["", "tag-only", "sha256:wrong"])
def test_image_build_metadata_must_publish_exact_digests_before_gpu_dispatch(
    tmp_path, bad_metadata
):
    contexts = tmp_path / "pnms-contexts"
    (contexts / "dev-context").mkdir(parents=True)
    (contexts / "dev-context/source.json").write_text(
        json.dumps({"source_sha": "a" * 40, "source_dirty": False})
    )
    docker = tmp_path / "docker"
    docker.write_text(
        '#!/usr/bin/env python3\nimport json,os,pathlib,sys\na=sys.argv[1:]; p=pathlib.Path(a[a.index("--metadata-file")+1]);p.write_text(json.dumps({"containerimage.digest":os.environ["DIGEST"]})); pathlib.Path(os.environ["CAPTURE"]).open("a").write(json.dumps(a)+"\\n")\n'
    )
    docker.chmod(0o755)
    result = invoke_ci_step(
        tmp_path,
        "Build and publish",
        env={
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "DIGEST": bad_metadata,
            "CAPTURE": str(tmp_path / "docker-argv"),
        },
    )
    assert result.returncode != 0
    assert not (tmp_path / "outputs").exists()


def test_build_commands_use_same_source_context_normal_builder_and_returned_digests(
    tmp_path,
):
    contexts = tmp_path / "pnms-contexts"
    (contexts / "dev-context").mkdir(parents=True)
    (contexts / "dev-context/source.json").write_text(
        json.dumps({"source_sha": "a" * 40, "source_dirty": False})
    )
    docker = tmp_path / "docker"
    docker.write_text(
        '#!/usr/bin/env python3\nimport json,os,pathlib,sys\na=sys.argv[1:]; p=pathlib.Path(a[a.index("--metadata-file")+1]);p.write_text(json.dumps({"containerimage.digest":"sha256:"+"c"*64})); pathlib.Path(os.environ["CAPTURE"]).open("a").write(json.dumps(a)+"\\n")\n'
    )
    docker.chmod(0o755)
    result = invoke_ci_step(
        tmp_path,
        "Build and publish",
        env={
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "CAPTURE": str(tmp_path / "docker-argv"),
        },
    )
    assert result.returncode == 0, result.stderr
    dev, builder = [
        json.loads(line) for line in (tmp_path / "docker-argv").read_text().splitlines()
    ]
    assert dev[-1] == str(contexts / "dev-context") and builder[-1] == str(
        contexts / "builder-context"
    )
    assert (
        "SOURCE_SHA=" + "a" * 40 in dev
        and "PNMS_QA_BASE_IMAGE=registry/base@sha256:" + "b" * 64 in dev
    )
    assert builder[builder.index("--target") + 1] == "default-builder"
    assert "PNMIR_ENABLE_TENSORRT=ON" in builder
    assert "--push" in dev and "--push" in builder
    outputs = (tmp_path / "outputs").read_text()
    assert "image=registry/org/app@sha256:" + "c" * 64 in outputs
    assert "builder_image=registry/org/app@sha256:" + "c" * 64 in outputs
    assert "repository=registry/org/app\n" in outputs
