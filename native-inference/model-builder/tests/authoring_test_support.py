"""Authoring project files and isolated container execution fixtures."""

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest import mock

from model_builder.build import authoring, cli, inputs
import worker_test_support


class _LinuxSys:
    platform = "linux"

    def __getattr__(self, name):
        return getattr(sys, name)


@contextlib.contextmanager
def linux_container():
    # Preserve the real host platform for pathlib, Torch and Windows controls.
    with (
        mock.patch.object(cli, "sys", _LinuxSys()),
        mock.patch.object(os, "getuid", return_value=1001, create=True),
        mock.patch.object(os, "getgid", return_value=1002, create=True),
    ):
        yield


class AuthoringConfigFixture:
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.path = self.root / "model-build.json"

    def write(self, **changes):
        document = {
            "format_version": 2,
            "name": "my-model",
            "version": "0.1.0",
            "adapter": "build_adapter.py",
            "source": [],
            "checkpoint": None,
            "config": {},
            "assets": {},
            "backends": ["aoti"],
            "executor": "container",
            "builder_image": None,
            "device": "cuda",
            **changes,
        }
        self.path.write_text(json.dumps(document))
        return document


class AuthoringContainerFixture:
    def setUp(self):
        container = linux_container()
        container.__enter__()
        self.addCleanup(container.__exit__, None, None, None)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.project = self.root / "customer project"
        self.project.mkdir()
        (self.project / "build_adapter.py").write_text(
            "def create_model(config, assets): return object()\n"
            "def create_cases(config, assets): return [(1,)]\n"
        )
        (self.project / "weights.pt").write_bytes(
            b"opaque checkpoint for frontend tests"
        )
        self.document = {
            "format_version": 2,
            "name": "customer-model",
            "version": "0.1.0",
            "adapter": "build_adapter.py",
            "source": [],
            "checkpoint": "weights.pt",
            "config": {},
            "assets": {},
            "executor": "container",
            "builder_image": "example/builder@sha256:" + "a" * 64,
            "device": "cuda:1",
            "required_gpu_arch": "sm90",
            "backends": ["aoti"],
        }
        self.write_project()
        self.count = 0
        self.real_run = subprocess.run

    def write_project(self):
        (self.project / "model-build.json").write_text(json.dumps(self.document))

    def invoke(self, behavior, operation="check"):
        self.count += 1
        self.output = self.root / f"output-{self.count}"
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch.object(cli.shutil, "which", return_value="/usr/bin/docker"),
            mock.patch.object(authoring.subprocess, "run", side_effect=behavior) as run,
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            code = cli.main(
                [operation, str(self.project), "--output", str(self.output), "--json"]
            )
        return code, json.loads(stdout.getvalue()), stderr.getvalue(), run

    def successful_process(self, command, *, env, stdout, stderr, mutate=None, code=0):
        mounts = [
            command[index + 1]
            for index, token in enumerate(command)
            if token == "--mount"
        ]
        source_mount = next(value for value in mounts if "dst=/inputs" in value)
        snapshot = Path(source_mount.split("src=", 1)[1].split(",dst=", 1)[0])
        manifest = json.loads((snapshot / "snapshot.json").read_text())
        self.output.mkdir()
        shutil.copytree(snapshot, self.output / "source")
        operation = command[command.index("--operation") + 1]
        report = {
            "format_version": 1,
            "status": "checked" if operation == "check" else "complete",
            "operation": operation,
            "device": "cuda:1",
            "case_count": 1,
            "input_identity": manifest["input_identity"],
            "target_check": {
                "device": "cuda:1",
                "actual_gpu_arch": "sm90",
                "required_gpu_arch": "sm90",
                "gpu_name": "contract-test H100",
            },
            "environment": {"compute_capability": [9, 0]},
            "weights": {"kind": "model_state", "state_sha256": "b" * 64},
            "tensor_contract": {
                "inputs": [{"name": "input_0", "dtype": "float32", "shape": [1]}],
                "outputs": [{"name": "output_0", "dtype": "float32", "shape": [1]}],
            },
        }
        if mutate:
            mutate(report)
        recipe = json.loads((snapshot / "recipe.json").read_text())
        recipe.pop("input_names", None)
        recipe.pop("output_names", None)
        recipe.update(report["tensor_contract"])
        (self.output / "source" / "effective-recipe.json").write_text(
            json.dumps(recipe)
        )
        report.setdefault(
            "source_files",
            [
                {
                    "path": path.relative_to(self.output).as_posix(),
                    **inputs._identity(path),
                }
                for path in sorted((self.output / "source").rglob("*"))
                if path.is_file()
            ],
        )
        (self.output / "check.json").write_text(json.dumps(report))
        stdout.write("mock worker diagnostics\n")
        return subprocess.CompletedProcess(command, code)

    def native_protocol_process(self, command, *, mode="valid", **kwargs):
        """Use existing simulated compiler/runtime fixtures to exercise receipt validation."""
        if "model_builder.build.authoring_worker" not in command:
            return self.real_run(command, **kwargs)
        from model_builder.build import worker

        fixture = worker_test_support.WorkerFixture()
        fixture.addCleanup = self.addCleanup
        if hasattr(fixture, "setUp"):
            fixture.setUp()
        if command[0] == "docker":
            mounts = [
                command[i + 1] for i, token in enumerate(command) if token == "--mount"
            ]
            mount = next(value for value in mounts if "dst=/inputs" in value)
            snapshot = Path(mount.split("src=", 1)[1].split(",dst=", 1)[0])
        else:
            snapshot = Path(command[command.index("--input") + 1])
        manifest = json.loads((snapshot / "snapshot.json").read_text())
        derived = fixture.root / "derived"
        shutil.copytree(snapshot, derived)
        recipe_path = derived / "recipe.json"
        recipe = json.loads(recipe_path.read_text())
        contract = {
            "inputs": [{"name": "input", "dtype": "float32", "shape": [4]}],
            "outputs": [{"name": "output", "dtype": "float32", "shape": [4]}],
        }
        recipe.pop("input_names", None)
        recipe.pop("output_names", None)
        recipe.update(contract)
        recipe_path.write_text(json.dumps(recipe))
        selected = inputs.resolve_inputs(recipe, recipe_path)

        def qualification(output, receipt):
            path = output / "model" / "source-check.json"
            path.write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "passed": True,
                        "scope": "Captured Python source integrity through compilation",
                        "source": manifest["input_identity"]["source"],
                    }
                )
            )
            return {
                "passed": True,
                "report": worker._file_identity(path, output / "model"),
            }

        def backend(*arguments):
            graph = fixture.backend(*arguments)
            return {**graph, "entrypoint": "program.pt2"}

        with (
            mock.patch.object(worker, "_prepare_model", return_value=fixture.prepared),
            mock.patch.object(worker, "_build_backend", side_effect=backend),
        ):
            receipt = worker.execute_build(
                recipe_path,
                self.output,
                ["aoti"],
                "cpu",
                fixture.runtime,
                model_inputs=selected,
                pre_release_check=qualification,
            )
        release_path = self.output / "model" / "model-release.json"
        release = json.loads(release_path.read_text())
        if mode == "missing-qualification":
            receipt.pop("qualification")
            release.pop("qualification")
        elif mode in ("failed-source-check", "wrong-source-identity"):
            check_path = self.output / "model" / "source-check.json"
            source_check = json.loads(check_path.read_text())
            if mode == "failed-source-check":
                source_check["passed"] = False
            else:
                source_check["source"] = {
                    "unrelated.py": {"sha256": "f" * 64, "size_bytes": 1}
                }
            check_path.write_text(json.dumps(source_check))
            receipt["qualification"]["report"] = worker._file_identity(
                check_path, self.output / "model"
            )
            release["qualification"] = receipt["qualification"]
        release_path.write_text(json.dumps(release))
        receipt["release"] = worker._file_identity(release_path, self.output)
        (self.output / "build.json").write_text(json.dumps(receipt))
        report = {
            "format_version": 1,
            "status": "complete",
            "operation": "build",
            "device": "cpu",
            "case_count": receipt["case_count"],
            "input_identity": manifest["input_identity"],
            "target_check": None,
            "environment": receipt["environment"],
            "weights": receipt["weights"],
            "tensor_contract": contract,
            "source_files": worker._inventory(self.output / "source", self.output),
        }
        (self.output / "check.json").write_text(json.dumps(report))
        kwargs["stdout"].write("simulated native protocol fixture completed\n")
        return subprocess.CompletedProcess(command, 0)
