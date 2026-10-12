"""Customer project checks through the real CLI and isolated CPU Torch worker."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

try:
    import torch
except ImportError:
    torch = None


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_builder.build import (
    authoring,
    authoring_config,
    authoring_sources,
    authoring_worker,
    inputs,
)

BUILDER = Path(__file__).resolve().parents[2] / "pnms-model-builder"

MODEL = """import torch

class CustomerModel(torch.nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor([1.0]))
        self.offset = offset

    def forward(self, value):
        from helper import normalize
        return normalize(value) * self.scale + self.offset
"""
ADAPTER = """import torch
from model import CustomerModel

def create_model(config, assets):
    print('customer adapter stdout must not corrupt CLI JSON')
    return CustomerModel(**config)

def create_cases(config, assets):
    return torch.load(assets['validation'], map_location='cpu', weights_only=True)
"""


@unittest.skipIf(torch is None, "requires the producer Torch environment")
class AuthoringIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.project = self.root / "customer-model"
        self.project.mkdir()
        (self.project / "model.py").write_text(MODEL)
        (self.project / "helper.py").write_text(
            "def normalize(value):\n    return value - 1.0\n"
        )
        (self.project / "build_adapter.py").write_text(ADAPTER)
        torch.save({"scale": torch.tensor([3.0])}, self.project / "weights.pt")
        torch.save(
            [(torch.tensor([[1.0, 2.0, 3.0]]),), (torch.tensor([[4.0, 5.0, 6.0]]),)],
            self.project / "validation.pt",
        )
        self.document = {
            "format_version": 2,
            "name": "customer-model",
            "version": "0.1.0",
            "adapter": "build_adapter.py",
            "source": ["model.py", "helper.py"],
            "checkpoint": "weights.pt",
            "config": {"offset": 2.0},
            "assets": {"validation": "validation.pt"},
            "executor": "local",
            "device": "cpu",
            "backends": ["aoti"],
        }
        self.write_config()

    def write_config(self):
        (self.project / "model-build.json").write_text(json.dumps(self.document))

    def invoke(self, operation, output=None, *, cwd=None, pythonpath=None):
        command = [sys.executable, str(BUILDER), operation, str(self.project), "--json"]
        if output is not None:
            command += ["--output", str(output)]
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        if pythonpath is not None:
            environment["PYTHONPATH"] = pythonpath
        environment["PATH"] = os.pathsep.join(
            (str(Path(sys.executable).parent), "/usr/bin", "/bin")
        )
        completed = subprocess.run(
            command,
            cwd=self.root if cwd is None else cwd,
            env=environment,
            text=True,
            capture_output=True,
        )
        try:
            value = json.loads(completed.stdout)
        except ValueError as exc:
            self.fail(
                f"CLI stdout must be one JSON document: {completed.stdout}\n{completed.stderr}\n{exc}"
            )
        return completed, value

    def test_check_loads_captured_modules_and_weights_without_native_sdk(self):
        output = self.root / "check-output"
        completed, result = self.invoke("check", output)
        self.assertEqual(completed.returncode, 0, f"{result}\n{completed.stderr}")
        self.assertEqual(result["status"], "checked")
        self.assertEqual(result["case_count"], 2)
        self.assertEqual(
            result["tensor_contract"],
            {
                "inputs": [{"name": "input_0", "dtype": "float32", "shape": [1, 3]}],
                "outputs": [{"name": "output_0", "dtype": "float32", "shape": [1, 3]}],
            },
        )
        self.assertFalse((self.project / "model-build.lock.json").exists())
        self.assertFalse((output / "model").exists())
        report = json.loads((output / "check.json").read_text())
        self.assertEqual(report["status"], "checked")
        self.assertEqual(report["case_count"], 2)
        self.assertEqual(
            report["input_identity"]["inputs"]["checkpoint"]["sha256"],
            hashlib.sha256((self.project / "weights.pt").read_bytes()).hexdigest(),
        )
        source_names = set(report["input_identity"]["source"])
        self.assertEqual(source_names, {"build_adapter.py", "model.py", "helper.py"})
        self.assertIn("customer adapter stdout", (output / "execution.log").read_text())
        self.assertEqual(
            json.loads((output / "execution.json").read_text())["exit_code"], 0
        )

    def test_invalid_checkpoint_has_actionable_failure_and_retained_report(self):
        torch.save({"unexpected": torch.tensor([3.0])}, self.project / "weights.pt")
        output = self.root / "invalid-checkpoint"
        completed, result = self.invoke("check", output)
        self.assertEqual(completed.returncode, 1, result)
        self.assertEqual(result["status"], "failed")
        self.assertIn("checkpoint keys", result["diagnostics"][0]["message"])
        report = json.loads((output / "check.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertIn("checkpoint keys", report["error"]["message"])
        self.assertFalse((self.project / "model-build.lock.json").exists())
        self.assertFalse((output / "model").exists())

    def test_missing_captured_dependency_is_reported_instead_of_using_project_cwd(self):
        self.document["source"] = ["model.py"]
        self.write_config()
        completed, result = self.invoke("check", self.root / "missing-helper")
        self.assertEqual(completed.returncode, 1, result)
        self.assertIn("helper", result["diagnostics"][0]["message"])
        self.assertIn("ModuleNotFoundError", completed.stderr)

    def test_check_rejects_installed_package_shadowing_captured_namespace(self):
        package = self.project / "customer_namespace"
        package.mkdir()
        model = (
            "import torch\nVALUE = {value}\n"
            "class Model(torch.nn.Module):\n"
            "    def forward(self, x): return x * VALUE\n"
        )
        (package / "model.py").write_text(model.format(value=2))
        environment = self.root / "environment"
        installed = environment / "customer_namespace"
        installed.mkdir(parents=True)
        (installed / "__init__.py").write_text("")
        (installed / "model.py").write_text(model.format(value=99))
        (self.project / "build_adapter.py").write_text(
            "import torch\nfrom customer_namespace.model import Model\n"
            "def create_model(config, assets): return Model()\n"
            "def create_cases(config, assets): return [(torch.ones(1),)]\n"
        )
        torch.save({}, self.project / "weights.pt")
        self.document.update(source=["customer_namespace"], config={})
        self.write_config()
        snapshot = self.snapshot()
        output = self.root / "namespace-shadow-check"
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from model_builder.build import authoring_worker; "
                f"sys.path.append({str(environment)!r}); "
                f"authoring_worker.execute({str(snapshot)!r}, {str(output)!r}, 'check', 'cpu')",
            ],
            cwd=self.root,
            env=dict(os.environ, PYTHONPATH=str(BUILDER.parent / "model-builder/src")),
            text=True,
            capture_output=True,
        )
        report = json.loads((output / "check.json").read_text())
        self.assertNotEqual(completed.returncode, 0, report)
        self.assertEqual(report["status"], "failed")
        self.assertIn("import conflict", report["error"]["message"])
        self.assertIn("customer_namespace", report["error"]["message"])
        self.assertFalse((output / "model").exists())
        self.assertFalse((self.project / "model-build.lock.json").exists())

    def caller_import_paths(self):
        return (
            ("project-cwd", self.project, None),
            ("relative-pythonpath", self.root, self.project.name),
            ("absolute-pythonpath", self.root, str(self.project)),
        )

    def test_check_rejects_uncaptured_lazy_import_from_caller_paths(self):
        self.document["source"] = ["model.py"]
        self.write_config()
        for label, cwd, pythonpath in self.caller_import_paths():
            with self.subTest(caller_path=label):
                output = self.root / label
                completed, result = self.invoke(
                    "check", output, cwd=cwd, pythonpath=pythonpath
                )
                self.assertEqual(completed.returncode, 1, result)
                self.assertEqual(result["status"], "failed")
                self.assertIn("helper", result["diagnostics"][0]["message"])
                self.assertIn("ModuleNotFoundError", completed.stderr)
                report = json.loads((output / "check.json").read_text())
                self.assertEqual(report["status"], "failed")
                self.assertNotIn("helper.py", report["input_identity"]["source"])

    def test_check_loads_captured_lazy_import_with_caller_paths(self):
        for label, cwd, pythonpath in self.caller_import_paths():
            with self.subTest(caller_path=label):
                output = self.root / label
                completed, result = self.invoke(
                    "check", output, cwd=cwd, pythonpath=pythonpath
                )
                self.assertEqual(
                    completed.returncode, 0, f"{result}\n{completed.stderr}"
                )
                self.assertEqual(result["status"], "checked")
                self.assertEqual(result["case_count"], 2)
                report = json.loads((output / "check.json").read_text())
                self.assertIn("helper.py", report["input_identity"]["source"])

    def test_build_requires_sdk_even_though_check_does_not(self):
        completed, result = self.invoke("build", self.root / "no-sdk")
        self.assertEqual(completed.returncode, 2, result)
        self.assertTrue(
            any(item["code"] == "MISSING_RUNTIME" for item in result["diagnostics"]),
            result,
        )
        self.assertFalse((self.root / "no-sdk").exists())

    def snapshot(self):
        project = authoring_config.load(self.project)
        source = authoring_sources.capture(
            self.project, self.document["source"], self.document["adapter"]
        )
        selected = {
            name: {
                "path": str(self.project / path),
                **inputs._identity(self.project / path),
            }
            for name, path in {
                "checkpoint": "weights.pt",
                "validation": "validation.pt",
            }.items()
        }
        configuration = self.document["config"]
        plan = {
            "device": "cpu",
            "backends": ["aoti"],
            "executor": "local",
            "toolchain_lock": {"sha256": "f" * 64},
        }
        identity = authoring._identity(project, source, selected, configuration, plan)
        snapshot = self.root / "snapshot"
        authoring._snapshot(
            snapshot, project, source, selected, configuration, identity
        )
        return snapshot

    def test_target_failure_retains_actionable_check_report(self):
        snapshot = self.snapshot()
        output = self.root / "target-failed"
        with mock.patch.object(
            authoring_worker.targets,
            "check_target",
            side_effect=ValueError(
                "required GPU architecture sm90 but selected device has sm80"
            ),
        ):
            with self.assertRaisesRegex(ValueError, "sm90"):
                authoring_worker.execute(
                    snapshot, output, "check", "cuda", required_gpu_arch="sm90"
                )
        self.assertTrue(
            (output / "check.json").is_file(),
            "GPU qualification failure must retain an actionable report",
        )
        report = json.loads((output / "check.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertIn("sm90", report["error"]["message"])

    def test_build_summary_describes_cases_and_weights_used_by_native_harness(self):
        snapshot = self.snapshot()
        output = self.root / "build-summary"
        actual_weights = {"kind": "model_state", "state_sha256": "a" * 64}
        actual_environment = {"torch_version": "native-harness-environment"}

        def compiled_build(recipe, destination, backends, device, runtime, **options):
            destination.mkdir()
            (destination / "source").mkdir()
            (destination / "source/recipe.json").write_bytes(recipe.read_bytes())
            return {
                "status": "complete",
                "case_count": 3,
                "weights": actual_weights,
                "environment": actual_environment,
            }

        with mock.patch.object(
            authoring_worker.worker, "execute_build", side_effect=compiled_build
        ):
            report = authoring_worker.execute(snapshot, output, "build", "cpu")
        self.assertEqual(report["status"], "complete")
        self.assertEqual(
            report["case_count"],
            3,
            "Report coverage must describe native harness execution, not initial shape inference",
        )
        self.assertEqual(report["weights"], actual_weights)
        self.assertEqual(report["environment"], actual_environment)


if __name__ == "__main__":
    unittest.main()
