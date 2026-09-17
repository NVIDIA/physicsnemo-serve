"""Workflow qualification behavior using an explicit tiny producer/native double."""

import copy
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "model-builder/tests"))
sys.path.insert(0, str(ROOT / "model-builder/src"))

import test_worker as worker_tests  # noqa: E402
from pnmir_build import worker  # noqa: E402
import qualification as workflows  # noqa: E402


MODEL = "geotransolver-surface-core"


class WorkflowTest(unittest.TestCase):
    def test_request_accepts_either_or_both_backends_and_rejects_invalid_selection(
        self,
    ):
        def request(backends, device="cuda", operation="build"):
            return workflows._request(
                operation, MODEL, self.sha, self.hashes, device, 32, 64, backends
            )

        for backends in (
            ["aoti"],
            ["tensorrt"],
            ["aoti", "tensorrt"],
            ["tensorrt", "aoti"],
        ):
            self.assertEqual(request(backends)["backends"], backends)
        for backends in ([], ["aoti", "aoti"], ["unknown"], "aoti", [None]):
            with (
                self.subTest(backends=backends),
                self.assertRaisesRegex(ValueError, "backend"),
            ):
                request(backends)
        with self.assertRaisesRegex(ValueError, "TensorRT requires CUDA"):
            request(["tensorrt"], device="cpu")
        self.assertEqual(request([], operation="prepare")["backends"], [])
        with self.assertRaisesRegex(ValueError, "backend"):
            request(["aoti"], operation="prepare")

    def setUp(self):
        self.fixture = worker_tests.WorkerTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        self.root = f.root.resolve()
        self.output = self.root / "workflow"
        self.archive = self.root / "model.mdlus"
        self.archive.write_bytes(b"explicit toy checkpoint, not PhysicsNeMo weights")
        self.sha = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        self.producer = self.root / "producer"
        self.producer.mkdir()
        for name in ("prepare.py", "adapter.py"):
            (self.producer / name).write_text("# explicit toy producer source\n")
        self.hashes = {
            name: hashlib.sha256((self.producer / name).read_bytes()).hexdigest()
            for name in ("prepare.py", "adapter.py")
        }
        self.wrong_reference = False
        for key in ("cases", "inputs", "references"):
            f.prepared[key].append(f.prepared[key][0])

    def producer_prepare(
        self, checkpoint, output, *, points, geometry_points, device, adapter_path=None
    ):
        output = Path(output)
        self.assertEqual(adapter_path, output.parent / "producer/adapter.py")
        output.mkdir(parents=True)
        (output / "config.json").write_text('{"test_double":true}\n')
        (output / "checkpoint.pt").write_bytes(b"plain toy state")
        (output / "fixtures.pt").write_bytes(b"toy fixture identity")
        (output / "adapter.py").write_bytes((self.producer / "adapter.py").read_bytes())
        recipe = copy.deepcopy(self.fixture.recipe)
        recipe.update(
            format_version=2,
            name=MODEL,
            supported_backends=["aoti"],
            adapter="adapter.py",
        )
        recipe["inputs"] = [{"name": "input", "dtype": "float32", "shape": [4]}]
        recipe["outputs"] = [{"name": "output", "dtype": "float32", "shape": [4]}]
        for key in ("input_names", "output_names", "dtype", "shape"):
            recipe.pop(key)
        for key in ("config", "checkpoint"):
            name = "config.json" if key == "config" else "checkpoint.pt"
            recipe[key] = {"path": name, "sha256": worker._sha256(output / name)}
        recipe["checkpoint"]["format"] = "torch-state-dict"
        recipe["assets"] = {
            "fixtures": {
                "path": "fixtures.pt",
                "sha256": worker._sha256(output / "fixtures.pt"),
            }
        }
        worker._write_json(output / "recipe.json", recipe)
        report = {
            "format_version": 1,
            "status": "complete",
            "point_count": points,
            "geometry_point_count": geometry_points,
            "environment": {"device": device},
            "checkpoint": worker._file_identity(Path(checkpoint)),
            "source": {
                "preparation": worker._file_identity(
                    output.parent / "producer/prepare.py"
                ),
                "adapter": worker._file_identity(output.parent / "producer/adapter.py"),
            },
            "model_state_sha256": self.fixture.prepared["weights"]["state_sha256"],
            "comparisons": [
                {"case": i, "passed": True, "bitwise_equal": True, "max_abs": 0.0}
                for i in range(3)
            ],
            "files": [
                worker._file_identity(output / name, output)
                for name in (
                    "recipe.json",
                    "adapter.py",
                    "config.json",
                    "checkpoint.pt",
                    "fixtures.pt",
                )
            ],
        }
        manifest = {
            "format_version": 1,
            "reference_kind": "upstream-full-model",
            "byte_order": "little",
            "checkpoint": report["checkpoint"],
            "source": report["source"],
            "model_state_sha256": report["model_state_sha256"],
            "comparisons": report["comparisons"],
            "prepared_files": list(report["files"]),
            "cases": [],
        }
        for i, (inputs, outputs) in enumerate(
            zip(
                self.fixture.prepared["inputs"],
                self.fixture.prepared["references"],
                strict=True,
            )
        ):
            case = {"case": i, "inputs": [], "outputs": []}
            for kind, tensors in (("inputs", inputs), ("outputs", outputs)):
                for j, tensor in enumerate(tensors):
                    path = output / f"references/case-{i}/{kind}-{j}.bin"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    data = tensor["data"]
                    if kind == "outputs" and self.wrong_reference:
                        data = struct.pack(
                            "<4f", *[x + 0.5 for x in struct.unpack("<4f", data)]
                        )
                    path.write_bytes(data)
                    identity = worker._file_identity(path, output)
                    case[kind].append(
                        {key: value for key, value in tensor.items() if key != "data"}
                        | identity
                    )
                    report["files"].append(identity)
            manifest["cases"].append(case)
        worker._write_json(output / "references/manifest.json", manifest)
        report["reference_manifest"] = worker._file_identity(
            output / "references/manifest.json", output
        )
        report["files"].append(report["reference_manifest"])
        worker._write_json(output / "preparation.json", report)
        return report

    def run_workflow(self, operation="build", **overrides):
        arguments = {
            "checkpoint_sha256": self.sha,
            "device": "cpu",
            "points": 8,
            "geometry_points": 16,
        } | overrides
        with (
            mock.patch.object(
                workflows,
                "_load_producer",
                return_value=SimpleNamespace(prepare=self.producer_prepare),
            ),
            mock.patch.object(
                worker, "_prepare_model", return_value=self.fixture.prepared
            ),
            mock.patch.object(worker, "_build_backend", side_effect=self.backend),
        ):
            if operation == "prepare":
                return workflows.prepare_workflow(
                    MODEL, self.producer, self.archive, self.output, **arguments
                )
            return workflows.build_workflow(
                MODEL,
                self.producer,
                self.archive,
                self.output,
                runtime=self.fixture.runtime,
                backends=["aoti"],
                **arguments,
            )

    def backend(self, *args):
        return self.fixture.backend(*args) | {"entrypoint": "program.pt2"}

    def expected(self, operation="build"):
        return dict(
            operation=operation,
            model_name=MODEL,
            checkpoint_sha256=self.sha,
            producer_hashes=self.hashes,
            device="cpu",
            points=8,
            geometry_points=16,
            backends=["aoti"] if operation == "build" else [],
        )

    def test_prepare_retains_inputs_and_portable_references_without_runtime(self):
        report = self.run_workflow("prepare")
        self.assertTrue(
            (self.output / "workflow.json").is_file(),
            "prepare must retain a workflow receipt",
        )
        self.assertEqual(report["status"], "prepared")
        self.assertTrue((self.output / "source/checkpoint.mdlus").is_file())
        self.assertTrue(
            (self.output / "preparation/references/manifest.json").is_file()
        )
        self.assertFalse((self.output / "build").exists())
        self.assertEqual(
            workflows.validate_workflow_completion(
                self.output, **self.expected("prepare")
            ),
            report,
        )

    def test_build_publishes_only_after_full_reference_and_native_bytes_match(self):
        report = self.run_workflow()
        self.assertTrue(
            (self.output / "workflow.json").is_file(),
            "build must retain a workflow receipt",
        )
        self.assertEqual(report["status"], "complete")
        release = json.loads(
            (self.output / "build/model/model-release.json").read_text()
        )
        self.assertTrue(release["qualification"]["passed"])
        qualification = json.loads(
            (self.output / "build/model/qualification/full-model.json").read_text()
        )
        self.assertTrue(qualification["passed"])
        self.assertEqual(len(qualification["variants"]["aoti"]["cases"]), 3)
        self.assertEqual(
            workflows.validate_workflow_completion(self.output, **self.expected()),
            report,
        )

    def test_full_reference_failure_preserves_failed_workflow_and_no_release(self):
        self.wrong_reference = True
        with self.assertRaisesRegex(ValueError, "parity|reference"):
            self.run_workflow()
        self.assertFalse((self.output / "build/model/model-release.json").exists())
        report = json.loads((self.output / "workflow.json").read_text())
        self.assertEqual(report["status"], "failed")
        qualification = json.loads(
            (self.output / "build/model/qualification/full-model.json").read_text()
        )
        self.assertFalse(qualification["passed"])

    def test_bad_checkpoint_pin_rejected_before_producer_or_output(self):
        for value in ("0" * 64, None, "not-a-sha256"):
            with (
                self.subTest(value=value),
                mock.patch.object(workflows, "_load_producer") as load,
            ):
                with self.assertRaisesRegex(ValueError, "checkpoint|SHA"):
                    workflows.prepare_workflow(
                        MODEL,
                        self.producer,
                        self.archive,
                        self.output,
                        checkpoint_sha256=value,
                    )
                load.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_completion_rejects_evidence_directories_replaced_by_external_symlinks(
        self,
    ):
        for index, relative in enumerate(
            (
                "producer",
                "preparation",
                "build",
                "build/model",
                "build/checks/aoti",
                "build/checks/aoti/case-0",
            )
        ):
            with self.subTest(directory=relative):
                self.output = self.root / f"workflow-symlink-{index}"
                self.run_workflow()
                directory = self.output / relative
                external = self.root / f"external-evidence-{index}"
                directory.rename(external)
                directory.symlink_to(external, target_is_directory=True)
                with self.assertRaisesRegex(ValueError, "symlink"):
                    workflows.validate_workflow_completion(
                        self.output, **self.expected()
                    )

    def test_existing_output_is_never_overwritten(self):
        self.output.mkdir()
        (self.output / "keep.txt").write_text("keep")
        with self.assertRaises(FileExistsError):
            self.run_workflow("prepare")
        self.assertEqual((self.output / "keep.txt").read_text(), "keep")

    def test_workflow_import_remains_framework_free(self):
        code = "import sys; sys.path[:0] = sys.argv[1:]; import qualification; assert 'torch' not in sys.modules"
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                code,
                str(ROOT / "model-builder/src"),
                str(Path(__file__).resolve().parent),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_completion_rejects_preparation_when_build_was_requested(self):
        self.run_workflow("prepare")
        with self.assertRaises((ValueError, RuntimeError)):
            workflows.validate_workflow_completion(self.output, **self.expected())

    def test_completion_binds_every_requested_selector(self):
        self.run_workflow("prepare")
        changes = {
            "checkpoint_sha256": "0" * 64,
            "producer_hashes": dict(self.hashes, **{"prepare.py": "0" * 64}),
            "device": "cuda",
            "points": 9,
            "geometry_points": 17,
            "model_name": "other-model",
            "backends": ["aoti"],
        }
        for key, value in changes.items():
            with self.subTest(key=key), self.assertRaises((ValueError, RuntimeError)):
                workflows.validate_workflow_completion(
                    self.output, **(self.expected("prepare") | {key: value})
                )

    def test_completion_rechecks_retained_source_reference_native_and_candidate_bytes(
        self,
    ):
        self.run_workflow()
        paths = [
            "producer/prepare.py",
            "source/checkpoint.mdlus",
            "preparation/config.json",
            "preparation/checkpoint.pt",
            "preparation/fixtures.pt",
            "preparation/references/case-0/inputs-0.bin",
            "preparation/references/case-0/outputs-0.bin",
            "build/checks/aoti/case-0/input-0.bin",
            "build/checks/aoti/case-0/output-0.bin",
            "build/checks/aoti/case-0/native-metadata.json",
            "build/model/backends/aoti/model.pt2",
            "build/model/qualification/full-model.json",
        ]
        for index, name in enumerate(paths):
            with self.subTest(path=name):
                root = self.root / f"tampered-{index}"
                shutil.copytree(self.output, root)
                workflows.validate_workflow_completion(root, **self.expected())
                with (root / name).open("ab") as handle:
                    handle.write(b"changed")
                with self.assertRaises((ValueError, RuntimeError)):
                    workflows.validate_workflow_completion(root, **self.expected())

    def test_rehashed_reference_case_cannot_substitute_different_cached_inputs(self):
        self.run_workflow()
        directory = self.output / "preparation"
        path = directory / "references/manifest.json"
        manifest = json.loads(path.read_text())
        manifest["cases"][0]["inputs"] = manifest["cases"][1]["inputs"]
        worker._write_json(path, manifest)
        report = json.loads((directory / "preparation.json").read_text())
        report["reference_manifest"] = worker._file_identity(path, directory)
        report["files"] = [
            worker._file_identity(directory / item["path"], directory)
            for item in report["files"]
        ]
        worker._write_json(directory / "preparation.json", report)
        workflow = json.loads((self.output / "workflow.json").read_text())
        workflow["preparation"] = worker._file_identity(
            directory / "preparation.json", self.output
        )
        worker._write_json(self.output / "workflow.json", workflow)
        with self.assertRaisesRegex((ValueError, RuntimeError), "input|reference"):
            workflows.validate_workflow_completion(self.output, **self.expected())


if __name__ == "__main__":
    unittest.main()
