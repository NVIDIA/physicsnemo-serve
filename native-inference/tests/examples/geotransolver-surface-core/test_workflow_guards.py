"""Preserve producer identity and target gates in the external example."""

import copy
import json
import unittest
from unittest import mock

import test_workflows
from test_workflows import worker, workflows


def identity(path):
    return {
        key: value
        for key, value in worker._file_identity(path).items()
        if key != "path"
    }


class WorkflowProjectBindingTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_workflows.WorkflowTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.expected = {
            "producer": copy.deepcopy(self.fixture.hashes),
            "runtime": identity(self.fixture.fixture.runtime),
        }

    def test_matching_producer_and_runtime_preserve_full_model_gate(self):
        receipt = self.fixture.run_workflow(expected_identity=self.expected)
        self.assertEqual(receipt["status"], "complete")
        self.assertTrue(receipt["qualification"]["passed"])

    def test_build_rejects_changed_producer_before_loading_or_output(self):
        f = self.fixture
        (f.producer / "prepare.py").write_text("# changed producer\n")
        with mock.patch.object(workflows, "_load_producer") as load:
            with self.assertRaisesRegex(ValueError, "producer"):
                workflows.build_workflow(
                    test_workflows.MODEL,
                    f.producer,
                    f.archive,
                    f.output,
                    checkpoint_sha256=f.sha,
                    runtime=f.fixture.runtime,
                    backends=["aoti"],
                    device="cpu",
                    points=8,
                    geometry_points=16,
                    expected_identity=self.expected,
                )
        load.assert_not_called()
        self.assertFalse(f.output.exists())

    def test_prepare_rejects_changed_producer_before_loading_or_output(self):
        f = self.fixture
        (f.producer / "adapter.py").write_text("# changed adapter\n")
        with mock.patch.object(workflows, "_load_producer") as load:
            with self.assertRaisesRegex(ValueError, "producer"):
                workflows.prepare_workflow(
                    test_workflows.MODEL,
                    f.producer,
                    f.archive,
                    f.output,
                    checkpoint_sha256=f.sha,
                    device="cpu",
                    points=8,
                    geometry_points=16,
                    expected_identity={"producer": self.expected["producer"]},
                )
        load.assert_not_called()
        self.assertFalse(f.output.exists())

    def test_workflow_passes_runtime_binding_to_generated_worker(self):
        runtime = self.fixture.fixture.runtime
        runtime.write_text(runtime.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "runtime"):
            self.fixture.run_workflow(expected_identity=self.expected)
        self.assertFalse(
            (self.fixture.output / "build/model/model-release.json").exists()
        )


class WorkflowTargetGateTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_workflows.WorkflowTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.fixture.set_runtime("device")
        self.fixture.fixture.prepared["environment"]["compute_capability"] = [9, 0]

    def test_workflow_forwards_matching_target_to_full_qualification_build(self):
        receipt = self.fixture.run_workflow(device="cuda", required_gpu_arch="sm90")
        self.assertEqual(receipt["status"], "complete")
        self.assertTrue(receipt["qualification"]["passed"])

    def test_workflow_mismatch_cannot_leave_complete_candidate(self):
        f = self.fixture
        f.fixture.prepared["environment"]["compute_capability"] = [8, 6]
        with self.assertRaisesRegex(ValueError, "architecture"):
            f.run_workflow(device="cuda", required_gpu_arch="sm90")
        self.assertFalse((f.output / "build/model/model-release.json").exists())
        self.assertEqual(
            json.loads((f.output / "workflow.json").read_text())["status"], "failed"
        )
        self.assertEqual(f.fixture.calls, [])

    def test_prepare_rejects_invalid_architecture_before_producer_loading(self):
        f = self.fixture
        with mock.patch.object(workflows, "_load_producer") as load:
            with self.assertRaisesRegex(ValueError, "architecture"):
                workflows.prepare_workflow(
                    test_workflows.MODEL,
                    f.producer,
                    f.archive,
                    f.output,
                    checkpoint_sha256=f.sha,
                    device="cpu",
                    required_gpu_arch="sm90",
                )
        load.assert_not_called()
        self.assertFalse(f.output.exists())


if __name__ == "__main__":
    unittest.main()
