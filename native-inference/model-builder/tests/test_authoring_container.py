"""Framework-free container launch and completion protocol contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_builder.build import authoring, inputs  # noqa: E402
from authoring_test_support import AuthoringContainerFixture


class AuthoringContainerTests(AuthoringContainerFixture, unittest.TestCase):
    def test_container_command_preserves_digest_gpu_uid_and_mount_boundaries(self):
        plan = {
            "output": self.root / "outputs" / "candidate",
            "executor": "container",
            "device": "cuda:1",
            "required_gpu_arch": "sm90",
            "image": self.document["builder_image"],
        }
        snapshot = self.root / "captured sources"
        command = authoring._execution_command(plan, snapshot, "build")
        self.assertEqual(command[:3], ["docker", "run", "--rm"])
        self.assertIn(plan["image"], command)
        self.assertEqual(command[command.index("--gpus") + 1], "all")
        self.assertEqual(
            command[command.index("--user") + 1], f"{os.getuid()}:{os.getgid()}"
        )
        self.assertIn(f"type=bind,src={snapshot},dst=/inputs,readonly", command)
        self.assertIn(f"type=bind,src={plan['output'].parent},dst=/outputs", command)
        self.assertEqual(command[command.index("--output") + 1], "/outputs/candidate")
        self.assertEqual(command[command.index("--device") + 1], "cuda:1")
        self.assertEqual(command[command.index("--required-gpu-arch") + 1], "sm90")
        self.assertEqual(command[command.index("--operation") + 1], "build")
        self.assertNotIn(str(self.project), command)

    def test_arbitrary_host_uid_has_identity_and_writable_framework_caches(self):
        command = authoring._execution_command(
            {
                "output": self.root / "candidate",
                "executor": "container",
                "device": "cuda",
                "image": self.document["builder_image"],
            },
            self.root / "snapshot",
            "build",
        )
        environment = dict(
            command[index + 1].split("=", 1)
            for index, token in enumerate(command)
            if token in ("--env", "-e")
        )
        self.assertTrue(
            environment.get("USER"), "unlisted host UID needs a discoverable username"
        )
        self.assertTrue(environment.get("LOGNAME"))
        for name in (
            "TRITON_CACHE_DIR",
            "TORCHINDUCTOR_CACHE_DIR",
            "TORCH_EXTENSIONS_DIR",
            "XDG_CACHE_HOME",
        ):
            with self.subTest(name=name):
                self.assertTrue(environment.get(name, "").startswith("/tmp/"))

    def test_mutable_builder_image_is_rejected_before_launch(self):
        self.document["builder_image"] = "example/builder:latest"
        self.write_project()
        code, result, _, run = self.invoke(self.successful_process)
        self.assertEqual(code, 2, result)
        self.assertIn("digest", result["diagnostics"][0]["message"])
        run.assert_not_called()

    def test_matching_check_receipt_is_accepted_and_logs_are_retained(self):
        code, result, _, run = self.invoke(self.successful_process)
        self.assertEqual(code, 0, result)
        self.assertEqual(result["status"], "checked")
        self.assertEqual(run.call_count, 1)
        self.assertIn(
            "mock worker diagnostics", (self.output / "execution.log").read_text()
        )
        self.assertFalse((self.output / "model").exists())

    def test_selected_aoti_profile_is_retained_and_bound_to_project_lock(self):
        from model_builder.build import project_lock

        profile = "aten-boundary-exact-v2"
        self.document["aoti_profile"] = profile
        self.write_project()
        code, result, _, _ = self.invoke(self.successful_process)
        self.assertEqual(code, 0, result)
        for name in ("recipe.json", "effective-recipe.json"):
            recipe = json.loads((self.output / "source" / name).read_text())
            self.assertEqual(recipe["aoti_profile"], profile)
        report = json.loads((self.output / "check.json").read_text())
        identity = report["input_identity"]
        self.assertEqual(identity["aoti_profile"], profile)
        lock_path = self.project / "model-build.lock.json"
        project_lock.publish_lock(
            lock_path, project_lock.inspect_lock(lock_path, "build", identity)
        )

        self.document["aoti_profile"] = "baseline"
        self.write_project()
        code, result, _, run = self.invoke(self.successful_process, "build")
        self.assertEqual(code, 2, result)
        self.assertEqual(result["diagnostics"][0]["code"], "PROJECT_LOCK_MISMATCH")
        run.assert_not_called()

    def test_retained_recipe_cannot_change_selected_aoti_profile(self):
        for selected, retained in (
            (None, "aten-boundary-exact-v2"),
            ("aten-boundary-exact-v2", "baseline"),
            ("aten-boundary-exact-v2", None),
        ):
            with self.subTest(selected=selected, retained=retained):
                self.document.pop("aoti_profile", None)
                if selected is not None:
                    self.document["aoti_profile"] = selected
                self.write_project()

                def wrong(command, **kwargs):
                    process = self.successful_process(command, **kwargs)
                    path = self.output / "source" / "effective-recipe.json"
                    recipe = json.loads(path.read_text())
                    recipe.pop("aoti_profile", None)
                    if retained is not None:
                        recipe["aoti_profile"] = retained
                    path.write_text(json.dumps(recipe))
                    report_path = self.output / "check.json"
                    report = json.loads(report_path.read_text())
                    for record in report["source_files"]:
                        if record["path"] == "source/effective-recipe.json":
                            record.update(inputs._identity(path))
                    report_path.write_text(json.dumps(report))
                    return process

                code, result, _, _ = self.invoke(wrong)
                self.assertEqual(code, 1, result)
                self.assertIn("aoti_profile", result["diagnostics"][0]["message"])

    def test_zero_exit_without_check_receipt_is_rejected(self):
        code, result, _, _ = self.invoke(
            lambda command, **kwargs: subprocess.CompletedProcess(command, 0)
        )
        self.assertEqual(code, 1, result)
        self.assertEqual(result["status"], "failed")

    def test_zero_exit_without_native_receipts_cannot_complete_build(self):
        code, result, _, _ = self.invoke(self.successful_process, "build")
        self.assertEqual(code, 1, result)
        self.assertEqual(result["status"], "failed")
        self.assertFalse((self.output / "model" / "model-release.json").exists())

    def test_failed_process_cannot_succeed_even_with_a_complete_check_receipt(self):
        def fail(command, **kwargs):
            return self.successful_process(command, **kwargs, code=17)

        code, result, _, _ = self.invoke(fail)
        self.assertEqual(code, 1, result)
        self.assertEqual(result["status"], "failed")

    def test_failure_before_worker_start_retains_the_promised_execution_log(self):
        def fail(command, **kwargs):
            kwargs["stdout"].write("Cannot connect to the Docker daemon\n")
            return subprocess.CompletedProcess(command, 125)

        code, result, stderr, _ = self.invoke(fail)
        self.assertEqual(code, 1, result)
        self.assertIn("Docker daemon", stderr)
        self.assertTrue(
            (self.output / "execution.log").is_file(),
            "launch failure must retain the diagnostic log the CLI points to",
        )
        self.assertIn("Docker daemon", (self.output / "execution.log").read_text())

    def test_check_receipt_must_bind_schema_and_requested_operation(self):
        for change in ({"format_version": 99}, {"operation": "build"}):
            with self.subTest(change=change):

                def wrong(command, **kwargs):
                    return self.successful_process(
                        command, **kwargs, mutate=lambda report: report.update(change)
                    )

                code, result, _, _ = self.invoke(wrong)
                self.assertEqual(
                    code,
                    1,
                    "malformed or mismatched check operation was accepted: "
                    + str(result),
                )

    def test_check_receipt_must_confirm_the_selected_gpu_device_and_environment(self):
        for field in ("device", "required_gpu_arch", "environment"):
            with self.subTest(field=field):

                def mutate(report):
                    if field == "environment":
                        report["environment"]["compute_capability"] = [8, 0]
                    else:
                        report["target_check"][field] = (
                            "cuda:0" if field == "device" else "sm80"
                        )

                def wrong(command, **kwargs):
                    return self.successful_process(command, **kwargs, mutate=mutate)

                code, result, _, _ = self.invoke(wrong)
                self.assertEqual(
                    code, 1, "mismatched target evidence was accepted: " + str(result)
                )

    def test_check_source_inventory_must_match_selected_checkpoint(self):
        for mode in ("omit", "replace"):
            with self.subTest(mode=mode):

                def mutate(report):
                    if mode == "omit":
                        one = self.output / "source" / "recipe.json"
                        report["source_files"] = [
                            {"path": "source/recipe.json", **inputs._identity(one)}
                        ]
                    else:
                        (self.output / "source" / "checkpoint.pt").write_bytes(
                            b"different checkpoint"
                        )

                def wrong(command, **kwargs):
                    return self.successful_process(command, **kwargs, mutate=mutate)

                code, result, _, _ = self.invoke(wrong)
                self.assertEqual(
                    code,
                    1,
                    "inventoried source was not bound to selected inputs: "
                    + str(result),
                )

    def test_complete_native_protocol_fixture_passes_frontend_validation(self):
        self.document["device"] = "cpu"
        self.document.pop("required_gpu_arch")
        self.write_project()
        code, result, _, _ = self.invoke(self.native_protocol_process, "build")
        self.assertEqual(code, 0, result)
        self.assertEqual(result["status"], "complete")

    def test_native_receipts_require_matching_successful_source_qualification(self):
        self.document["device"] = "cpu"
        self.document.pop("required_gpu_arch")
        self.write_project()
        for mode in (
            "missing-qualification",
            "failed-source-check",
            "wrong-source-identity",
        ):
            with self.subTest(mode=mode):

                def wrong(command, **kwargs):
                    return self.native_protocol_process(command, mode=mode, **kwargs)

                code, result, _, _ = self.invoke(wrong, "build")
                self.assertEqual(
                    code, 1, "unqualified source receipt was accepted: " + str(result)
                )
                self.assertIn("source integrity", result["diagnostics"][0]["message"])

    def test_local_native_receipts_must_match_the_selected_runtime(self):
        runtime = self.project / "physicsnemo-infer"
        runtime.write_text("#!/bin/sh\nexit 0\n")
        runtime.chmod(0o755)
        self.document.update(executor="local", device="cpu", runtime=str(runtime))
        self.document.pop("required_gpu_arch")
        self.write_project()
        code, result, _, _ = self.invoke(self.native_protocol_process, "build")
        self.assertEqual(
            code,
            1,
            "native receipts from a different runtime were accepted: " + str(result),
        )
        self.assertIn("different native runtime", result["diagnostics"][0]["message"])


if __name__ == "__main__":
    unittest.main()
