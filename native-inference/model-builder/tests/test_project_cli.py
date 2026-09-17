"""Project commands use the existing build gates without agent orchestration."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pnmir_build import cli, worker
import test_worker


class ProjectCommandTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_worker.WorkerTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root.resolve()
        self.project = self.root / "customer-project"
        self.project.mkdir()
        self.config = self.project / "model-build.json"
        self.output_root = self.root / "results"
        self.settings = {
            "format_version": 1,
            "recipe": "../recipe.json",
            "executor": "local",
            "runtime": "../physicsnemo-infer",
            "device": "cpu",
            "backends": ["aoti"],
            "output_root": "../results",
        }
        self.write_config()

    def write_config(self):
        self.config.write_text(json.dumps(self.settings))

    def invoke(self, *arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                code = cli.main(list(arguments))
            except SystemExit as error:
                code = error.code
        return code, stdout.getvalue(), stderr.getvalue()

    def build(self, *extra):
        def prepare(*args, **kwargs):
            print("adapter diagnostic on Python stdout")
            os.write(1, b"native diagnostic on fd stdout\n")
            return self.fixture.prepared

        with (
            mock.patch.object(worker, "_prepare_model", side_effect=prepare),
            mock.patch.object(
                worker, "_build_backend", side_effect=self.fixture.backend
            ),
        ):
            return self.invoke("build", str(self.project), "--json", *extra)

    def result(self, invocation, expected=0):
        code, stdout, stderr = invocation
        self.assertEqual(code, expected, stdout + stderr)
        try:
            result = json.loads(stdout)
        except ValueError:
            self.fail(
                f"expected one JSON result, received {stdout!r}; stderr={stderr!r}"
            )
        self.assertEqual(result["schema_version"], 1)
        return result

    def test_doctor_resolves_project_from_other_cwd_without_ml_or_writes(self):
        launcher = Path(__file__).resolve().parents[2] / "physicsnemo-model-builder"
        with tempfile.TemporaryDirectory() as unrelated:
            run = subprocess.run(
                [
                    sys.executable,
                    "-S",
                    str(launcher),
                    "doctor",
                    str(self.project),
                    "--json",
                ],
                cwd=unrelated,
                capture_output=True,
                text=True,
                timeout=15,
            )
        result = self.result((run.returncode, run.stdout, run.stderr))
        self.assertEqual(result["status"], "configuration-ok")
        self.assertEqual(
            result["effective_config"]["runtime"], str(self.fixture.runtime.resolve())
        )
        self.assertEqual(result["effective_config"]["backends"], ["aoti"])
        self.assertFalse(self.output_root.exists())
        self.assertFalse((self.project / "model-build.lock.json").exists())

    def test_project_build_retains_effective_config_lock_and_all_native_checks(self):
        invocation = self.build()
        result = self.result(invocation)
        self.assertEqual(result["status"], "complete")
        output = Path(result["output"])
        self.assertEqual(output.parent, self.output_root)
        self.assertIn("adapter diagnostic", invocation[2])
        self.assertIn("native diagnostic", invocation[2])
        effective = json.loads(
            (output / "project" / "effective-config.json").read_text()
        )
        self.assertEqual(effective["backends"], ["aoti"])
        self.assertEqual(
            json.loads((output / "project" / "model-build.json").read_text()),
            self.settings,
        )
        lock = self.project / "model-build.lock.json"
        self.assertTrue(lock.is_file())
        retained = output / "project" / "model-build.lock.json"
        self.assertEqual(retained.read_bytes(), lock.read_bytes())
        execution = json.loads((output / "execution.json").read_text())
        self.assertEqual(
            execution["project"]["lock"]["sha256"],
            hashlib.sha256(retained.read_bytes()).hexdigest(),
        )
        for backend in ("aoti",):
            report = json.loads((output / "checks" / f"{backend}.json").read_text())
            self.assertTrue(report["passed"])
            self.assertEqual(len(report["cases"]), 2)
        second = self.result(self.build())
        self.assertNotEqual(result["output"], second["output"])

    def test_locked_source_change_fails_before_execution_until_explicit_refresh(self):
        self.result(self.build())
        adapter = self.fixture.recipe_path.parent / "export.py"
        adapter.write_text("# new customer adapter\n")
        previous = (self.project / "model-build.lock.json").read_bytes()
        with mock.patch.object(worker, "execute_build") as execute:
            result = self.result(self.invoke("build", str(self.project), "--json"), 2)
        execute.assert_not_called()
        self.assertEqual(result["diagnostics"][0]["code"], "PROJECT_LOCK_MISMATCH")
        self.assertEqual(
            (self.project / "model-build.lock.json").read_bytes(), previous
        )
        refreshed = self.result(self.build("--update-lock"))
        self.assertEqual(refreshed["status"], "complete")
        self.assertNotEqual(
            (self.project / "model-build.lock.json").read_bytes(), previous
        )

    def test_profile_then_cli_backend_override_replaces_list(self):
        self.settings.update(
            device="cuda",
            backends=["aoti", "tensorrt"],
            default_profile="single",
            profiles={"single": {"backends": ["aoti"]}},
        )
        self.write_config()
        result = self.result(
            self.invoke("doctor", str(self.project), "--backend", "tensorrt", "--json")
        )
        self.assertEqual(result["effective_config"]["backends"], ["tensorrt"])
        self.assertEqual(result["profile"], "single")

    def test_native_failure_preserves_failed_receipt_and_no_candidate(self):
        self.fixture.set_runtime("nan")
        result = self.result(self.build(), 1)
        self.assertEqual(result["status"], "failed")
        output = Path(result["output"])
        self.assertFalse((output / "model" / "model-release.json").exists())
        self.assertEqual(
            json.loads((output / "build.json").read_text())["status"], "failed"
        )
        self.assertTrue((output / "project" / "effective-config.json").is_file())

    def test_json_covers_parser_error_and_invalid_project_before_output(self):
        result = self.result(
            self.invoke("build", str(self.project), "--unknown", "--json"), 2
        )
        self.assertEqual(result["diagnostics"][0]["code"], "INVALID_ARGUMENT")
        self.settings["backedns"] = ["aoti"]
        self.write_config()
        result = self.result(self.invoke("build", str(self.project), "--json"), 2)
        self.assertEqual(result["diagnostics"][0]["code"], "INVALID_PROJECT")
        self.assertFalse(self.output_root.exists())

    def test_recipe_project_calculates_checkpoint_identity_without_loading_tensors(
        self,
    ):
        checkpoint = self.project / "weights.pt"
        checkpoint.write_bytes(b"opaque tensors only hashed by frontend")
        recipe = self.fixture.recipe | {
            "format_version": 2,
            "config": {"path": "config.json"},
            "checkpoint": {"format": "torch-state-dict"},
        }
        self.fixture.recipe_path.write_text(json.dumps(recipe))
        (self.root / "config.json").write_text("{}")
        self.settings.update(checkpoint="weights.pt", backends=["aoti"])
        self.write_config()
        result = self.result(self.invoke("doctor", str(self.project), "--json"))
        self.assertEqual(
            result["effective_config"]["checkpoint_sha256"],
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        )
        self.assertFalse((self.project / "model-build.lock.json").exists())

    def test_effective_config_includes_recipe_and_toolchain_defaults(self):
        self.settings.pop("backends")
        self.settings.pop("runtime")
        lock = self.project / "toolchain.json"
        lock.write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "runtime_path": str(self.fixture.runtime.resolve()),
                }
            )
        )
        self.settings["toolchain_lock"] = "toolchain.json"
        self.write_config()
        result = self.result(self.invoke("doctor", str(self.project), "--json"))
        self.assertEqual(result["effective_config"].get("backends"), ["aoti"])
        self.assertEqual(
            result["effective_config"].get("runtime"),
            str(self.fixture.runtime.resolve()),
        )

    def test_named_default_profile_does_not_collide_with_unprofiled_lock(self):
        self.result(self.build())
        self.settings["profiles"] = {
            "default": {"backends": ["aoti"], "device": "cuda"}
        }
        self.write_config()
        result = self.result(
            self.invoke("doctor", str(self.project), "--profile", "default", "--json")
        )
        self.assertEqual(result["status"], "configuration-ok")

    def test_changed_adapter_after_lock_is_rejected_before_model_loading(self):
        def change_after_lock(*args):
            (self.fixture.recipe_path.parent / "export.py").write_text(
                "# changed after lock\n"
            )

        with (
            mock.patch(
                "pnmir_build.targets.check_target", side_effect=change_after_lock
            ),
            mock.patch.object(
                worker, "_prepare_model", return_value=self.fixture.prepared
            ) as prepare,
            mock.patch.object(
                worker, "_build_backend", side_effect=self.fixture.backend
            ),
        ):
            result = self.result(self.invoke("build", str(self.project), "--json"), 1)
        prepare.assert_not_called()
        self.assertEqual(result["status"], "failed")
        self.assertFalse(
            (Path(result["output"]) / "model" / "model-release.json").exists()
        )

    def test_model_system_exit_zero_is_a_failed_build_not_successful_help(self):
        with mock.patch.object(worker, "_prepare_model", side_effect=SystemExit(0)):
            result = self.result(self.invoke("build", str(self.project), "--json"), 1)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["diagnostics"][0]["type"], "SystemExit")
        self.assertFalse(
            (Path(result["output"]) / "model" / "model-release.json").exists()
        )

    def test_abbreviated_json_option_is_rejected_instead_of_silent_success(self):
        code, stdout, stderr = self.invoke("list", "--j")
        self.assertEqual(code, 2, stdout + stderr)
        self.assertIn("unrecognized", stderr)

    def test_explicit_checkpoint_hash_does_not_bypass_regular_file_validation(self):
        archive = self.project / "real.pt"
        archive.write_bytes(b"opaque archive")
        alias = self.project / "alias.pt"
        alias.symlink_to(archive)
        self.settings.update(
            checkpoint="alias.pt",
            checkpoint_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        )
        self.write_config()
        result = self.result(self.invoke("doctor", str(self.project), "--json"), 2)
        self.assertEqual(result["diagnostics"][0]["code"], "INVALID_PROJECT")
        self.assertIn("symlink", result["diagnostics"][0]["message"])

    def test_target_environment_mismatch_stops_before_compilation(self):
        self.settings.update(device="cuda", required_gpu_arch="sm90")
        self.write_config()
        self.fixture.prepared["environment"]["compute_capability"] = [8, 6]
        self.fixture.runtime.write_text(
            self.fixture.runtime.read_text().replace("'type':'cpu'", "'type':'cuda'")
        )
        checked = {
            "device": "cuda",
            "required_gpu_arch": "sm90",
            "actual_gpu_arch": "sm90",
            "gpu_name": "test GPU",
        }
        with mock.patch("pnmir_build.targets.check_target", return_value=checked):
            result = self.result(self.build(), 1)
        self.assertEqual(
            self.fixture.calls, [], "a target mismatch must stop before compilation"
        )
        self.assertFalse(
            (Path(result["output"]) / "model" / "model-release.json").exists()
        )


if __name__ == "__main__":
    unittest.main()
