"""Explicit TensorRT plugins are captured inputs, never implicit environment paths."""

from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pnmir_build import authoring_config, cli, inputs, project_lock, worker
import test_authoring_config
import test_authoring_container
import test_model_input_cli
import test_worker


EXACT = "layout-order-exact"
PLUGINS = (
    "exact_linear",
    "exact_gemm",
    "exact_token_sum",
    "exact_slice_bmm",
    "exact_layer_norm",
    "exact_softmax",
    "exact_attention",
    "exact_gelu",
)


class TensorRTRecipeProfileTests(unittest.TestCase):
    def fixture(self, fixture_type):
        fixture = fixture_type(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture

    def test_project_accepts_profile_and_keeps_omitted_default(self):
        fixture = self.fixture(test_authoring_config.AuthoringConfigurationTests)
        fixture.write()
        self.assertNotIn(
            "tensorrt_profile", authoring_config.load(fixture.path)["effective"]
        )
        for profile in ("baseline", EXACT):
            fixture.write(tensorrt_profile=profile)
            self.assertEqual(
                authoring_config.load(fixture.path)["effective"]["tensorrt_profile"],
                profile,
            )

    def test_invalid_profile_rejected_by_project_frontend_and_worker(self):
        project = self.fixture(test_authoring_config.AuthoringConfigurationTests)
        recipe = self.fixture(test_model_input_cli.ModelInputCliTests)
        recipe.recipe.update(dtype="float32", shape=[4])
        for profile in ("unknown", "", None, {}, [], 2):
            project.write(tensorrt_profile=profile)
            recipe.recipe["tensorrt_profile"] = profile
            recipe.recipe_path.write_text(json.dumps(recipe.recipe))
            for reader in (
                lambda: authoring_config.load(project.path),
                lambda: cli.read_recipe(recipe.recipe_path),
                lambda: worker._read_recipe(recipe.recipe_path, ["aoti"]),
            ):
                with self.subTest(profile=profile, reader=reader):
                    with self.assertRaisesRegex(ValueError, "(?i)tensorrt.*profile"):
                        reader()

    def test_profile_preflight_imports_no_framework(self):
        fixture = self.fixture(test_model_input_cli.ModelInputCliTests)
        fixture.recipe.update(dtype="float32", shape=[4], tensorrt_profile=EXACT)
        fixture.recipe_path.write_text(json.dumps(fixture.recipe))
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
                "from pnmir_build import cli, worker; p = Path(sys.argv[2]); "
                "assert cli.read_recipe(p)['tensorrt_profile'] == sys.argv[3]; "
                "assert worker._read_recipe(p, ['aoti'])[0]['tensorrt_profile'] == sys.argv[3]; "
                "assert not {'torch', 'onnx', 'tensorrt'} & sys.modules.keys()",
                str(Path(__file__).resolve().parents[1] / "src"),
                str(fixture.recipe_path),
                EXACT,
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def backend_modules(self):
        modules = {"torch": ModuleType("torch")}
        modules["torch"].cuda = SimpleNamespace(
            current_device=lambda: 0, set_device=mock.Mock()
        )
        modules["torch"].device = lambda device: device
        for module_name, symbol in (
            ("exporter", "export_package"),
            ("onnx_exporter", "export_onnx_model"),
            ("tensorrt_builder", "build_tensorrt_package"),
            ("tensorrt_exporter", "tensorrt_ieee_fp32"),
        ):
            name = f"pnmir_export.{module_name}"
            modules[name] = ModuleType(name)
            setattr(modules[name], symbol, mock.Mock())
        modules["pnmir_export.onnx_exporter"].export_onnx_model.return_value = Path(
            "model.onnx"
        )
        modules["pnmir_export.tensorrt_exporter"].tensorrt_ieee_fp32 = nullcontext
        return modules

    def test_worker_forwards_only_exact_profile_with_all_captured_assets(self):
        fixture = self.fixture(test_model_input_cli.ModelInputCliTests)
        fixture.recipe.update(dtype="float32", shape=[4])
        libraries = {name: fixture.root / (name + ".so") for name in PLUGINS}
        assets = {
            "tensorrt_" + name + "_plugin": path for name, path in libraries.items()
        }
        for profile in (EXACT, "baseline"):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as temp:
                modules = self.backend_modules()
                build = modules["pnmir_export.tensorrt_builder"].build_tensorrt_package
                with mock.patch.dict(sys.modules, modules):
                    worker._build_backend(
                        "tensorrt",
                        {"model": object(), "cases": [()], "assets": assets},
                        dict(fixture.recipe, tensorrt_profile=profile),
                        "cuda:1",
                        Path(temp) / "package",
                        Path(temp) / "exported",
                    )
                if profile == EXACT:
                    self.assertEqual(build.call_args.kwargs.get("profile"), EXACT)
                    self.assertEqual(
                        build.call_args.kwargs.get("plugin_libraries"), libraries
                    )
                else:
                    self.assertNotIn("profile", build.call_args.kwargs)
                    self.assertNotIn("plugin_libraries", build.call_args.kwargs)

    def test_missing_plugin_fails_before_export_but_does_not_affect_aoti(self):
        fixture = self.fixture(test_model_input_cli.ModelInputCliTests)
        fixture.recipe.update(dtype="float32", shape=[4], tensorrt_profile=EXACT)
        for backend in ("tensorrt", "aoti"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as temp:
                modules = self.backend_modules()
                export = modules["pnmir_export.onnx_exporter"].export_onnx_model
                with mock.patch.dict(sys.modules, modules):
                    args = (
                        backend,
                        {"model": object(), "cases": [()]},
                        fixture.recipe,
                        "cuda",
                        Path(temp) / "package",
                        Path(temp) / "exported",
                    )
                    if backend == "tensorrt":
                        with self.assertRaisesRegex(
                            ValueError, "tensorrt_exact_linear_plugin"
                        ):
                            worker._build_backend(*args)
                        export.assert_not_called()
                    else:
                        worker._build_backend(*args)

    def exact_project(self):
        fixture = self.fixture(test_authoring_container.AuthoringContainerTests)
        fixture.document["tensorrt_profile"] = EXACT
        for name in PLUGINS:
            asset = "tensorrt_" + name + "_plugin"
            path = fixture.project / (name + ".so")
            path.write_bytes((name + " test plugin input").encode())
            fixture.document["assets"][asset] = path.name
        fixture.write_project()
        return fixture

    def test_profile_and_plugin_bytes_are_retained_and_bound_to_lock(self):
        for change in ("profile", "plugin"):
            with self.subTest(change=change):
                fixture = self.exact_project()
                code, result, _, _ = fixture.invoke(fixture.successful_process)
                self.assertEqual(code, 0, result)
                for filename in ("recipe.json", "effective-recipe.json"):
                    recipe = json.loads(
                        (fixture.output / "source" / filename).read_text()
                    )
                    self.assertEqual(recipe["tensorrt_profile"], EXACT)
                    for name in PLUGINS:
                        asset = recipe["assets"]["tensorrt_" + name + "_plugin"]
                        self.assertEqual(
                            (fixture.output / "source" / asset["path"]).read_bytes(),
                            (fixture.project / (name + ".so")).read_bytes(),
                        )
                identity = json.loads((fixture.output / "check.json").read_text())[
                    "input_identity"
                ]
                self.assertEqual(identity["tensorrt_profile"], EXACT)
                lock_path = fixture.project / "model-build.lock.json"
                project_lock.publish_lock(
                    lock_path, project_lock.inspect_lock(lock_path, "build", identity)
                )
                if change == "profile":
                    fixture.document["tensorrt_profile"] = "baseline"
                    fixture.write_project()
                else:
                    (fixture.project / "exact_linear.so").write_bytes(b"changed plugin")
                code, result, _, run = fixture.invoke(
                    fixture.successful_process, "build"
                )
                self.assertEqual(code, 2, result)
                self.assertEqual(
                    result["diagnostics"][0]["code"], "PROJECT_LOCK_MISMATCH"
                )
                run.assert_not_called()

    def test_retained_recipe_cannot_change_profile(self):
        fixture = self.exact_project()

        def wrong(command, **kwargs):
            process = fixture.successful_process(command, **kwargs)
            path = fixture.output / "source" / "effective-recipe.json"
            recipe = json.loads(path.read_text())
            recipe["tensorrt_profile"] = "baseline"
            path.write_text(json.dumps(recipe))
            report_path = fixture.output / "check.json"
            report = json.loads(report_path.read_text())
            for record in report["source_files"]:
                if record["path"] == "source/effective-recipe.json":
                    record.update(inputs._identity(path))
            report_path.write_text(json.dumps(report))
            return process

        code, result, _, _ = fixture.invoke(wrong)
        self.assertEqual(code, 1, result)
        self.assertIn("tensorrt_profile", result["diagnostics"][0]["message"])

    def test_completion_rejects_missing_or_wrong_artifact_profile(self):
        fixture = self.fixture(test_worker.WorkerTest)
        build = fixture.run_build(["tensorrt"])
        build_path = fixture.output / "build.json"
        release_path = fixture.output / "model" / "model-release.json"
        release = json.loads(release_path.read_text())
        variant = build["variants"]["tensorrt"]
        variant["graph"]["entrypoint"] = "model.onnx"
        package = fixture.output / "model" / variant["package"]
        manifest_path = package / "model.json"
        manifest = json.loads(manifest_path.read_text())
        plan = {
            "recipe": dict(fixture.recipe, tensorrt_profile=EXACT),
            "backends": ["tensorrt"],
            "device": "cpu",
            "output": fixture.output,
        }

        def publish(profile):
            artifact = manifest["artifacts"][0]
            artifact.pop("correctness_profile", None)
            if profile is not None:
                artifact["correctness_profile"] = {"name": profile, "version": 1}
            manifest_path.write_text(json.dumps(manifest))
            files = worker._inventory(package, fixture.output / "model")
            variant["files"] = files
            release["variants"]["tensorrt"]["files"] = files
            release_path.write_text(json.dumps(release))
            build["release"] = worker._file_identity(release_path, fixture.output)
            build_path.write_text(json.dumps(build))

        for profile in (None, "baseline", "unknown"):
            publish(profile)
            with self.subTest(profile=profile):
                with self.assertRaisesRegex(RuntimeError, "TensorRT.*profile"):
                    cli._validate_container_completion(plan)
        publish(EXACT)
        cli._validate_container_completion(plan)


if __name__ == "__main__":
    unittest.main()
