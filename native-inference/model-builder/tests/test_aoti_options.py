"""Framework-free compiler-option validation, configuration and receipt contracts."""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_builder.export.aoti_options import validate_aoti_options
from model_builder.build import authoring_config, cli, inputs, project_lock, worker
import authoring_test_support
import model_input_test_support
import worker_test_support


class AotiOptionTests(unittest.TestCase):
    def test_supported_booleans_are_copied_without_defaults_or_mutation(self):
        options = {
            "max_autotune": True,
            "epilogue_fusion": False,
            "shape_padding": True,
            "coordinate_descent_tuning": True,
        }
        validated = validate_aoti_options(options)
        self.assertEqual(validated, options)
        self.assertIsNot(validated, options)
        self.assertEqual(validate_aoti_options({}), {})

    def test_unknown_and_builder_owned_options_are_rejected(self):
        for key in (
            "max_autotune_typo",
            "triton.cudagraphs",
            "fallback_by_default",
            "aot_inductor.output_path",
            "post_grad_custom_pre_pass",
        ):
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(ValueError, "Unsupported AOTI option"),
            ):
                validate_aoti_options({key: True})

    def test_options_must_be_an_object(self):
        for options in (None, [], True, 1, "max-autotune"):
            with (
                self.subTest(options=options),
                self.assertRaisesRegex(ValueError, "aoti_options must be an object"),
            ):
                validate_aoti_options(options)

    def test_boolean_values_are_strictly_typed(self):
        for value in (None, 0, 1, "true", [], {}):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "must be a boolean"),
            ):
                validate_aoti_options({"max_autotune": value})

    def test_epilogue_fusion_requires_explicit_autotuning(self):
        for options in (
            {"epilogue_fusion": True},
            {"epilogue_fusion": True, "max_autotune": False},
        ):
            with (
                self.subTest(options=options),
                self.assertRaisesRegex(ValueError, "epilogue_fusion.*max_autotune"),
            ):
                validate_aoti_options(options)
        options = {"epilogue_fusion": True, "max_autotune": True}
        self.assertEqual(validate_aoti_options(options), options)
        self.assertEqual(
            validate_aoti_options({"epilogue_fusion": False}),
            {"epilogue_fusion": False},
        )

    def test_exact_profile_rejects_enabled_performance_options(self):
        for key in (
            "max_autotune",
            "epilogue_fusion",
            "shape_padding",
            "coordinate_descent_tuning",
        ):
            options = {key: True}
            if key == "epilogue_fusion":
                options["max_autotune"] = True
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(ValueError, "aten-boundary-exact-v2"),
            ):
                validate_aoti_options(options, "aten-boundary-exact-v2")
        self.assertEqual(
            validate_aoti_options({"max_autotune": False}, "aten-boundary-exact-v2"),
            {"max_autotune": False},
        )

    def test_validation_is_framework_free(self):
        command = [
            sys.executable,
            "-S",
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from model_builder.export.aoti_options import validate_aoti_options; "
            "assert validate_aoti_options({'max_autotune': True}) == {'max_autotune': True}; "
            "assert not {'torch', 'onnx', 'tensorrt'} & sys.modules.keys()",
            str(Path(__file__).resolve().parents[1] / "src"),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


EXACT = "aten-boundary-exact-v2"
SPEED = {"max_autotune": True, "epilogue_fusion": True}


class AotiOptionsConfigurationTests(unittest.TestCase):
    def fixture(self, kind):
        fixture = kind()
        fixture.addCleanup = self.addCleanup
        if hasattr(fixture, "setUp"):
            fixture.setUp()
        return fixture

    def test_options_remain_omitted_by_default_and_accept_explicit_map(self):
        fixture = self.fixture(authoring_test_support.AuthoringConfigFixture)
        fixture.write()
        self.assertNotIn(
            "aoti_options", authoring_config.load(fixture.path)["effective"]
        )
        for options in ({}, SPEED, {"shape_padding": False}):
            with self.subTest(options=options):
                fixture.write(aoti_options=options)
                self.assertEqual(
                    authoring_config.load(fixture.path)["effective"]["aoti_options"],
                    options,
                )

    def test_invalid_options_fail_project_and_recipe_readers(self):
        project = self.fixture(authoring_test_support.AuthoringConfigFixture)
        recipe = self.fixture(model_input_test_support.ModelInputFixture)
        recipe.recipe.update(dtype="float32", shape=[4])
        for options in (
            None,
            [],
            {"unknown": True},
            {"max_autotune": 1},
            {"max_autotune": "true"},
            {"epilogue_fusion": True},
        ):
            project.write(aoti_options=options)
            recipe.recipe["aoti_options"] = options
            recipe.recipe_path.write_text(json.dumps(recipe.recipe))
            for reader in (
                lambda: authoring_config.load(project.path),
                lambda: cli.read_recipe(recipe.recipe_path),
                lambda: worker._read_recipe(recipe.recipe_path, ["aoti"]),
            ):
                with self.subTest(options=options, reader=reader):
                    with self.assertRaisesRegex(
                        ValueError, "(?i)aoti.*option|epilogue_fusion"
                    ):
                        reader()

    def test_named_profiles_replace_option_maps_and_resolve_precision_policy(self):
        fixture = self.fixture(authoring_test_support.AuthoringConfigFixture)
        document = fixture.write(
            aoti_profile=EXACT,
            aoti_options={},
            profiles={
                "speed": {"aoti_profile": "baseline", "aoti_options": SPEED},
                "exact": {"aoti_profile": EXACT, "aoti_options": {}},
                "tensor": {"tensorrt_profile": "layout-order-exact"},
            },
        )
        selected = authoring_config.load(
            fixture.path, argparse.Namespace(profile="speed")
        )
        self.assertEqual(selected["effective"]["aoti_profile"], "baseline")
        self.assertEqual(selected["effective"]["aoti_options"], SPEED)
        self.assertEqual(selected["document"], document)
        document.update(aoti_profile="baseline", aoti_options=SPEED)
        fixture.path.write_text(json.dumps(document))
        selected = authoring_config.load(
            fixture.path, argparse.Namespace(profile="exact")
        )
        self.assertEqual(selected["effective"]["aoti_options"], {})
        self.assertEqual(selected["effective"]["aoti_profile"], EXACT)
        selected = authoring_config.load(
            fixture.path, argparse.Namespace(profile="tensor")
        )
        self.assertEqual(
            selected["effective"]["tensorrt_profile"], "layout-order-exact"
        )

    def test_inherited_profile_compatibility_is_checked_after_selection(self):
        fixture = self.fixture(authoring_test_support.AuthoringConfigFixture)
        fixture.write(aoti_profile=EXACT, profiles={"speed": {"aoti_options": SPEED}})
        with self.assertRaisesRegex(ValueError, "(?i)exact|aten-boundary"):
            authoring_config.load(fixture.path, argparse.Namespace(profile="speed"))
        fixture.write(aoti_options=SPEED, profiles={"exact": {"aoti_profile": EXACT}})
        with self.assertRaisesRegex(ValueError, "(?i)exact|aten-boundary"):
            authoring_config.load(fixture.path, argparse.Namespace(profile="exact"))
        fixture.write(
            aoti_profile=EXACT,
            profiles={"disabled": {"aoti_options": {"max_autotune": False}}},
        )
        selected = authoring_config.load(
            fixture.path, argparse.Namespace(profile="disabled")
        )
        self.assertEqual(selected["effective"]["aoti_options"], {"max_autotune": False})

    def test_exact_recipe_rejects_enabled_options_before_execution(self):
        fixture = self.fixture(model_input_test_support.ModelInputFixture)
        fixture.recipe.update(
            dtype="float32", shape=[4], aoti_profile=EXACT, aoti_options=SPEED
        )
        fixture.recipe_path.write_text(json.dumps(fixture.recipe))
        for reader in (
            cli.read_recipe,
            lambda path: worker._read_recipe(path, ["aoti"]),
        ):
            with (
                self.subTest(reader=reader),
                self.assertRaisesRegex(ValueError, "(?i)exact|aten-boundary"),
            ):
                reader(fixture.recipe_path)

    def test_framework_free_preflight_accepts_options(self):
        fixture = self.fixture(model_input_test_support.ModelInputFixture)
        fixture.recipe.update(dtype="float32", shape=[4], aoti_options=SPEED)
        fixture.recipe_path.write_text(json.dumps(fixture.recipe))
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
                "from model_builder.build import cli, worker; p=Path(sys.argv[2]); "
                "assert cli.read_recipe(p)['aoti_options']['max_autotune'] is True; "
                "assert worker._read_recipe(p,['aoti'])[0]['aoti_options']['max_autotune'] is True; "
                "assert 'torch' not in sys.modules",
                str(Path(__file__).resolve().parents[1] / "src"),
                str(fixture.recipe_path),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_worker_forwards_explicit_options_and_preserves_omitted_call(self):
        fixture = self.fixture(model_input_test_support.ModelInputFixture)
        fixture.recipe.update(dtype="float32", shape=[4])
        for options in (None, {}, SPEED):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as temp:
                recipe = copy.deepcopy(fixture.recipe)
                if options is not None:
                    recipe["aoti_options"] = options
                modules = worker_test_support.backend_modules()
                export = modules["model_builder.export.exporter"].export_package
                with mock.patch.dict(sys.modules, modules):
                    worker._build_backend(
                        "aoti",
                        {"model": object(), "cases": [()]},
                        recipe,
                        "cpu",
                        Path(temp) / "package",
                        Path(temp) / "exported",
                    )
                if options is None:
                    self.assertNotIn("aoti_options", export.call_args.kwargs)
                else:
                    self.assertEqual(
                        export.call_args.kwargs.get("aoti_options"), options
                    )

    def test_options_are_retained_and_bound_to_project_lock(self):
        fixture = self.fixture(authoring_test_support.AuthoringContainerFixture)
        fixture.document["aoti_options"] = SPEED
        fixture.write_project()
        code, result, _, _ = fixture.invoke(fixture.successful_process)
        self.assertEqual(code, 0, result)
        for name in ("recipe.json", "effective-recipe.json"):
            self.assertEqual(
                json.loads((fixture.output / "source" / name).read_text())[
                    "aoti_options"
                ],
                SPEED,
            )
        identity = json.loads((fixture.output / "check.json").read_text())[
            "input_identity"
        ]
        self.assertEqual(identity["aoti_options"], SPEED)
        lock = fixture.project / "model-build.lock.json"
        project_lock.publish_lock(
            lock, project_lock.inspect_lock(lock, "build", identity)
        )
        fixture.document["aoti_options"] = {}
        fixture.write_project()
        code, result, _, run = fixture.invoke(fixture.successful_process, "build")
        self.assertEqual(code, 2, result)
        self.assertEqual(result["diagnostics"][0]["code"], "PROJECT_LOCK_MISMATCH")
        run.assert_not_called()

    def test_retained_recipe_cannot_change_options(self):
        fixture = self.fixture(authoring_test_support.AuthoringContainerFixture)
        fixture.document["aoti_options"] = SPEED
        fixture.write_project()

        def wrong(command, **kwargs):
            process = fixture.successful_process(command, **kwargs)
            path = fixture.output / "source" / "effective-recipe.json"
            recipe = json.loads(path.read_text())
            recipe["aoti_options"] = {}
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
        self.assertIn("aoti_options", result["diagnostics"][0]["message"])

    def test_completion_requires_matching_requested_and_applied_options(self):
        fixture = self.fixture(model_input_test_support.ModelInputFixture)
        plan, _, _, build, release, _ = fixture.completed_v2()
        package = plan["output"] / "model" / build["variants"]["aoti"]["package"]
        manifest_path = package / "model.json"
        manifest = json.loads(manifest_path.read_text())

        def publish(metadata):
            artifact = manifest["artifacts"][0]
            artifact.pop("compiler_options", None)
            if metadata is not None:
                artifact["compiler_options"] = metadata
            manifest_path.write_text(json.dumps(manifest))
            worker_test_support.publish_build_receipts(
                plan["output"], build, release, "aoti"
            )

        for selected in ({}, SPEED):
            plan["recipe"]["aoti_options"] = selected
            wrong = SPEED if not selected else {}
            for metadata in (
                {"requested": wrong, "applied": selected},
                {"requested": selected, "applied": wrong},
                {"requested": selected, "applied": {"max_autotune": 1}},
            ):
                publish(metadata)
                with self.subTest(selected=selected, metadata=metadata):
                    with self.assertRaisesRegex(
                        RuntimeError, "(?i)options|max_autotune"
                    ):
                        cli._validate_container_completion(plan)
            publish(None)
            if selected:
                with self.assertRaisesRegex(RuntimeError, "(?i)compiler.options"):
                    cli._validate_container_completion(plan)
            else:
                cli._validate_container_completion(plan)
            publish({"requested": selected, "applied": selected})
            cli._validate_container_completion(plan)
            if selected:
                publish({"requested": selected, "applied": selected, "effective": {}})
                with self.assertRaisesRegex(RuntimeError, "effective compiler options"):
                    cli._validate_container_completion(plan)
                publish(
                    {"requested": selected, "applied": selected, "effective": selected}
                )
                cli._validate_container_completion(plan)


if __name__ == "__main__":
    unittest.main()
