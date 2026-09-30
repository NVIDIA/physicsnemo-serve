"""Geo v2 captures deslicing without changing the legacy nine-plugin contract."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pnmir_build import cli, worker
from pnmir_export import tensorrt_exact_graphs as graphs, tensorrt_profiles as profiles
import test_geotransolver_exact_profile
import test_tensorrt_profiles
import test_tensorrt_recipe_profiles
import test_worker


PROFILE = "geotransolver-exact-v2"
PLUGINS = (*profiles.EXACT_PLUGIN_NAMES, "exact_weighted_blend", "exact_deslice_bmm")


class GeoTransolverV2ProfileTests(unittest.TestCase):
    def fixture(self, kind):
        fixture = kind(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture

    def build(self, **options):
        helper = self.fixture(test_geotransolver_exact_profile.GeoTransolverProfileTests)
        return helper.build(PROFILE, **options)

    def test_tenth_library_is_required_without_changing_legacy_contract(self):
        helper = self.fixture(test_tensorrt_profiles.TensorRTProfileTests)
        libraries = dict(helper.libraries)
        for name in ("exact_weighted_blend", "exact_deslice_bmm"):
            path = helper.root / (name + ".dll")
            path.write_bytes((name + " native library").encode())
            libraries[name] = path
        self.assertEqual(profiles.validate_tensorrt_profile(PROFILE), PROFILE)
        self.assertEqual(tuple(profiles.resolve_plugin_libraries(PROFILE, libraries)), PLUGINS)
        self.assertEqual(profiles.profile_version(PROFILE), 3)
        self.assertTrue(profiles.requires_byte_identical(PROFILE))
        legacy = {name: path for name, path in libraries.items() if name != "exact_deslice_bmm"}
        with self.assertRaisesRegex(ValueError, "ten.*plugin"):
            profiles.resolve_plugin_libraries(PROFILE, legacy)
        self.assertEqual(tuple(profiles.resolve_plugin_libraries("geotransolver-exact", legacy)), PLUGINS[:-1])
        self.assertEqual(profiles.profile_version("geotransolver-exact"), 2)
        with self.assertRaisesRegex(ValueError, "nine.*plugin"):
            profiles.resolve_plugin_libraries("geotransolver-exact", libraries)
        registry = SimpleNamespace(get_creator=mock.Mock(return_value=object()))
        trt = SimpleNamespace(get_plugin_registry=lambda: registry)
        with mock.patch.object(profiles.ctypes, "CDLL"):
            handles, records = profiles.load_exact_plugins(trt, libraries, PROFILE)
        self.assertEqual(len(handles), 10)
        self.assertEqual(registry.get_creator.call_args_list[-1], mock.call("PNMIRExactDesliceBmm", "1", ""))
        self.assertEqual(records["exact_deslice_bmm"]["sha256"], hashlib.sha256(libraries["exact_deslice_bmm"].read_bytes()).hexdigest())
        self.assertEqual(profiles.required_operators(PROFILE)[-1], {"id": "pnmir.tensorrt-exact-deslice-bmm", "abi": "1"})

    def test_deslice_requires_attention_and_weighted_blend_before_rewrite(self):
        calls = []
        symbols = dict(test_tensorrt_profiles.TRANSFORMS,
                       exact_weighted_blend="_replace_weighted_blends",
                       exact_deslice_bmm="_replace_deslice_bmms")
        replacements = {symbol: mock.Mock(side_effect=lambda onnx, model, name=name: calls.append(name) or 1)
                        for name, symbol in symbols.items()}
        with mock.patch.multiple(graphs, **replacements):
            counts = profiles.prepare_exact_graph(object(), object(), PROFILE)
            self.assertEqual(set(counts), set(PLUGINS))
            self.assertLess(calls.index("exact_attention"), calls.index("exact_weighted_blend"))
            self.assertLess(calls.index("exact_weighted_blend"), calls.index("exact_deslice_bmm"))
            replacements["_replace_deslice_bmms"].side_effect = None
            replacements["_replace_deslice_bmms"].return_value = 0
            with self.assertRaisesRegex(ValueError, "no supported exact_deslice_bmm"):
                profiles.prepare_exact_graph(object(), object(), PROFILE)
            replacements["_replace_deslice_bmms"].reset_mock()
            profiles.prepare_exact_graph(object(), object(), "geotransolver-exact")
            replacements["_replace_deslice_bmms"].assert_not_called()

    def test_worker_preserves_scalar_sigmoid_gate_pass_and_captures_all_plugins(self):
        helper = self.fixture(test_tensorrt_recipe_profiles.TensorRTRecipeProfileTests)
        fixture = self.fixture(test_worker.WorkerTest)
        modules = helper.backend_modules()
        libraries = {name: Path(name + ".dll") for name in PLUGINS}
        assets = {"tensorrt_" + name + "_plugin": path for name, path in libraries.items()}
        with mock.patch.dict(sys.modules, modules):
            worker._build_backend("tensorrt", {"model": object(), "cases": [()], "assets": assets},
                                  dict(fixture.recipe, tensorrt_profile=PROFILE), "cuda",
                                  fixture.root / "package", fixture.root / "exported")
        exported = modules["pnmir_export.onnx_exporter"].export_onnx_model
        passes = exported.call_args.kwargs["options"].onnx_passes
        self.assertEqual([type(value).__name__ for value in passes], ["FreezeScalarSigmoidGates"])
        built = modules["pnmir_export.tensorrt_builder"].build_tensorrt_package
        self.assertEqual(built.call_args.kwargs["plugin_libraries"], libraries)
        self.assertEqual(built.call_args.kwargs["profile"], PROFILE)

    def test_deslice_asset_is_captured_and_locked(self):
        helper = self.fixture(test_tensorrt_recipe_profiles.TensorRTRecipeProfileTests)
        fixture = helper.exact_project()
        fixture.document["tensorrt_profile"] = PROFILE
        for name in ("exact_weighted_blend", "exact_deslice_bmm"):
            path = fixture.project / (name + ".dll")
            path.write_bytes((name + " native library").encode())
            fixture.document["assets"]["tensorrt_" + name + "_plugin"] = path.name
        fixture.write_project()
        code, result, _, _ = fixture.invoke(fixture.successful_process)
        self.assertEqual(code, 0, result)
        recipe = json.loads((fixture.output / "source/effective-recipe.json").read_text())
        asset = recipe["assets"]["tensorrt_exact_deslice_bmm_plugin"]
        self.assertEqual((fixture.output / "source" / asset["path"]).read_bytes(), path.read_bytes())
        identity = json.loads((fixture.output / "check.json").read_text())["input_identity"]
        from pnmir_build import project_lock
        lock = fixture.project / "model-build.lock.json"
        project_lock.publish_lock(lock, project_lock.inspect_lock(lock, "build", identity))
        path.write_bytes(b"changed deslice library")
        code, result, _, run = fixture.invoke(fixture.successful_process, "build")
        self.assertEqual(code, 2, result)
        self.assertEqual(result["diagnostics"][0]["code"], "PROJECT_LOCK_MISMATCH")
        run.assert_not_called()

    def test_native_gate_rejects_small_numeric_and_signed_zero_differences(self):
        for options in ({"difference": True}, {"signed_zero": True}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "byte-identical"):
                self.build(**options)

    def test_success_records_zero_limits_and_matching_hashes(self):
        fixture, _ = self.build()
        check = json.loads((fixture.output / "checks/tensorrt.json").read_text())
        self.assertTrue(check["require_byte_identical"])
        self.assertEqual(check["limits"], {"max_abs": 0.0, "relative_l2": 0.0})
        for case in check["cases"]:
            for output in case["outputs"]:
                self.assertEqual(output["actual_sha256"], output["reference_sha256"])

    def test_completion_rejects_missing_deslice_and_wrong_version(self):
        fixture, build = self.build()
        variant = build["variants"]["tensorrt"]
        variant["graph"]["entrypoint"] = "model.onnx"
        release_path = fixture.output / "model/model-release.json"
        release = json.loads(release_path.read_text())
        manifest_path = fixture.output / "model" / variant["package"] / "model.json"
        manifest = json.loads(manifest_path.read_text())
        artifact = manifest["artifacts"][0]
        artifact.update(correctness_profile={
            "name": PROFILE, "version": 3, "plugins": list(PLUGINS),
            "replacement_counts": {name: 1 for name in PLUGINS},
            "plugin_libraries": {name: {"filename": name + ".dll", "sha256": "0" * 64} for name in PLUGINS},
        }, required_operators=profiles.required_operators(PROFILE))
        plan = {"recipe": fixture.recipe, "backends": ["tensorrt"], "device": "cpu", "output": fixture.output}

        def publish():
            manifest_path.write_text(json.dumps(manifest))
            variant["files"] = worker._inventory(manifest_path.parent, fixture.output / "model")
            release["variants"]["tensorrt"]["files"] = variant["files"]
            release_path.write_text(json.dumps(release))
            build["release"] = worker._file_identity(release_path, fixture.output)
            (fixture.output / "build.json").write_text(json.dumps(build))

        publish()
        cli._validate_container_completion(plan)
        for change in ("operator", "count", "version"):
            with self.subTest(change=change):
                if change == "operator":
                    artifact["required_operators"].pop()
                elif change == "count":
                    artifact["correctness_profile"]["replacement_counts"]["exact_deslice_bmm"] = 0
                else:
                    artifact["correctness_profile"]["version"] = 2
                publish()
                with self.assertRaisesRegex(RuntimeError, "GeoTransolver"):
                    cli._validate_container_completion(plan)
                artifact["required_operators"] = profiles.required_operators(PROFILE)
                artifact["correctness_profile"]["replacement_counts"]["exact_deslice_bmm"] = 1
                artifact["correctness_profile"]["version"] = 3


if __name__ == "__main__":
    unittest.main()
