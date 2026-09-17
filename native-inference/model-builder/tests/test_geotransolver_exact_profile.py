"""GeoTransolver exact profile, captured dependencies and byte parity gates."""

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import struct
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pnmir_build import cli, worker
from pnmir_export import tensorrt_exact_graphs as graphs, tensorrt_profiles as profiles
import test_tensorrt_profiles
import test_tensorrt_recipe_profiles
import test_tensorrt_payload
import test_worker


GEO = "geotransolver-exact"
PLUGINS = (*profiles.EXACT_PLUGIN_NAMES, "exact_weighted_blend")


class GeoTransolverProfileTests(unittest.TestCase):
    def fixture(self, kind=test_worker.WorkerTest):
        fixture = kind(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture

    def test_geo_profile_requires_ninth_library_and_preserves_transolver_contract(self):
        fixture = self.fixture(test_tensorrt_profiles.TensorRTProfileTests)
        path = fixture.root / "exact_weighted_blend.so"
        path.write_bytes(b"weighted blend test library")
        libraries = dict(fixture.libraries, exact_weighted_blend=path)
        try:
            actual = profiles.resolve_plugin_libraries(GEO, libraries)
        except ValueError as error:
            self.fail(f"Geo profile must accept all nine declared libraries: {error}")
        self.assertEqual(tuple(actual), PLUGINS)
        with self.assertRaisesRegex(ValueError, "nine.*plugin"):
            profiles.resolve_plugin_libraries(GEO, fixture.libraries)
        with self.assertRaisesRegex(ValueError, "eight.*plugin"):
            profiles.resolve_plugin_libraries("layout-order-exact", libraries)
        self.assertEqual(len(profiles.required_operators()), 8)
        operators = profiles.required_operators(GEO)
        self.assertEqual(len(operators), 9)
        self.assertEqual(operators[-1], {"id": "pnmir.tensorrt-exact-weighted-blend", "abi": "1"})

    def test_geo_graph_adds_blend_rewrite_and_requires_a_match(self):
        calls = []
        symbols = dict(test_tensorrt_profiles.TRANSFORMS)
        symbols["exact_weighted_blend"] = "_replace_weighted_blends"
        replacements = {
            symbol: mock.Mock(side_effect=lambda onnx, model, name=name: calls.append(name) or 1)
            for name, symbol in symbols.items()
        }
        options = {"profile": GEO} if "profile" in inspect.signature(profiles.prepare_exact_graph).parameters else {}
        with mock.patch.multiple(graphs, create=True, **replacements):
            counts = profiles.prepare_exact_graph(object(), object(), **options)
            self.assertEqual(set(counts), set(PLUGINS))
            self.assertEqual(calls[-1], "exact_weighted_blend")
            replacements["_replace_weighted_blends"].side_effect = None
            replacements["_replace_weighted_blends"].return_value = 0
            with self.assertRaisesRegex(ValueError, "no supported exact_weighted_blend"):
                profiles.prepare_exact_graph(object(), object(), profile=GEO)

    def test_worker_captures_ninth_plugin_and_enables_gate_pass_only_for_geo(self):
        helper = self.fixture(test_tensorrt_recipe_profiles.TensorRTRecipeProfileTests)
        fixture = self.fixture()
        libraries = {name: fixture.root / (name + ".so") for name in PLUGINS}
        assets = {"tensorrt_" + name + "_plugin": path for name, path in libraries.items()}
        modules = helper.backend_modules()
        exporter = modules["pnmir_export.onnx_exporter"].export_onnx_model
        build = modules["pnmir_export.tensorrt_builder"].build_tensorrt_package
        with mock.patch.dict(sys.modules, modules):
            worker._build_backend(
                "tensorrt", {"model": object(), "cases": [()], "assets": assets},
                dict(fixture.recipe, tensorrt_profile=GEO), "cuda",
                fixture.root / "package", fixture.root / "exported",
            )
        self.assertEqual(build.call_args.kwargs["plugin_libraries"], libraries)
        self.assertEqual(build.call_args.kwargs["profile"], GEO)
        passes = exporter.call_args.kwargs["options"].onnx_passes
        self.assertEqual([type(value).__name__ for value in passes], ["FreezeScalarSigmoidGates"])

    def test_ninth_plugin_is_captured_and_changes_fail_the_project_lock(self):
        helper = self.fixture(test_tensorrt_recipe_profiles.TensorRTRecipeProfileTests)
        fixture = helper.exact_project()
        fixture.document["tensorrt_profile"] = GEO
        path = fixture.project / "exact_weighted_blend.so"
        path.write_bytes(b"original weighted blend library")
        fixture.document["assets"]["tensorrt_exact_weighted_blend_plugin"] = path.name
        fixture.write_project()
        code, result, _, _ = fixture.invoke(fixture.successful_process)
        self.assertEqual(code, 0, result)
        recipe = json.loads((fixture.output / "source/effective-recipe.json").read_text())
        asset = recipe["assets"]["tensorrt_exact_weighted_blend_plugin"]
        self.assertEqual((fixture.output / "source" / asset["path"]).read_bytes(), path.read_bytes())
        identity = json.loads((fixture.output / "check.json").read_text())["input_identity"]
        self.assertEqual(identity["tensorrt_profile"], GEO)
        from pnmir_build import project_lock
        lock_path = fixture.project / "model-build.lock.json"
        project_lock.publish_lock(lock_path, project_lock.inspect_lock(lock_path, "build", identity))
        path.write_bytes(b"changed weighted blend library")
        code, result, _, run = fixture.invoke(fixture.successful_process, "build")
        self.assertEqual(code, 2, result)
        self.assertEqual(result["diagnostics"][0]["code"], "PROJECT_LOCK_MISMATCH")
        run.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional Torch")
    def test_published_geo_payload_names_all_nine_verified_libraries(self):
        from types import SimpleNamespace
        fixture = self.fixture(test_tensorrt_payload.TensorRTPayloadTests)
        libraries, records = {}, {}
        for name in PLUGINS:
            path = fixture.root / (name + ".so")
            path.write_bytes((name + " test native bytes").encode())
            libraries[name] = path
            records[name] = {"filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        model = SimpleNamespace(SerializeToString=lambda: b"rewritten-onnx")
        fixture.onnx.load_model.return_value = model
        fixture.trt.OnnxParser.return_value.parse = mock.Mock(return_value=True)
        with (
            mock.patch.object(fixture.module, "load_exact_plugins", return_value=([object()], records)) as load,
            mock.patch.object(fixture.module, "prepare_exact_graph", return_value={name: 1 for name in PLUGINS}) as rewrite,
        ):
            fixture.build(profile=GEO, plugin_libraries=libraries)
        load.assert_called_once_with(fixture.trt, {name: path.resolve() for name, path in libraries.items()}, GEO)
        rewrite.assert_called_once_with(fixture.onnx, model, GEO)
        artifact = json.loads((fixture.output / "model.json").read_text())["artifacts"][0]
        correctness = artifact["correctness_profile"]
        self.assertEqual(correctness["version"], 2)
        self.assertEqual(correctness["name"], GEO)
        self.assertEqual(correctness["plugins"], list(PLUGINS))
        self.assertEqual(correctness["plugin_libraries"], records)
        self.assertEqual(artifact["required_operators"], profiles.required_operators(GEO))

    def build(self, profile, *, difference=False, signed_zero=False):
        fixture = self.fixture()
        fixture.recipe["tensorrt_profile"] = profile
        fixture.recipe_path.write_text(json.dumps(fixture.recipe))
        if difference:
            values = list(struct.unpack("<4f", fixture.prepared["references"][0][0]["data"]))
            values[0] += 1.0e-6
            fixture.prepared["references"][0][0]["data"] = struct.pack("<4f", *values)
        if signed_zero:
            fixture.prepared["inputs"][0] = (test_worker.tensor("input", [-0.5, 0.0, 1.0, 2.0]),)
            fixture.prepared["references"][0] = (test_worker.tensor("output", [-0.0, 1.0, 3.0, 5.0]),)
        # Separate the native gate behavior from profile schema acceptance.
        with mock.patch.object(profiles, "validate_tensorrt_profile", side_effect=lambda name: name):
            report = fixture.run_build(["tensorrt"])
        return fixture, report

    def test_geo_rejects_small_numeric_difference_that_baseline_accepts(self):
        fixture, report = self.build("baseline", difference=True)
        self.assertEqual(report["status"], "complete")
        with self.assertRaisesRegex(ValueError, "byte-identical"):
            self.build(GEO, difference=True)

    def test_geo_rejects_signed_zero_byte_difference(self):
        fixture, report = self.build("baseline", signed_zero=True)
        self.assertEqual(report["status"], "complete")
        with self.assertRaisesRegex(ValueError, "byte-identical"):
            self.build(GEO, signed_zero=True)

    def test_exact_native_success_records_zero_limits_and_matching_hashes(self):
        fixture, report = self.build(GEO)
        check = json.loads((fixture.output / "checks/tensorrt.json").read_text())
        self.assertTrue(check["passed"])
        self.assertEqual(check["limits"], {"max_abs": 0.0, "relative_l2": 0.0})
        self.assertTrue(check["require_byte_identical"])
        for case in check["cases"]:
            for value in case["outputs"]:
                self.assertEqual(value["actual_sha256"], value["reference_sha256"])
                self.assertEqual(value["max_abs"], 0.0)
                self.assertEqual(value["relative_l2"], 0.0)

    def test_completion_rechecks_geo_byte_evidence_and_operator_metadata(self):
        fixture, build = self.build(GEO)
        variant = build["variants"]["tensorrt"]
        variant["graph"]["entrypoint"] = "model.onnx"
        release_path = fixture.output / "model/model-release.json"
        release = json.loads(release_path.read_text())
        manifest_path = fixture.output / "model" / variant["package"] / "model.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifacts"][0].update(
            correctness_profile={
                "name": GEO, "version": 2, "plugins": list(PLUGINS),
                "replacement_counts": {name: 1 for name in PLUGINS},
                "plugin_libraries": {name: {"filename": name + ".so", "sha256": "0" * 64} for name in PLUGINS},
            },
            required_operators=[{"id": "pnmir.tensorrt-" + name.replace("_", "-"), "abi": "1"} for name in PLUGINS],
        )
        check_path = fixture.output / "checks/tensorrt.json"
        original_check = check_path.read_bytes()
        plan = {"recipe": fixture.recipe, "backends": ["tensorrt"], "device": "cpu", "output": fixture.output}

        def publish():
            manifest_path.write_text(json.dumps(manifest))
            variant["files"] = worker._inventory(manifest_path.parent, fixture.output / "model")
            release["variants"]["tensorrt"]["files"] = variant["files"]
            variant["checks"] = worker._file_identity(check_path, fixture.output)
            release_path.write_text(json.dumps(release))
            build["release"] = worker._file_identity(release_path, fixture.output)
            (fixture.output / "build.json").write_text(json.dumps(build))

        publish()
        cli._validate_container_completion(plan)
        for change in ("metric", "hash", "missing-policy", "operator", "count", "raw"):
            with self.subTest(change=change):
                check = json.loads(original_check)
                artifact = manifest["artifacts"][0]
                original_operators = list(artifact["required_operators"])
                native_path = fixture.output / "checks/tensorrt/case-0/output-0.bin"
                native_bytes = native_path.read_bytes()
                if change == "metric":
                    check["cases"][0]["outputs"][0]["max_abs"] = 1.0e-8
                elif change == "hash":
                    check["cases"][0]["outputs"][0]["actual_sha256"] = "f" * 64
                elif change == "missing-policy":
                    check.pop("require_byte_identical")
                elif change == "operator":
                    artifact["required_operators"].pop()
                elif change == "count":
                    artifact["correctness_profile"]["replacement_counts"]["exact_weighted_blend"] = 0
                else:
                    native_path.write_bytes(b"x" + native_bytes[1:])
                check_path.write_text(json.dumps(check))
                publish()
                with self.assertRaisesRegex(RuntimeError, "GeoTransolver"):
                    cli._validate_container_completion(plan)
                check_path.write_bytes(original_check)
                artifact["required_operators"] = original_operators
                artifact["correctness_profile"]["replacement_counts"]["exact_weighted_blend"] = 1
                native_path.write_bytes(native_bytes)


if __name__ == "__main__":
    unittest.main()
