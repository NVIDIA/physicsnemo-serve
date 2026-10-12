"""GeoTransolver exact profile, captured dependencies and byte parity gates."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tensorrt_test_support import ProfileBuildFixture
from model_builder.build import cli, worker
from model_builder.export import (
    tensorrt_exact_graphs as graphs,
    tensorrt_profiles as profiles,
)
import tensorrt_test_support
import worker_test_support


GEO = "geotransolver-exact"
PLUGINS = (*profiles.EXACT_PLUGIN_NAMES, "exact_weighted_blend")


class GeoTransolverProfileTests(ProfileBuildFixture, unittest.TestCase):
    def test_geo_profile_requires_ninth_library_and_preserves_transolver_contract(self):
        fixture = self.fixture(tensorrt_test_support.TensorRTPluginFixture)
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
        self.assertEqual(
            operators[-1], {"id": "pnmir.tensorrt-exact-weighted-blend", "abi": "1"}
        )

    def test_geo_graph_adds_blend_rewrite_and_requires_a_match(self):
        calls = []
        symbols = dict(tensorrt_test_support.TRANSFORMS)
        symbols["exact_weighted_blend"] = "_replace_weighted_blends"
        replacements = {
            symbol: mock.Mock(
                side_effect=lambda onnx, model, name=name: calls.append(name) or 1
            )
            for name, symbol in symbols.items()
        }
        with mock.patch.multiple(graphs, create=True, **replacements):
            counts = profiles.prepare_exact_graph(object(), object(), profile=GEO)
            self.assertEqual(set(counts), set(PLUGINS))
            self.assertEqual(calls[-1], "exact_weighted_blend")
            replacements["_replace_weighted_blends"].side_effect = None
            replacements["_replace_weighted_blends"].return_value = 0
            with self.assertRaisesRegex(
                ValueError, "no supported exact_weighted_blend"
            ):
                profiles.prepare_exact_graph(object(), object(), profile=GEO)

    def test_worker_captures_ninth_plugin_and_enables_gate_pass_only_for_geo(self):
        helper = self.fixture(tensorrt_test_support.TensorRTRecipeFixture)
        fixture = self.fixture()
        libraries = {name: fixture.root / (name + ".so") for name in PLUGINS}
        assets = {
            "tensorrt_" + name + "_plugin": path for name, path in libraries.items()
        }
        modules = helper.backend_modules()
        exporter = modules["model_builder.export.onnx_exporter"].export_onnx_model
        build = modules["model_builder.export.tensorrt_builder"].build_tensorrt_package
        with mock.patch.dict(sys.modules, modules):
            worker._build_backend(
                "tensorrt",
                {"model": object(), "cases": [()], "assets": assets},
                dict(fixture.recipe, tensorrt_profile=GEO),
                "cuda",
                fixture.root / "package",
                fixture.root / "exported",
            )
        self.assertEqual(build.call_args.kwargs["plugin_libraries"], libraries)
        self.assertEqual(build.call_args.kwargs["profile"], GEO)
        passes = exporter.call_args.kwargs["options"].onnx_passes
        self.assertEqual(
            [type(value).__name__ for value in passes], ["FreezeScalarSigmoidGates"]
        )

    def test_ninth_plugin_is_captured_and_changes_fail_the_project_lock(self):
        helper = self.fixture(tensorrt_test_support.TensorRTRecipeFixture)
        fixture = helper.exact_project()
        fixture.document["tensorrt_profile"] = GEO
        path = fixture.project / "exact_weighted_blend.so"
        path.write_bytes(b"original weighted blend library")
        fixture.document["assets"]["tensorrt_exact_weighted_blend_plugin"] = path.name
        fixture.write_project()
        code, result, _, _ = fixture.invoke(fixture.successful_process)
        self.assertEqual(code, 0, result)
        recipe = json.loads(
            (fixture.output / "source/effective-recipe.json").read_text()
        )
        asset = recipe["assets"]["tensorrt_exact_weighted_blend_plugin"]
        self.assertEqual(
            (fixture.output / "source" / asset["path"]).read_bytes(), path.read_bytes()
        )
        identity = json.loads((fixture.output / "check.json").read_text())[
            "input_identity"
        ]
        self.assertEqual(identity["tensorrt_profile"], GEO)
        from model_builder.build import project_lock

        lock_path = fixture.project / "model-build.lock.json"
        project_lock.publish_lock(
            lock_path, project_lock.inspect_lock(lock_path, "build", identity)
        )
        path.write_bytes(b"changed weighted blend library")
        code, result, _, run = fixture.invoke(fixture.successful_process, "build")
        self.assertEqual(code, 2, result)
        self.assertEqual(result["diagnostics"][0]["code"], "PROJECT_LOCK_MISMATCH")
        run.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional Torch")
    def test_published_geo_payload_names_all_nine_verified_libraries(self):
        from types import SimpleNamespace

        fixture = self.fixture(tensorrt_test_support.TensorRTPayloadFixture)
        libraries, records = {}, {}
        for name in PLUGINS:
            path = fixture.root / (name + ".so")
            path.write_bytes((name + " test native bytes").encode())
            libraries[name] = path
            records[name] = {
                "filename": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        model = SimpleNamespace(SerializeToString=lambda: b"rewritten-onnx")
        fixture.onnx.load_model.return_value = model
        fixture.trt.OnnxParser.return_value.parse = mock.Mock(return_value=True)
        with (
            mock.patch.object(
                fixture.module, "load_exact_plugins", return_value=([object()], records)
            ) as load,
            mock.patch.object(
                fixture.module,
                "prepare_exact_graph",
                return_value={name: 1 for name in PLUGINS},
            ) as rewrite,
        ):
            fixture.build(profile=GEO, plugin_libraries=libraries)
        load.assert_called_once_with(
            fixture.trt, {name: path.resolve() for name, path in libraries.items()}, GEO
        )
        rewrite.assert_called_once_with(fixture.onnx, model, GEO)
        artifact = json.loads((fixture.output / "model.json").read_text())["artifacts"][
            0
        ]
        correctness = artifact["correctness_profile"]
        self.assertEqual(correctness["version"], 2)
        self.assertEqual(correctness["name"], GEO)
        self.assertEqual(correctness["plugins"], list(PLUGINS))
        self.assertEqual(correctness["plugin_libraries"], records)
        self.assertEqual(
            artifact["required_operators"], profiles.required_operators(GEO)
        )

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
                "name": GEO,
                "version": 2,
                "plugins": list(PLUGINS),
                "replacement_counts": {name: 1 for name in PLUGINS},
                "plugin_libraries": {
                    name: {"filename": name + ".so", "sha256": "0" * 64}
                    for name in PLUGINS
                },
            },
            required_operators=[
                {"id": "pnmir.tensorrt-" + name.replace("_", "-"), "abi": "1"}
                for name in PLUGINS
            ],
        )
        check_path = fixture.output / "checks/tensorrt.json"
        original_check = check_path.read_bytes()
        plan = {
            "recipe": fixture.recipe,
            "backends": ["tensorrt"],
            "device": "cpu",
            "output": fixture.output,
        }

        def publish():
            manifest_path.write_text(json.dumps(manifest))
            worker_test_support.publish_build_receipts(
                fixture.output, build, release, "tensorrt", check_path=check_path
            )

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
                    artifact["correctness_profile"]["replacement_counts"][
                        "exact_weighted_blend"
                    ] = 0
                else:
                    native_path.write_bytes(b"x" + native_bytes[1:])
                check_path.write_text(json.dumps(check))
                publish()
                with self.assertRaisesRegex(RuntimeError, "GeoTransolver"):
                    cli._validate_container_completion(plan)
                check_path.write_bytes(original_check)
                artifact["required_operators"] = original_operators
                artifact["correctness_profile"]["replacement_counts"][
                    "exact_weighted_blend"
                ] = 1
                native_path.write_bytes(native_bytes)


GEO_V2 = "geotransolver-exact-v2"
GEO_V2_PLUGINS = (
    *profiles.EXACT_PLUGIN_NAMES,
    "exact_weighted_blend",
    "exact_deslice_bmm",
)


class GeoTransolverV2ProfileTests(unittest.TestCase):
    def fixture(self, kind):
        fixture = kind()
        fixture.addCleanup = self.addCleanup
        if hasattr(fixture, "setUp"):
            fixture.setUp()
        return fixture

    def build(self, **options):
        helper = self.fixture(ProfileBuildFixture)
        return helper.build(GEO_V2, **options)

    def test_tenth_library_is_required_without_changing_legacy_contract(self):
        helper = self.fixture(tensorrt_test_support.TensorRTPluginFixture)
        libraries = dict(helper.libraries)
        for name in ("exact_weighted_blend", "exact_deslice_bmm"):
            path = helper.root / (name + ".dll")
            path.write_bytes((name + " native library").encode())
            libraries[name] = path
        self.assertEqual(profiles.validate_tensorrt_profile(GEO_V2), GEO_V2)
        self.assertEqual(
            tuple(profiles.resolve_plugin_libraries(GEO_V2, libraries)), GEO_V2_PLUGINS
        )
        self.assertEqual(profiles.profile_version(GEO_V2), 3)
        self.assertTrue(profiles.requires_byte_identical(GEO_V2))
        legacy = {
            name: path
            for name, path in libraries.items()
            if name != "exact_deslice_bmm"
        }
        with self.assertRaisesRegex(ValueError, "ten.*plugin"):
            profiles.resolve_plugin_libraries(GEO_V2, legacy)
        self.assertEqual(
            tuple(profiles.resolve_plugin_libraries("geotransolver-exact", legacy)),
            GEO_V2_PLUGINS[:-1],
        )
        self.assertEqual(profiles.profile_version("geotransolver-exact"), 2)
        with self.assertRaisesRegex(ValueError, "nine.*plugin"):
            profiles.resolve_plugin_libraries("geotransolver-exact", libraries)
        registry = SimpleNamespace(get_creator=mock.Mock(return_value=object()))
        trt = SimpleNamespace(get_plugin_registry=lambda: registry)
        with mock.patch.object(profiles.ctypes, "CDLL"):
            handles, records = profiles.load_exact_plugins(trt, libraries, GEO_V2)
        self.assertEqual(len(handles), 10)
        self.assertEqual(
            registry.get_creator.call_args_list[-1],
            mock.call("PNMIRExactDesliceBmm", "1", ""),
        )
        self.assertEqual(
            records["exact_deslice_bmm"]["sha256"],
            hashlib.sha256(libraries["exact_deslice_bmm"].read_bytes()).hexdigest(),
        )
        self.assertEqual(
            profiles.required_operators(GEO_V2)[-1],
            {"id": "pnmir.tensorrt-exact-deslice-bmm", "abi": "1"},
        )

    def test_deslice_requires_attention_and_weighted_blend_before_rewrite(self):
        calls = []
        symbols = dict(
            tensorrt_test_support.TRANSFORMS,
            exact_weighted_blend="_replace_weighted_blends",
            exact_deslice_bmm="_replace_deslice_bmms",
        )
        replacements = {
            symbol: mock.Mock(
                side_effect=lambda onnx, model, name=name: calls.append(name) or 1
            )
            for name, symbol in symbols.items()
        }
        with mock.patch.multiple(graphs, **replacements):
            counts = profiles.prepare_exact_graph(object(), object(), GEO_V2)
            self.assertEqual(set(counts), set(GEO_V2_PLUGINS))
            self.assertLess(
                calls.index("exact_attention"), calls.index("exact_weighted_blend")
            )
            self.assertLess(
                calls.index("exact_weighted_blend"), calls.index("exact_deslice_bmm")
            )
            replacements["_replace_deslice_bmms"].side_effect = None
            replacements["_replace_deslice_bmms"].return_value = 0
            with self.assertRaisesRegex(ValueError, "no supported exact_deslice_bmm"):
                profiles.prepare_exact_graph(object(), object(), GEO_V2)
            replacements["_replace_deslice_bmms"].reset_mock()
            profiles.prepare_exact_graph(object(), object(), "geotransolver-exact")
            replacements["_replace_deslice_bmms"].assert_not_called()

    def test_worker_preserves_scalar_sigmoid_gate_pass_and_captures_all_plugins(self):
        helper = self.fixture(tensorrt_test_support.TensorRTRecipeFixture)
        fixture = self.fixture(worker_test_support.WorkerFixture)
        modules = helper.backend_modules()
        libraries = {name: Path(name + ".dll") for name in GEO_V2_PLUGINS}
        assets = {
            "tensorrt_" + name + "_plugin": path for name, path in libraries.items()
        }
        with mock.patch.dict(sys.modules, modules):
            worker._build_backend(
                "tensorrt",
                {"model": object(), "cases": [()], "assets": assets},
                dict(fixture.recipe, tensorrt_profile=GEO_V2),
                "cuda",
                fixture.root / "package",
                fixture.root / "exported",
            )
        exported = modules["model_builder.export.onnx_exporter"].export_onnx_model
        passes = exported.call_args.kwargs["options"].onnx_passes
        self.assertEqual(
            [type(value).__name__ for value in passes], ["FreezeScalarSigmoidGates"]
        )
        built = modules["model_builder.export.tensorrt_builder"].build_tensorrt_package
        self.assertEqual(built.call_args.kwargs["plugin_libraries"], libraries)
        self.assertEqual(built.call_args.kwargs["profile"], GEO_V2)

    def test_deslice_asset_is_captured_and_locked(self):
        helper = self.fixture(tensorrt_test_support.TensorRTRecipeFixture)
        fixture = helper.exact_project()
        fixture.document["tensorrt_profile"] = GEO_V2
        for name in ("exact_weighted_blend", "exact_deslice_bmm"):
            path = fixture.project / (name + ".dll")
            path.write_bytes((name + " native library").encode())
            fixture.document["assets"]["tensorrt_" + name + "_plugin"] = path.name
        fixture.write_project()
        code, result, _, _ = fixture.invoke(fixture.successful_process)
        self.assertEqual(code, 0, result)
        recipe = json.loads(
            (fixture.output / "source/effective-recipe.json").read_text()
        )
        asset = recipe["assets"]["tensorrt_exact_deslice_bmm_plugin"]
        self.assertEqual(
            (fixture.output / "source" / asset["path"]).read_bytes(), path.read_bytes()
        )
        identity = json.loads((fixture.output / "check.json").read_text())[
            "input_identity"
        ]
        from model_builder.build import project_lock

        lock = fixture.project / "model-build.lock.json"
        project_lock.publish_lock(
            lock, project_lock.inspect_lock(lock, "build", identity)
        )
        path.write_bytes(b"changed deslice library")
        code, result, _, run = fixture.invoke(fixture.successful_process, "build")
        self.assertEqual(code, 2, result)
        self.assertEqual(result["diagnostics"][0]["code"], "PROJECT_LOCK_MISMATCH")
        run.assert_not_called()

    def test_native_gate_rejects_small_numeric_and_signed_zero_differences(self):
        for options in ({"difference": True}, {"signed_zero": True}):
            with (
                self.subTest(options=options),
                self.assertRaisesRegex(ValueError, "byte-identical"),
            ):
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
        artifact.update(
            correctness_profile={
                "name": GEO_V2,
                "version": 3,
                "plugins": list(GEO_V2_PLUGINS),
                "replacement_counts": {name: 1 for name in GEO_V2_PLUGINS},
                "plugin_libraries": {
                    name: {"filename": name + ".dll", "sha256": "0" * 64}
                    for name in GEO_V2_PLUGINS
                },
            },
            required_operators=profiles.required_operators(GEO_V2),
        )
        plan = {
            "recipe": fixture.recipe,
            "backends": ["tensorrt"],
            "device": "cpu",
            "output": fixture.output,
        }

        def publish():
            manifest_path.write_text(json.dumps(manifest))
            worker_test_support.publish_build_receipts(
                fixture.output, build, release, "tensorrt"
            )

        publish()
        cli._validate_container_completion(plan)
        for change in ("operator", "count", "version"):
            with self.subTest(change=change):
                if change == "operator":
                    artifact["required_operators"].pop()
                elif change == "count":
                    artifact["correctness_profile"]["replacement_counts"][
                        "exact_deslice_bmm"
                    ] = 0
                else:
                    artifact["correctness_profile"]["version"] = 2
                publish()
                with self.assertRaisesRegex(RuntimeError, "GeoTransolver"):
                    cli._validate_container_completion(plan)
                artifact["required_operators"] = profiles.required_operators(GEO_V2)
                artifact["correctness_profile"]["replacement_counts"][
                    "exact_deslice_bmm"
                ] = 1
                artifact["correctness_profile"]["version"] = 3


if __name__ == "__main__":
    unittest.main()
