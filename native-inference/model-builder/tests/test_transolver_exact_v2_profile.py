"""Versioned deslice profile, dependency capture, and strict parity publication."""

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_builder.build import cli
from model_builder.export import (
    tensorrt_exact_graphs as graphs,
    tensorrt_profiles as profiles,
)
import tensorrt_test_support
import worker_test_support


PROFILE = "layout-order-exact-v2"
PLUGINS = (*profiles.EXACT_PLUGIN_NAMES, "exact_deslice_bmm")


class TransolverV2ProfileTests(unittest.TestCase):
    def fixture(self, kind):
        fixture = kind()
        fixture.addCleanup = self.addCleanup
        if hasattr(fixture, "setUp"):
            fixture.setUp()
        return fixture

    def build(self, **options):
        helper = self.fixture(tensorrt_test_support.ProfileBuildFixture)
        return helper.build(PROFILE, **options)

    def test_versioned_contract_preserves_existing_profiles(self):
        self.assertEqual(profiles.validate_tensorrt_profile(PROFILE), PROFILE)
        self.assertEqual(profiles.plugin_names(PROFILE), PLUGINS)
        self.assertEqual(profiles.profile_version(PROFILE), 2)
        self.assertTrue(profiles.requires_byte_identical(PROFILE))
        self.assertEqual(
            profiles.required_operators(PROFILE)[-1],
            {"id": "pnmir.tensorrt-exact-deslice-bmm", "abi": "1"},
        )
        self.assertEqual(
            profiles.plugin_names("layout-order-exact"), profiles.EXACT_PLUGIN_NAMES
        )
        self.assertEqual(profiles.profile_version("layout-order-exact"), 1)
        self.assertFalse(profiles.requires_byte_identical("layout-order-exact"))
        self.assertEqual(
            profiles.plugin_names("geotransolver-exact"),
            (*profiles.EXACT_PLUGIN_NAMES, "exact_weighted_blend"),
        )
        self.assertEqual(profiles.profile_version("geotransolver-exact"), 2)

    def test_deslice_library_is_required_and_creator_bytes_are_bound(self):
        helper = self.fixture(tensorrt_test_support.TensorRTPluginFixture)
        library = helper.root / "exact_deslice_bmm.dll"
        library.write_bytes(b"test deslice library")
        libraries = dict(helper.libraries, exact_deslice_bmm=library)
        self.assertEqual(
            tuple(profiles.resolve_plugin_libraries(PROFILE, libraries)), PLUGINS
        )
        with self.assertRaisesRegex(ValueError, "nine.*plugin"):
            profiles.resolve_plugin_libraries(PROFILE, helper.libraries)
        with self.assertRaisesRegex(ValueError, "eight.*plugin"):
            profiles.resolve_plugin_libraries("layout-order-exact", libraries)
        registry = SimpleNamespace(get_creator=mock.Mock(return_value=object()))
        trt = SimpleNamespace(get_plugin_registry=lambda: registry)
        with mock.patch.object(profiles.ctypes, "CDLL"):
            handles, records = profiles.load_exact_plugins(trt, libraries, PROFILE)
        self.assertEqual(len(handles), 9)
        self.assertEqual(
            registry.get_creator.call_args_list[-1],
            mock.call("PNMIRExactDesliceBmm", "1", ""),
        )
        self.assertEqual(
            records["exact_deslice_bmm"],
            {
                "filename": library.name,
                "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            },
        )

    def test_deslice_rewrite_requires_attention_and_a_match(self):
        calls = []
        symbols = dict(
            tensorrt_test_support.TRANSFORMS, exact_deslice_bmm="_replace_deslice_bmms"
        )
        replacements = {
            symbol: mock.Mock(
                side_effect=lambda onnx, model, name=name: calls.append(name) or 1
            )
            for name, symbol in symbols.items()
        }
        with mock.patch.multiple(graphs, **replacements):
            counts = profiles.prepare_exact_graph(object(), object(), PROFILE)
            self.assertEqual(set(counts), set(PLUGINS))
            self.assertLess(
                calls.index("exact_attention"), calls.index("exact_deslice_bmm")
            )
            replacements["_replace_deslice_bmms"].side_effect = None
            replacements["_replace_deslice_bmms"].return_value = 0
            with self.assertRaisesRegex(ValueError, "no supported exact_deslice_bmm"):
                profiles.prepare_exact_graph(object(), object(), PROFILE)
            # Existing Transolver/Geo profiles never acquire the new dependency.
            replacements["_replace_deslice_bmms"].reset_mock()
            profiles.prepare_exact_graph(object(), object(), "layout-order-exact")
            replacements["_replace_deslice_bmms"].assert_not_called()

    def test_deslice_asset_is_captured_and_bound_by_project_lock(self):
        helper = self.fixture(tensorrt_test_support.TensorRTRecipeFixture)
        fixture = helper.exact_project()
        fixture.document["tensorrt_profile"] = PROFILE
        path = fixture.project / "exact_deslice_bmm.dll"
        path.write_bytes(b"original deslice library")
        fixture.document["assets"]["tensorrt_exact_deslice_bmm_plugin"] = path.name
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
        self.assertEqual(identity["tensorrt_profile"], PROFILE)
        from model_builder.build import project_lock

        lock_path = fixture.project / "model-build.lock.json"
        project_lock.publish_lock(
            lock_path, project_lock.inspect_lock(lock_path, "build", identity)
        )
        path.write_bytes(b"changed deslice library")
        code, result, _, run = fixture.invoke(fixture.successful_process, "build")
        self.assertEqual(code, 2, result)
        self.assertEqual(result["diagnostics"][0]["code"], "PROJECT_LOCK_MISMATCH")
        run.assert_not_called()

    def test_v2_rejects_numeric_and_signed_zero_differences(self):
        for options in ({"difference": True}, {"signed_zero": True}):
            with (
                self.subTest(options=options),
                self.assertRaisesRegex(ValueError, "byte-identical"),
            ):
                self.build(**options)

    def test_success_requires_zero_limits_and_equal_hashes(self):
        fixture, _ = self.build()
        check = json.loads((fixture.output / "checks/tensorrt.json").read_text())
        self.assertTrue(check["require_byte_identical"])
        self.assertEqual(check["limits"], {"max_abs": 0.0, "relative_l2": 0.0})
        for case in check["cases"]:
            for output in case["outputs"]:
                self.assertEqual(output["actual_sha256"], output["reference_sha256"])

    def test_completion_rejects_missing_deslice_or_false_byte_evidence(self):
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
                "name": PROFILE,
                "version": 2,
                "plugins": list(PLUGINS),
                "replacement_counts": {name: 1 for name in PLUGINS},
                "plugin_libraries": {
                    name: {"filename": name + ".dll", "sha256": "0" * 64}
                    for name in PLUGINS
                },
            },
            required_operators=profiles.required_operators(PROFILE),
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
        for change in ("hash", "missing-policy", "operator", "count", "version"):
            with self.subTest(change=change):
                check = json.loads(original_check)
                if change == "hash":
                    check["cases"][0]["outputs"][0]["actual_sha256"] = "f" * 64
                elif change == "missing-policy":
                    check.pop("require_byte_identical")
                elif change == "operator":
                    artifact["required_operators"].pop()
                elif change == "count":
                    artifact["correctness_profile"]["replacement_counts"][
                        "exact_deslice_bmm"
                    ] = 0
                else:
                    artifact["correctness_profile"]["version"] = 1
                check_path.write_text(json.dumps(check))
                publish()
                with self.assertRaisesRegex(RuntimeError, "Transolver"):
                    cli._validate_container_completion(plan)
                check_path.write_bytes(original_check)
                artifact["required_operators"] = profiles.required_operators(PROFILE)
                artifact["correctness_profile"]["replacement_counts"][
                    "exact_deslice_bmm"
                ] = 1
                artifact["correctness_profile"]["version"] = 2


if __name__ == "__main__":
    unittest.main()
