"""DoMINO surface dependencies, graph selection and byte parity gates."""

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_builder.build import cli, worker
from model_builder.export import (
    tensorrt_exact_graphs as graphs,
    tensorrt_profiles as profiles,
)
import tensorrt_test_support as geo
import tensorrt_test_support
import worker_test_support


DOMINO = "domino-surface-exact"
PLUGINS = (
    "exact_linear",
    "exact_gelu",
    "exact_scalar_div",
    "exact_inverse_distance_blend",
)


class DominoProfileTests(unittest.TestCase):
    def fixture(self):
        helper = geo.ProfileBuildFixture()
        helper.addCleanup = self.addCleanup
        return helper

    def test_surface_profile_requires_only_four_declared_plugins(self):
        fixture = self.fixture().fixture()
        libraries = {}
        for name in PLUGINS:
            path = fixture.root / (name + ".so")
            path.write_bytes(name.encode())
            libraries[name] = path
        try:
            selected = profiles.resolve_plugin_libraries(DOMINO, libraries)
        except ValueError as error:
            self.fail(f"DoMINO surface profile must accept its four plugins: {error}")
        self.assertEqual(tuple(selected), PLUGINS)
        with self.assertRaisesRegex(ValueError, "four.*plugin"):
            profiles.resolve_plugin_libraries(
                DOMINO, {k: v for k, v in libraries.items() if k != "exact_scalar_div"}
            )
        self.assertEqual(len(profiles.required_operators()), 8)
        self.assertEqual(len(profiles.required_operators("geotransolver-exact")), 9)
        self.assertEqual(
            profiles.required_operators(DOMINO)[-1],
            {
                "id": "pnmir.tensorrt-exact-inverse-distance-blend",
                "abi": "1",
            },
        )

    def test_surface_rewrite_covers_all_linear_and_following_gelu(self):
        replacements = {
            "_replace_linear_subgraphs": mock.Mock(return_value=140),
            "_replace_gelu_subgraphs": mock.Mock(return_value=112),
            "_replace_scalar_divs": mock.Mock(return_value=2),
            "_replace_inverse_distance_blends": mock.Mock(return_value=4),
        }
        model, onnx = object(), object()
        with mock.patch.multiple(graphs, create=True, **replacements):
            try:
                counts = profiles.prepare_exact_graph(onnx, model, profile=DOMINO)
            except ValueError as error:
                self.fail(f"DoMINO surface rewrites must be selectable: {error}")
            self.assertEqual(counts, dict(zip(PLUGINS, (140, 112, 2, 4), strict=True)))
            prefixes = replacements["_replace_linear_subgraphs"].call_args.kwargs[
                "bias_name_prefixes"
            ]
            self.assertEqual(
                prefixes,
                ("",),
                "The surface core requires exact Linear across all learned stages",
            )
            replacements["_replace_gelu_subgraphs"].assert_called_once_with(
                onnx,
                model,
                exact_linear_sources_only=True,
            )
            replacements["_replace_inverse_distance_blends"].return_value = 0
            with self.assertRaisesRegex(
                ValueError, "no supported exact_inverse_distance_blend"
            ):
                profiles.prepare_exact_graph(onnx, model, profile=DOMINO)

    def test_worker_passes_four_assets_without_geo_gate_pass(self):
        helper = self.fixture()
        fixture = helper.fixture()
        backend = helper.fixture(tensorrt_test_support.TensorRTRecipeFixture)
        modules = backend.backend_modules()
        libraries = {name: fixture.root / (name + ".so") for name in PLUGINS}
        prepared = {
            "model": object(),
            "cases": [()],
            "assets": {
                "tensorrt_" + name + "_plugin": path for name, path in libraries.items()
            },
        }
        with mock.patch.dict(sys.modules, modules):
            try:
                worker._build_backend(
                    "tensorrt",
                    prepared,
                    dict(fixture.recipe, tensorrt_profile=DOMINO),
                    "cuda",
                    fixture.root / "package",
                    fixture.root / "exported",
                )
            except ValueError as error:
                self.fail(f"Worker must forward the selected surface profile: {error}")
        build = modules["model_builder.export.tensorrt_builder"].build_tensorrt_package
        self.assertEqual(build.call_args.kwargs["plugin_libraries"], libraries)
        options = modules[
            "model_builder.export.onnx_exporter"
        ].export_onnx_model.call_args.kwargs["options"]
        self.assertEqual(options.onnx_passes, ())

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional Torch")
    def test_payload_records_scope_counts_and_four_plugin_identities(self):
        from types import SimpleNamespace

        fixture = self.fixture().fixture(tensorrt_test_support.TensorRTPayloadFixture)
        libraries, records = {}, {}
        for name in PLUGINS:
            path = fixture.root / (name + ".so")
            path.write_bytes(name.encode())
            libraries[name] = path
            records[name] = {
                "filename": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        fixture.onnx.load_model.return_value = SimpleNamespace(
            SerializeToString=lambda: b"rewritten-graph"
        )
        fixture.trt.OnnxParser.return_value.parse = mock.Mock(return_value=True)
        counts = dict(zip(PLUGINS, (140, 112, 2, 4), strict=True))
        with (
            mock.patch.object(
                fixture.module, "load_exact_plugins", return_value=([object()], records)
            ),
            mock.patch.object(
                fixture.module, "prepare_exact_graph", return_value=counts
            ),
        ):
            fixture.build(profile=DOMINO, plugin_libraries=libraries)
        artifact = json.loads((fixture.output / "model.json").read_text())["artifacts"][
            0
        ]
        metadata = artifact["correctness_profile"]
        self.assertEqual(metadata["version"], 1)
        self.assertEqual(metadata["name"], DOMINO)
        self.assertEqual(metadata["plugins"], list(PLUGINS))
        self.assertEqual(metadata["plugin_libraries"], records)
        self.assertEqual(metadata["replacement_counts"], counts)
        self.assertEqual(
            metadata["exact_linear_bias_name_prefixes"],
            list(profiles.DOMINO_LINEAR_PREFIXES),
        )
        self.assertIs(metadata["exact_gelu_after_exact_linear"], True)
        self.assertEqual(
            artifact["required_operators"], profiles.required_operators(DOMINO)
        )

    def test_native_gate_rejects_tiny_and_signed_zero_differences(self):
        helper = self.fixture()
        for options in ({"difference": True}, {"signed_zero": True}):
            with self.subTest(options=options):
                self.assertEqual(
                    helper.build("baseline", **options)[1]["status"], "complete"
                )
                with self.assertRaisesRegex(ValueError, "byte-identical"):
                    helper.build(DOMINO, **options)

    def test_aoti_v3_records_zero_limits_and_rejects_small_differences(self):
        helper = self.fixture()
        for difference in (False, True):
            with self.subTest(difference=difference):
                fixture = helper.fixture()
                fixture.recipe["aoti_profile"] = "aten-boundary-exact-v3"
                fixture.recipe_path.write_text(json.dumps(fixture.recipe))
                if difference:
                    ref = fixture.prepared["references"][0][0]
                    values = list(struct.unpack("<4f", ref["data"]))
                    values[0] += 1e-6
                    ref["data"] = struct.pack("<4f", *values)
                with (
                    mock.patch(
                        "model_builder.export.aoti_profiles.validate_aoti_profile",
                        side_effect=lambda name: name,
                    ),
                    mock.patch(
                        "model_builder.export.aoti_options.validate_aoti_options",
                        return_value={},
                    ),
                ):
                    if difference:
                        with self.assertRaisesRegex(ValueError, "byte-identical"):
                            fixture.run_build(["aoti"])
                    else:
                        fixture.run_build(["aoti"])
                        check = json.loads(
                            (fixture.output / "checks/aoti.json").read_text()
                        )
                        self.assertTrue(
                            check.get("require_byte_identical"),
                            "AOTI v3 must require byte identity",
                        )
                        self.assertEqual(
                            check["limits"], {"max_abs": 0.0, "relative_l2": 0.0}
                        )

    def test_completion_checks_byte_policy_and_all_linear_scope(self):
        fixture, build = self.fixture().build(DOMINO)
        check_path = fixture.output / "checks/tensorrt.json"
        check = json.loads(check_path.read_text())
        self.assertTrue(
            check.get("require_byte_identical"),
            "DoMINO must record its strict byte policy",
        )
        self.assertEqual(check["limits"], {"max_abs": 0.0, "relative_l2": 0.0})
        variant = build["variants"]["tensorrt"]
        variant["graph"]["entrypoint"] = "model.onnx"
        release_path = fixture.output / "model/model-release.json"
        release = json.loads(release_path.read_text())
        manifest_path = fixture.output / "model" / variant["package"] / "model.json"
        manifest = json.loads(manifest_path.read_text())
        correctness = {
            "name": DOMINO,
            "version": 1,
            "plugins": list(PLUGINS),
            "replacement_counts": dict(zip(PLUGINS, (140, 112, 2, 4), strict=True)),
            "plugin_libraries": {
                name: {
                    "filename": name + ".so",
                    "sha256": hashlib.sha256(name.encode()).hexdigest(),
                }
                for name in PLUGINS
            },
            "exact_linear_bias_name_prefixes": list(profiles.DOMINO_LINEAR_PREFIXES),
            "exact_gelu_after_exact_linear": True,
        }
        manifest["artifacts"][0].update(
            correctness_profile=correctness,
            required_operators=profiles.required_operators(DOMINO),
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
                fixture.output, build, release, "tensorrt", check_path=check_path
            )

        publish()
        cli._validate_container_completion(plan)
        original = check_path.read_bytes()
        for change in ("metric", "scope", "policy", "raw"):
            with self.subTest(change=change):
                check = json.loads(original)
                native_path = fixture.output / "checks/tensorrt/case-0/output-0.bin"
                native = native_path.read_bytes()
                if change == "metric":
                    check["cases"][0]["outputs"][0]["max_abs"] = 1e-8
                elif change == "scope":
                    correctness["exact_linear_bias_name_prefixes"] = ["nn_basis."]
                elif change == "policy":
                    check.pop("require_byte_identical")
                else:
                    native_path.write_bytes(b"x" + native[1:])
                check_path.write_text(json.dumps(check))
                publish()
                with self.assertRaisesRegex(RuntimeError, "DoMINO"):
                    cli._validate_container_completion(plan)
                correctness["exact_linear_bias_name_prefixes"] = list(
                    profiles.DOMINO_LINEAR_PREFIXES
                )
                check_path.write_bytes(original)
                native_path.write_bytes(native)


if __name__ == "__main__":
    unittest.main()
