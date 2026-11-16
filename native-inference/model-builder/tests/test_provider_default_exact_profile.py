"""Provider math is an explicit, independently verified TensorRT profile."""

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_builder.export import tensorrt_profiles as profiles
from model_builder.export import tensorrt_exact_graphs as graphs
import tensorrt_test_support
import worker_test_support


PROFILE = "provider-default-exact-v1"
BLAS = {
    "exact_linear": "PNMIRExactLinear",
    "exact_gemm": "PNMIRExactGemm",
    "exact_slice_bmm": "PNMIRExactSliceBmm",
    "exact_deslice_bmm": "PNMIRExactDesliceBmm",
}


class ProviderDefaultExactProfileTests(unittest.TestCase):
    def framework(self):
        state = dict(
            precision="highest", deterministic=True, warn_only=True, blas="cublaslt"
        )
        cudnn = SimpleNamespace(allow_tf32=False, benchmark=True, deterministic=True)

        def deterministic(value, *, warn_only=False):
            state.update(deterministic=value, warn_only=warn_only)

        def blas(value=None):
            if value is not None:
                state["blas"] = value
            return state["blas"]

        framework = SimpleNamespace(
            cuda=SimpleNamespace(get_device_capability=lambda device: (9, 0)),
            backends=SimpleNamespace(
                cudnn=cudnn, cuda=SimpleNamespace(preferred_blas_library=blas)
            ),
            get_float32_matmul_precision=lambda: state["precision"],
            set_float32_matmul_precision=lambda value: state.update(precision=value),
            are_deterministic_algorithms_enabled=lambda: state["deterministic"],
            is_deterministic_algorithms_warn_only_enabled=lambda: state["warn_only"],
            use_deterministic_algorithms=deterministic,
        )
        return framework, state, cudnn

    def test_provider_math_restores_caller_on_success_and_failure(self):
        for fail in (False, True):
            framework, state, cudnn = self.framework()
            original = dict(state)
            with (
                mock.patch.dict(sys.modules, torch=framework),
                mock.patch.dict(os.environ, {}, clear=True),
            ):
                try:
                    with profiles.provider_default_math("cuda:0"):
                        self.assertEqual(
                            state,
                            dict(
                                precision="high",
                                deterministic=False,
                                warn_only=False,
                                blas="cublas",
                            ),
                        )
                        self.assertTrue(cudnn.allow_tf32)
                        self.assertFalse(cudnn.benchmark)
                        self.assertFalse(cudnn.deterministic)
                        if fail:
                            raise RuntimeError("intentional export failure")
                except RuntimeError as error:
                    self.assertEqual(str(error), "intentional export failure")
            self.assertEqual(state, original)
            self.assertEqual(
                vars(cudnn), dict(allow_tf32=False, benchmark=True, deterministic=True)
            )

    def test_provider_math_rejects_different_workspace_before_changing_state(self):
        for environment in (
            {"CUBLAS_WORKSPACE_CONFIG": ":4096:2"},
            {"CUBLASLT_WORKSPACE_SIZE": "2048"},
            {"TORCH_CUBLASLT_UNIFIED_WORKSPACE": "1"},
        ):
            framework, state, _ = self.framework()
            original = dict(state)
            with (
                mock.patch.dict(sys.modules, torch=framework),
                mock.patch.dict(os.environ, environment, clear=True),
            ):
                with self.assertRaisesRegex(ValueError, "workspace"):
                    with profiles.provider_default_math("cuda:0"):
                        self.fail("different workspace entered provider inference")
            self.assertEqual(state, original)

    def test_provider_profile_rejects_mixed_backends_before_export(self):
        from model_builder.build import worker

        fixture = worker_test_support.WorkerFixture()
        fixture.addCleanup = self.addCleanup
        fixture.setUp()
        fixture.recipe["tensorrt_profile"] = PROFILE
        fixture.recipe_path.write_text(json.dumps(fixture.recipe))
        with self.assertRaisesRegex(ValueError, "TensorRT-only"):
            worker._read_recipe(fixture.recipe_path, ["aoti", "tensorrt"])

    def test_frontend_rejects_mixed_backends_before_container_launch(self):
        from model_builder.build import cli

        fixture = worker_test_support.WorkerFixture()
        fixture.addCleanup = self.addCleanup
        fixture.setUp()
        fixture.recipe["tensorrt_profile"] = PROFILE
        fixture.recipe_path.write_text(json.dumps(fixture.recipe))
        args = cli.parser().parse_args(
            [
                "build",
                "--recipe",
                str(fixture.recipe_path),
                "--backend",
                "aoti",
                "--backend",
                "tensorrt",
                "--device",
                "cuda",
                "--executor",
                "local",
                "--runtime",
                str(fixture.runtime),
            ]
        )
        with self.assertRaisesRegex(cli.UsageError, "TensorRT-only"):
            cli._resolve_selection(args)

    def test_profile_selects_provider_blas_and_retains_ieee_contract(self):
        self.assertEqual(profiles.validate_tensorrt_profile(PROFILE), PROFILE)
        self.assertEqual(
            profiles.plugin_names(PROFILE), profiles.TRANSOLVER_V2_PLUGIN_NAMES
        )
        self.assertTrue(profiles.requires_byte_identical(PROFILE))
        metadata = profiles.selection_metadata(PROFILE)["math_policy"]
        self.assertEqual(metadata["name"], "provider-default-v1")
        self.assertEqual(metadata["float32_matmul_precision"], "high")
        self.assertTrue(metadata["allow_tf32_matmul"])
        self.assertEqual(metadata["cublas_workspace_bytes"], 32 * 1024 * 1024)
        self.assertEqual(metadata["cublaslt_workspace_bytes"], 1024 * 1024)
        old = profiles.required_operators("layout-order-exact-v2")
        new = profiles.required_operators(PROFILE)
        for name, original, selected in zip(profiles.plugin_names(PROFILE), old, new):
            if name in BLAS:
                self.assertEqual(
                    selected["id"],
                    "pnmir.tensorrt-provider-default-" + name.replace("_", "-"),
                )
                self.assertNotEqual(selected["id"], original["id"])
            else:
                self.assertEqual(selected, original)
        self.assertEqual(profiles.selection_metadata("layout-order-exact-v2"), {})
        self.assertEqual(profiles.profile_version("layout-order-exact-v2"), 2)

    def test_graph_serializes_math_and_workspace_only_into_blas_v2(self):
        def attribute(name, value):
            return SimpleNamespace(name=name, value=value)

        model = SimpleNamespace(
            graph=SimpleNamespace(
                node=[
                    SimpleNamespace(
                        op_type=name,
                        attribute=[attribute("plugin_version", "1")],
                    )
                    for name in [*BLAS.values(), "PNMIRExactGelu"]
                ]
            )
        )
        transforms = dict(
            tensorrt_test_support.TRANSFORMS,
            exact_deslice_bmm="_replace_deslice_bmms",
        )
        with mock.patch.multiple(
            graphs,
            **{symbol: mock.Mock(return_value=1) for symbol in transforms.values()},
        ):
            counts = profiles.prepare_exact_graph(
                SimpleNamespace(helper=SimpleNamespace(make_attribute=attribute)),
                model,
                PROFILE,
            )
        self.assertEqual(set(counts), set(profiles.TRANSOLVER_V2_PLUGIN_NAMES))
        for node in model.graph.node:
            values = {field.name: field.value for field in node.attribute}
            if node.op_type in BLAS.values():
                self.assertEqual(values["plugin_version"], "2")
                self.assertEqual(values["compute_mode"], 1)
                self.assertEqual(values["cublas_workspace_bytes"], 33554432)
                self.assertEqual(values["cublaslt_workspace_bytes"], 1048576)
            else:
                self.assertEqual(values, {"plugin_version": "1"})

    def test_provider_profile_keeps_byte_identical_publication_gate(self):
        fixture = tensorrt_test_support.ProfileBuildFixture()
        fixture.addCleanup = self.addCleanup
        with self.assertRaisesRegex(ValueError, "byte-identical"):
            fixture.build(PROFILE, difference=True)
        built, _ = fixture.build(PROFILE)
        check = json.loads((built.output / "checks/tensorrt.json").read_text())
        self.assertEqual(check["limits"], {"max_abs": 0.0, "relative_l2": 0.0})


if __name__ == "__main__":
    unittest.main()
