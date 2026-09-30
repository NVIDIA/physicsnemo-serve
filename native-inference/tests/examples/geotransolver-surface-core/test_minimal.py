"""The customer example consumes ordinary imported weights without preparation."""

import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import test_example as fixture

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "examples/geotransolver-surface-core"
sys.path.insert(0, str(ROOT / "model-builder/src"))
from pnmir_build import (  # noqa: E402
    authoring,
    authoring_config,
    authoring_sources,
    authoring_worker,
)


@unittest.skipIf(fixture.torch is None, "requires Torch")
class MinimalExampleTests(unittest.TestCase):
    def test_customer_project_selects_exact_backends_and_ten_plugin_assets(self):
        document = json.loads((EXAMPLE / "model-build.json").read_text())
        self.assertEqual(document.get("tensorrt_profile"), "geotransolver-exact-v2")
        self.assertEqual(document["aoti_profile"], "aten-boundary-exact-v2")
        names = {
            "linear",
            "gemm",
            "token_sum",
            "slice_bmm",
            "layer_norm",
            "softmax",
            "attention",
            "gelu",
            "weighted_blend",
            "deslice_bmm",
        }
        self.assertEqual(
            set(document["assets"]), {f"tensorrt_exact_{name}_plugin" for name in names}
        )

    def setUp(self):
        self.fixture = fixture.GeoTransolverExampleTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        spec = importlib.util.spec_from_file_location(
            "minimal_geo", EXAMPLE / "adapter.py"
        )
        self.adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.adapter)
        self.config = self.fixture.model_args

    def model(self):
        try:
            return self.adapter.create_model(self.config, {})
        except ValueError as exc:
            self.fail(
                f"Plain imported constructor config must work without fixtures: {exc}"
            )

    def test_plain_imported_constructor_and_weights_preserve_full_model_core(self):
        torch = fixture.torch
        core = self.model().eval()
        full = self.fixture.official
        core.load_state_dict(full.state_dict(), strict=True)
        self.assertEqual(set(core.state_dict()), set(full.state_dict()))
        with torch.no_grad():
            for index in range(3):
                raw = fixture.preparation._full_inputs(8, 16, index, "cpu")
                cached = fixture.preparation._cached_inputs(full, raw)
                self.assertTrue(torch.equal(core(*cached), full(*raw)))

    def test_cases_are_repeatable_and_have_the_imported_surface_dimensions(self):
        torch = fixture.torch
        state = torch.get_rng_state().clone()
        try:
            cases = self.adapter.create_cases(self.config, {})
        except ValueError as exc:
            self.fail(f"Synthetic core cases must not require prepared assets: {exc}")
        self.assertEqual(len(cases), 3)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        again = self.adapter.create_cases(self.config, {})
        for case, repeated in zip(cases, again, strict=True):
            self.assertEqual(
                [list(x.shape) for x in case],
                [[1, 32, 6], [1, 32, 192], [1, 8, 128, 224], [1, 1, 2]],
            )
            for value, other in zip(case, repeated, strict=True):
                self.assertEqual(value.dtype, torch.float32)
                self.assertTrue(torch.isfinite(value).all())
                self.assertTrue(torch.equal(value, other))
        for column in range(4):
            self.assertFalse(torch.equal(cases[0][column], cases[1][column]))

    def test_generic_authoring_check_uses_only_imported_config_and_checkpoint(self):
        torch = fixture.torch
        # Same upstream API double with dimensions matching its tiny layers.
        self.config.update(
            n_hidden=4,
            n_head=2,
            slice_num=3,
            n_hidden_local=2,
            radii=[0.1],
            neighbors_in_radius=[4],
        )
        project_dir = self.fixture.root / "project"
        project_dir.mkdir()
        for name in ("adapter.py", "model-build.json"):
            (project_dir / name).write_bytes((EXAMPLE / name).read_bytes())
        weights = project_dir / "weights"
        weights.mkdir()
        (weights / "config.json").write_text(json.dumps(self.config))
        torch.save(self.fixture.official.state_dict(), weights / "checkpoint.pt")
        document = json.loads((project_dir / "model-build.json").read_text())
        self.assertEqual(document["config"], "weights/config.json")
        self.assertEqual(document["checkpoint"], "weights/checkpoint.pt")
        self.assertNotIn("fixtures", document["assets"])
        # Check captures SDK library identities but does not load native plugins.
        for name, relative in document["assets"].items():
            path = project_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"SDK plugin identity fixture: {name}".encode())
        project = authoring_config.load(project_dir)
        source, selected, configuration = authoring._selected(project)
        snapshot = self.fixture.root / "snapshot"
        authoring._snapshot(
            snapshot,
            project,
            source,
            selected,
            configuration,
            {"adapter": "adapter.py"},
        )
        # Production uses one worker process per project; isolate its imports here.
        with (
            mock.patch.object(authoring_sources, "_TREES", {}),
            mock.patch.object(authoring_sources, "_IMPORT_ROOTS", set()),
            mock.patch.object(sys, "path", list(sys.path)),
            mock.patch.dict(sys.modules),
        ):
            try:
                result = authoring_worker.execute(
                    snapshot, self.fixture.root / "checked", "check", "cpu"
                )
            finally:
                for temporary, _, _ in authoring_sources._TREES.values():
                    temporary.cleanup()
        self.assertEqual(result["status"], "checked")
        self.assertEqual(result["case_count"], 3)
        self.assertEqual(result["tensor_contract"]["outputs"][0]["shape"], [1, 32, 4])

    def test_incompatible_surface_configuration_is_rejected(self):
        self.model()
        for key, value in (
            ("functional_dim", [6, 6]),
            ("include_local_features", False),
            ("use_te", True),
            ("guard_config", {"enabled": True}),
            ("structured_shape", [4, 8]),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "surface"):
                self.adapter.create_model(dict(self.config, **{key: value}), {})


if __name__ == "__main__":
    unittest.main()
