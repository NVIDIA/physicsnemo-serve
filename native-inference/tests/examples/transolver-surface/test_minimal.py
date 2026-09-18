"""The Transolver project preserves the checkpoint and raw workflow boundary."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "examples/transolver-surface"
CONFIG = {
    "functional_dim": 2, "out_dim": 4, "embedding_dim": 6,
    "n_layers": 20, "n_hidden": 256, "dropout": 0.0, "n_head": 8,
    "act": "gelu", "mlp_ratio": 2, "slice_num": 512,
    "unified_pos": False, "ref": 8, "structured_shape": None,
    "use_te": False, "time_input": False, "plus": False,
}


class MinimalExampleTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "transolver_adapter", PROJECT / "adapter.py"
        )
        self.adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.adapter)

    def test_project_selects_exact_aoti_and_the_workflow_tensor_names(self):
        project = json.loads((PROJECT / "model-build.json").read_text())
        self.assertEqual(project["aoti_profile"], "aten-boundary-exact-v2")
        self.assertEqual(project["backends"], ["aoti"])
        self.assertEqual(project["config"], "weights/config.json")
        self.assertEqual(project["checkpoint"], "weights/checkpoint.pt")
        self.assertEqual(project["input_names"], ["fx", "embedding"])
        self.assertEqual(project["output_names"], ["surface_fields_standardized"])

    def test_incompatible_surface_configuration_is_rejected_before_construction(self):
        module = types.ModuleType("physicsnemo.models.transolver")
        module.Transolver = mock.Mock()
        with mock.patch.dict(sys.modules, {module.__name__: module}):
            for key, value in (
                ("functional_dim", 3), ("embedding_dim", 7), ("out_dim", 5),
                ("unified_pos", True), ("structured_shape", [5, 15]),
                ("use_te", True), ("time_input", True), ("plus", True),
            ):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "surface"):
                    self.adapter.create_model(dict(CONFIG, **{key: value}), {})
        module.Transolver.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional Torch")
    def test_cases_preserve_workflow_features_and_do_not_change_global_rng(self):
        import torch

        state = torch.get_rng_state().clone()
        cases = self.adapter.create_cases(CONFIG, {})
        repeated = self.adapter.create_cases(CONFIG, {})
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(len(cases), 3)
        for (fx, embedding), repeat in zip(cases, repeated, strict=True):
            self.assertEqual(fx.shape, (1, 75, 2))
            self.assertEqual(embedding.shape, (1, 75, 6))
            self.assertTrue(torch.equal(fx, fx[:, :1, :].expand_as(fx)))
            self.assertTrue((fx > 0).all())
            torch.testing.assert_close(
                torch.linalg.vector_norm(embedding[..., 3:], dim=-1), torch.ones(1, 75)
            )
            for value, expected in zip((fx, embedding), repeat, strict=True):
                self.assertEqual(value.dtype, torch.float32)
                self.assertTrue(value.is_contiguous())
                self.assertTrue(torch.isfinite(value).all())
                self.assertTrue(torch.equal(value, expected))
        for index in (0, 1):
            self.assertFalse(torch.equal(cases[0][index], cases[1][index]))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional Torch")
    def test_imported_checkpoint_passes_generic_authoring_check_without_rekeying(self):
        import torch

        sys.path.insert(0, str(ROOT / "model-builder/src"))
        self.addCleanup(sys.path.remove, str(ROOT / "model-builder/src"))
        from pnmir_build import (
            authoring, authoring_config, authoring_sources, authoring_worker,
        )

        class TransolverDouble(torch.nn.Module):
            def __init__(self, **config):
                super().__init__()
                self.config = config
                self.preprocess = torch.nn.Linear(
                    config["functional_dim"] + config["embedding_dim"], config["out_dim"]
                )

            def forward(self, fx, embedding):
                return self.preprocess(torch.cat((fx, embedding), dim=-1))

        upstream = types.ModuleType("physicsnemo.models.transolver")
        upstream.Transolver = TransolverDouble
        with mock.patch.dict(sys.modules, {upstream.__name__: upstream}):
            official = TransolverDouble(**CONFIG).eval()
            adapted = self.adapter.create_model(CONFIG, {}).eval()
            self.assertEqual(adapted.config, CONFIG)
            adapted.load_state_dict(official.state_dict(), strict=True)
            self.assertEqual(set(adapted.state_dict()), set(official.state_dict()))
            with torch.no_grad():
                for case in self.adapter.create_cases(CONFIG, {}):
                    self.assertTrue(torch.equal(adapted(*case), official(*case)))

            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                project_dir = root / "project"
                weights = project_dir / "weights"
                weights.mkdir(parents=True)
                for name in ("adapter.py", "model-build.json"):
                    (project_dir / name).write_bytes((PROJECT / name).read_bytes())
                (weights / "config.json").write_text(json.dumps(CONFIG))
                torch.save(official.state_dict(), weights / "checkpoint.pt")
                project = authoring_config.load(project_dir)
                source, selected, configuration = authoring._selected(project)
                snapshot = root / "snapshot"
                authoring._snapshot(
                    snapshot, project, source, selected, configuration, {"adapter": "adapter.py"}
                )
                with (
                    mock.patch.object(authoring_sources, "_TREES", {}),
                    mock.patch.object(authoring_sources, "_IMPORT_ROOTS", set()),
                    mock.patch.object(sys, "path", list(sys.path)),
                    mock.patch.dict(sys.modules),
                ):
                    try:
                        result = authoring_worker.execute(snapshot, root / "checked", "check", "cpu")
                    finally:
                        for temporary_tree, _, _ in authoring_sources._TREES.values():
                            temporary_tree.cleanup()
        self.assertEqual(result["status"], "checked")
        self.assertEqual(result["case_count"], 3)
        contract = result["tensor_contract"]
        self.assertEqual([entry["shape"] for entry in contract["inputs"]], [[1, 75, 2], [1, 75, 6]])
        self.assertEqual(contract["outputs"][0]["shape"], [1, 75, 4])


if __name__ == "__main__":
    unittest.main()
