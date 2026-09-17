"""Prepared GeoTransolver inputs work with the generic two-file authoring project."""

import json
from pathlib import Path
import sys
import unittest

import test_example as fixture

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "model-builder/src"))
from pnmir_build import authoring, authoring_config, authoring_worker  # noqa: E402


@unittest.skipIf(fixture.torch is None, "requires producer Torch")
class PreparedProjectTests(unittest.TestCase):
    def setUp(self):
        self.example = fixture.GeoTransolverExampleTests()
        self.example.setUp()
        self.addCleanup(self.example.doCleanups)
        self.project = self.example.root / "customer-project"

    def prepare(self):
        return fixture.preparation.prepare_project(
            self.example.checkpoint,
            self.project,
            points=8,
            geometry_points=16,
            device="cpu",
        )

    def test_staged_project_keeps_verified_adapter_and_prepared_assets(self):
        self.prepare()
        self.assertTrue((self.project / "model-build.json").is_file())
        self.assertTrue((self.project / "adapter.py").is_file())
        prepared = self.project / "prepared"
        self.assertEqual(
            (self.project / "adapter.py").read_bytes(),
            (prepared / "adapter.py").read_bytes(),
        )
        config = json.loads((self.project / "model-build.json").read_text())
        self.assertEqual(config["format_version"], 2)
        self.assertEqual(config["aoti_profile"], "aten-boundary-exact-v2")
        self.assertEqual(config["backends"], ["aoti", "tensorrt"])
        self.assertEqual(config["config"], "prepared/config.json")
        self.assertEqual(config["checkpoint"], "prepared/checkpoint.pt")
        self.assertEqual(config["assets"], {"fixtures": "prepared/fixtures.pt"})
        self.assertEqual(
            json.loads((prepared / "preparation.json").read_text())["status"],
            "complete",
        )

    def test_generic_authoring_worker_checks_all_cases_through_captured_adapter(self):
        self.prepare()
        self.assertTrue((self.project / "model-build.json").is_file())
        project = authoring_config.load(self.project)
        source, selected, configuration = authoring._selected(project)
        snapshot = self.example.root / "snapshot"
        authoring._snapshot(
            snapshot,
            project,
            source,
            selected,
            configuration,
            {"adapter": "adapter.py"},
        )
        result = authoring_worker.execute(
            snapshot, self.example.root / "checked", "check", "cpu"
        )
        self.assertEqual(result["status"], "checked")
        self.assertEqual(result["case_count"], 3)
        self.assertEqual(
            [x["name"] for x in result["tensor_contract"]["inputs"]],
            ["local_embedding", "local_features", "static_context", "global_embedding"],
        )
        self.assertEqual(
            result["tensor_contract"]["outputs"],
            [
                {
                    "name": "surface_fields_standardized",
                    "dtype": "float32",
                    "shape": [1, 8, 4],
                }
            ],
        )

    def test_existing_project_is_preserved(self):
        self.project.mkdir()
        sentinel = self.project / "keep.txt"
        sentinel.write_text("keep existing inputs")
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(sentinel.read_text(), "keep existing inputs")
        self.assertEqual(list(self.project.iterdir()), [sentinel])


if __name__ == "__main__":
    unittest.main()
