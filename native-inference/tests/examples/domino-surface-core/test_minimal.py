"""The DoMINO external example stays small and supplies repeatable valid inputs."""

import importlib.util
import json
from pathlib import Path
import unittest

PROJECT = Path(__file__).resolve().parents[3] / "examples/domino-surface-core"
CONFIG = {
    "input_features": 3, "output_features_vol": None,
    "output_features_surf": 4, "global_features": 2,
    "model_parameters": {
        "model_type": "surface", "num_neighbors_surface": 7,
        "combine_volume_surface": False, "encode_parameters": False,
        "geometry_encoding_type": "both", "use_surface_normals": True,
        "use_surface_area": True,
        "geometry_rep": {"geo_conv": {"surface_radii": [.01, .05, 1.0]}},
        "geometry_local": {"surface_neighbors_in_radius": [32, 128]},
    },
}


class MinimalExampleTests(unittest.TestCase):
    def test_external_project_selects_both_exact_backends_and_only_two_files(self):
        self.assertTrue(PROJECT.is_dir(), "Add a minimal external DoMINO example")
        self.assertEqual({p.name for p in PROJECT.iterdir()}, {"adapter.py", "model-build.json"})
        project = json.loads((PROJECT / "model-build.json").read_text())
        self.assertEqual(project["backends"], ["aoti", "tensorrt"])
        self.assertEqual(project["aoti_profile"], "aten-boundary-exact-v3")
        self.assertEqual(project["tensorrt_profile"], "domino-surface-exact")
        self.assertEqual(len(project["assets"]), 5)
        self.assertEqual(project["checkpoint"], "weights/checkpoint.pt")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional Torch")
    def test_cases_have_positive_areas_and_distinct_repeatable_inputs(self):
        import torch

        self.assertTrue((PROJECT / "adapter.py").is_file(), "DoMINO needs an adapter")
        spec = importlib.util.spec_from_file_location("domino_adapter", PROJECT / "adapter.py")
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        cases = adapter.create_cases(CONFIG, {})
        again = adapter.create_cases(CONFIG, {})
        self.assertEqual(len(cases), 3)
        self.assertEqual(len(cases[0]), 9)
        self.assertEqual(cases[0][0].shape, (1, 32, 128))
        self.assertEqual(cases[0][1].shape, (1, 32, 512))
        for case, repeat in zip(cases, again, strict=True):
            for value, expected in zip(case, repeat, strict=True):
                self.assertEqual(value.dtype, torch.float32)
                self.assertTrue(torch.isfinite(value).all())
                self.assertTrue(torch.equal(value, expected))
            self.assertTrue((case[7] > 0).all())
            self.assertTrue((case[8] > 0).all())
        self.assertFalse(torch.equal(cases[0][0], cases[1][0]))


if __name__ == "__main__":
    unittest.main()
