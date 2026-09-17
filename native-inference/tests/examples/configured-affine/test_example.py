"""Independent analytical expectations for the customer example."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[3] / "examples/configured-affine"
SPEC = importlib.util.spec_from_file_location("configured_affine", ROOT / "adapter.py")
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)


class ExampleTest(unittest.TestCase):
    def test_checkpoint_config_and_asset_each_change_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            asset = Path(temporary) / "normalization.json"
            asset.write_text(json.dumps({"mean": 1.0, "std": 2.0}))
            config = {"offset": 3.0}
            inputs = torch.tensor([0.0, 1.0, -1.0, 4.0])
            model = ADAPTER.create_model(config, {"normalization": asset})
            model.load_state_dict(
                {"scale": torch.tensor(2.0), "bias": torch.tensor(1.0)}
            )
            self.assertEqual(model(inputs).tolist(), [3.0, 4.0, 2.0, 7.0])
            model.load_state_dict(
                {"scale": torch.tensor(-1.0), "bias": torch.tensor(0.5)}
            )
            self.assertEqual(model(inputs).tolist(), [4.0, 3.5, 4.5, 2.0])

            alternate = ADAPTER.create_model({"offset": -2.0}, {"normalization": asset})
            alternate.load_state_dict(model.state_dict())
            self.assertEqual(alternate(inputs).tolist(), [-1.0, -1.5, -0.5, -3.0])

            asset.write_text(json.dumps({"mean": 0.0, "std": 1.0}))
            unnormalized = ADAPTER.create_model(config, {"normalization": asset})
            unnormalized.load_state_dict(model.state_dict())
            self.assertEqual(unnormalized(inputs).tolist(), [3.5, 2.5, 4.5, -0.5])

            cases = ADAPTER.create_cases(config, {"normalization": asset})
            self.assertEqual(len(cases), 3)
            self.assertTrue(all(case[0].shape == (4,) for case in cases))
            self.assertTrue(all(case[0].dtype == torch.float32 for case in cases))


if __name__ == "__main__":
    unittest.main()
