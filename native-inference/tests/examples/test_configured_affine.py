"""Analytical expectations and prepared-project contracts for the affine example."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch


NATIVE = Path(__file__).resolve().parents[2]
ROOT = NATIVE / "examples/configured-affine"
HELPER = NATIVE / "tools/examples/prepare_configured_affine.py"
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


class PreparationTests(unittest.TestCase):
    def invoke(self, output):
        return subprocess.run(
            [sys.executable, str(HELPER), "--output", str(output)],
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_stages_project_with_both_checkpoints_and_normalization(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "affine"
            result = self.invoke(output)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(
                {p.name for p in output.iterdir()},
                {
                    "adapter.py",
                    "model-build.json",
                    "checkpoint-a.pt",
                    "checkpoint-b.pt",
                    "normalization.json",
                },
            )
            for name in ("adapter.py", "model-build.json"):
                self.assertEqual(
                    (output / name).read_bytes(), (ROOT / name).read_bytes()
                )
            config = json.loads((output / "model-build.json").read_text())
            self.assertEqual(config["format_version"], 2)
            self.assertEqual(config["adapter"], "adapter.py")
            self.assertEqual(config["config"], {"offset": 3.0})
            self.assertEqual(config["checkpoint"], "checkpoint-a.pt")
            self.assertEqual(config["assets"], {"normalization": "normalization.json"})
            self.assertEqual(config["backends"], ["aoti"])
            self.assertEqual(config["device"], "cpu")
            self.assertEqual(config["executor"], "local")
            self.assertEqual(config["output_root"], "builds")
            self.assertEqual(
                json.loads((output / "normalization.json").read_text()),
                {"mean": 1.0, "std": 2.0},
            )
            for name, expected in (
                ("checkpoint-a.pt", {"scale": 2.0, "bias": 1.0}),
                ("checkpoint-b.pt", {"scale": -1.0, "bias": 0.5}),
            ):
                weights = torch.load(output / name, weights_only=True)
                self.assertEqual(set(weights), set(expected))
                self.assertEqual({k: v.item() for k, v in weights.items()}, expected)
                self.assertTrue(all(v.dtype == torch.float32 for v in weights.values()))
            for arguments in ([], ["--backend", "tensorrt", "--device", "cuda"]):
                config_check = subprocess.run(
                    [
                        sys.executable,
                        str(NATIVE / "pnms-model-builder"),
                        "check",
                        "--config-only",
                        str(output),
                        *arguments,
                        "--json",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(
                    config_check.returncode,
                    0,
                    config_check.stdout + config_check.stderr,
                )
                self.assertEqual(
                    json.loads(config_check.stdout)["status"], "configuration-ok"
                )

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            sentinel = output / "customer-file"
            sentinel.write_bytes(b"preserve me")
            result = self.invoke(output)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("must be a new directory", result.stderr)
            self.assertEqual(list(output.iterdir()), [sentinel])
            self.assertEqual(sentinel.read_bytes(), b"preserve me")


if __name__ == "__main__":
    unittest.main()
