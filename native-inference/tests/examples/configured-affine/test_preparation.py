"""A prepared affine example is a directly usable authoring project."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch


NATIVE = Path(__file__).resolve().parents[3]
HELPER = NATIVE / "tools/examples/prepare_configured_affine.py"
EXAMPLE = NATIVE / "examples/configured-affine"


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
                    (output / name).read_bytes(), (EXAMPLE / name).read_bytes()
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
                doctor = subprocess.run(
                    [
                        sys.executable,
                        str(NATIVE / "physicsnemo-model-builder"),
                        "doctor",
                        str(output),
                        *arguments,
                        "--json",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(doctor.returncode, 0, doctor.stdout + doctor.stderr)
                self.assertEqual(
                    json.loads(doctor.stdout)["status"], "configuration-ok"
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
