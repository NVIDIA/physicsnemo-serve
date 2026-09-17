"""Compiler selections share the original deterministic MLP and case fixtures."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch


NATIVE = Path(__file__).resolve().parents[3]
HELPER = NATIVE / "tools/examples/prepare_aoti_profiles.py"
EXAMPLE = NATIVE / "examples/aoti-profiles"


class PreparationTests(unittest.TestCase):
    def invoke(self, output):
        return subprocess.run(
            [sys.executable, str(HELPER), "--output", str(output)],
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_preparation_preserves_saved_fixtures_and_all_four_profiles(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "aoti"
            result = self.invoke(output)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(
                {p.name for p in output.iterdir()},
                {
                    "adapter.py",
                    "model-build.json",
                    "weights.pt",
                    "cases.pt",
                    "features.bin",
                },
            )
            for name in ("adapter.py", "model-build.json"):
                self.assertEqual(
                    (output / name).read_bytes(), (EXAMPLE / name).read_bytes()
                )
            document = json.loads((output / "model-build.json").read_text())
            self.assertEqual(document["default_profile"], "accuracy")
            self.assertEqual(document["device"], "cuda")
            self.assertEqual(
                document["profiles"],
                {
                    "accuracy": {
                        "aoti_profile": "aten-boundary-exact-v2",
                        "aoti_options": {
                            "max_autotune": False,
                            "epilogue_fusion": False,
                            "shape_padding": False,
                            "coordinate_descent_tuning": False,
                        },
                    },
                    "standard": {"aoti_profile": "baseline", "aoti_options": {}},
                    "autotune": {
                        "aoti_profile": "baseline",
                        "aoti_options": {"max_autotune": True, "epilogue_fusion": True},
                    },
                    "autotune-no-fusion": {
                        "aoti_profile": "baseline",
                        "aoti_options": {
                            "max_autotune": True,
                            "epilogue_fusion": False,
                        },
                    },
                },
            )
            # Tensor-byte fingerprints captured before moving the original helper.
            weights = torch.load(output / "weights.pt", weights_only=True)
            digest = hashlib.sha256()
            for name, value in sorted(weights.items()):
                digest.update(name.encode())
                digest.update(value.numpy().tobytes())
            self.assertEqual(
                digest.hexdigest(),
                "28e3c7740f666652476a6e9433de0cdc65bdf259d9579fa197d34e839aea6af6",
            )
            cases = torch.load(output / "cases.pt", weights_only=True)
            self.assertEqual(len(cases), 3)
            self.assertTrue(all(case[0].shape == (16, 128) for case in cases))
            self.assertTrue(all(case[0].dtype == torch.float32 for case in cases))
            self.assertEqual(
                hashlib.sha256(
                    b"".join(case[0].numpy().tobytes() for case in cases)
                ).hexdigest(),
                "c01d714afa501a0a67fc382f48d8643d42c49bb8b055fe57e9d826843902b704",
            )
            self.assertEqual(
                (output / "features.bin").read_bytes(),
                cases[0][0].numpy().astype("<f4", copy=False).tobytes(),
            )
            for profile in document["profiles"]:
                doctor = subprocess.run(
                    [
                        sys.executable,
                        str(NATIVE / "physicsnemo-model-builder"),
                        "doctor",
                        str(output),
                        "--profile",
                        profile,
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
