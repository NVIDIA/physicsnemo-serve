"""Portable reference characterization with a tiny upstream API double."""

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import unittest

spec = importlib.util.spec_from_file_location(
    "geo_reference_fixture", Path(__file__).with_name("test_example.py")
)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


@unittest.skipIf(fixture.torch is None, "requires optional producer Torch")
class PortableReferencesTest(unittest.TestCase):
    def setUp(self):
        self.example = fixture.GeoTransolverExampleTests()
        self.example.setUp()
        self.addCleanup(self.example.doCleanups)

    def test_portable_references_bind_full_model_outputs_inputs_and_prepared_sources(
        self,
    ):
        report = self.example.run_preparation()
        root = self.example.output
        path = root / "references/manifest.json"
        self.assertTrue(
            path.is_file(), "preparation must emit portable full-model references"
        )
        manifest = json.loads(path.read_text())
        self.assertEqual(manifest["reference_kind"], "upstream-full-model")
        self.assertEqual(manifest["checkpoint"], report["checkpoint"])
        self.assertEqual(manifest["model_state_sha256"], report["model_state_sha256"])
        self.assertEqual(manifest["source"], report["source"])
        self.assertEqual(manifest["comparisons"], report["comparisons"])
        fixtures = fixture.torch.load(
            root / "fixtures.pt", weights_only=True, map_location="cpu"
        )
        self.assertEqual(len(manifest["cases"]), 3)
        for case, tensors in zip(manifest["cases"], fixtures["cases"], strict=True):
            for descriptor, value in zip(
                case["inputs"] + case["outputs"],
                [*tensors["inputs"], tensors["expected"]],
                strict=True,
            ):
                data = (root / descriptor["path"]).read_bytes()
                self.assertEqual(descriptor["sha256"], hashlib.sha256(data).hexdigest())
                self.assertEqual(descriptor["size_bytes"], len(data))
                self.assertEqual(descriptor["shape"], list(value.shape))
                self.assertEqual(descriptor["dtype"], "float32")
                self.assertEqual(
                    struct.unpack(f"<{value.numel()}f", data),
                    tuple(value.reshape(-1).tolist()),
                )
        for descriptor in manifest["prepared_files"]:
            self.assertEqual(
                hashlib.sha256((root / descriptor["path"]).read_bytes()).hexdigest(),
                descriptor["sha256"],
            )
        inventory = {item["path"] for item in report["files"]}
        self.assertIn("references/manifest.json", inventory)
        for case in manifest["cases"]:
            for descriptor in case["inputs"] + case["outputs"]:
                self.assertIn(descriptor["path"], inventory)


if __name__ == "__main__":
    unittest.main()
