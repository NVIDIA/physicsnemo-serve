"""Exercise native output metadata and the viewer using a built surface package."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "demo" / "export_viewer.py"
SPEC = importlib.util.spec_from_file_location("export_viewer", SCRIPT)
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


class CliOutputPathsTest(unittest.TestCase):
    fixture = None

    @classmethod
    def setUpClass(cls):
        if cls.fixture is None:
            raise unittest.SkipTest("run this script with native CLI and surface fixture paths")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def run_workflow(self, outputs, metadata=None):
        fixture = self.fixture
        command = [
            str(fixture.executable), "--backend", fixture.backend,
            "--package", str(fixture.package),
            "--mesh", str(fixture.mesh), "--stl", str(fixture.stl),
            "--stats", str(fixture.stats), "--domain", "surface",
            "--point-limit", str(fixture.point_count),
            "--block-size", str(fixture.point_count), "--seed", "0",
        ]
        for name, path in outputs.items():
            command.extend(["--" + name.replace("_", "-"), str(path)])
        if metadata is not None:
            command.extend(["--metadata", str(metadata)])
        result = subprocess.run(command, cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        if metadata is None:
            metadata = str(outputs.get("physical_output", outputs.get("standardized_output"))) + ".json"
        return self.root / metadata

    def test_viewer_resolves_outputs_with_default_and_custom_metadata(self):
        outputs = {
            "physical_output": Path("results/physical.f32"),
            "standardized_output": Path("results/standardized.f32"),
        }
        for metadata in (None, Path("sidecars/nested/metadata.json")):
            with self.subTest(metadata=metadata):
                metadata_path = self.run_workflow(outputs, metadata)
                try:
                    payload = EXPORT.build_payload(metadata_path)
                except FileNotFoundError as error:
                    self.fail(f"viewer could not find the CLI's generated output: {error}")
                self.assertEqual(len(payload["fields"]), self.fixture.point_count)
                for name, path in outputs.items():
                    expected_path = self.root / path
                    self.assertEqual(Path(payload["metadata"][name]), expected_path)
                    self.assertEqual(
                        payload["hashes"][name],
                        hashlib.sha256(expected_path.read_bytes()).hexdigest(),
                    )

    def test_omitted_output_stays_empty(self):
        for present, omitted in (("physical_output", "standardized_output"),
                                 ("standardized_output", "physical_output")):
            with self.subTest(omitted=omitted):
                output = Path("results") / (present + ".f32")
                metadata_path = self.run_workflow({present: output})
                metadata = json.loads(metadata_path.read_text())
                self.assertEqual(metadata[omitted], "")
                self.assertEqual(Path(metadata[present]), self.root / output)
                self.assertTrue((self.root / output).is_file())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("executable", "package", "mesh", "stl", "stats"):
        parser.add_argument("--" + name, required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--backend", choices=("aoti", "tensorrt"), default="aoti")
    parser.add_argument("--point-count", type=int, default=75,
                        help="Point count supported by the supplied package (default: 75)")
    CliOutputPathsTest.fixture, remaining = parser.parse_known_args()
    unittest.main(argv=[sys.argv[0], *remaining])
