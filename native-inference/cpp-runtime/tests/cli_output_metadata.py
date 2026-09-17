"""Exercise the native CLI's observed-output contract without a Python model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest


class OutputMetadataTests(unittest.TestCase):
    executable: Path

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pnmir-metadata-")
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.package = self.root / "fixture.pnmir"
        self.package.mkdir()
        (self.package / "identity.mock").write_text("identity\n")
        self.metadata = self.root / "observed.json"

    def fixture(self, *, dtype="float32", name="output", shape=None):
        shape = shape or [-1]
        (self.package / "model.json").write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "model": {"name": "metadata-fixture", "version": "1"},
                    "inputs": [{"name": "input", "dtype": dtype, "shape": shape}],
                    "outputs": [{"name": name, "dtype": dtype, "shape": shape}],
                    "artifacts": [
                        {
                            "backend": "mock",
                            "target": "cpu",
                            "precision": "fp32",
                            "path": "identity.mock",
                        }
                    ],
                }
            )
        )

    def run_cli(self, *arguments):
        return subprocess.run(
            [
                str(self.executable),
                "run",
                str(self.package),
                *arguments,
                "--output-metadata",
                str(self.metadata),
            ],
            capture_output=True,
            text=True,
        )

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(
            self.metadata.is_file(), "successful native run must emit metadata"
        )

    def test_dynamic_shape_and_selected_backend_are_observed(self):
        self.fixture(shape=[-1, -1])
        result = self.run_cli("--values", "2,4,6,8", "--input-shape", "input=2,2")
        self.assert_success(result)
        self.assertEqual(
            json.loads(self.metadata.read_text()),
            {
                "schema_version": 1,
                "backend": "mock",
                "execution_device": {"type": "cpu", "index": 0},
                "completed": True,
                "outputs": [
                    {
                        "name": "output",
                        "dtype": "float32",
                        "shape": [2, 2],
                        "device": {"type": "cpu", "index": 0},
                        "byte_size": 16,
                    }
                ],
            },
        )
        self.assertEqual(result.stdout.strip(), "output: 2 4 6 8")

    def test_integer_dtype_escaped_name_and_written_bytes_match(self):
        name = 'indices"\n'
        self.fixture(dtype="int32", name=name)
        source = self.root / "input.bin"
        source.write_bytes(struct.pack("=3i", 7, -2, 19))
        output = self.root / "output.bin"
        result = self.run_cli(
            "--backend",
            "mock",
            "--input-file",
            f"input={source}",
            "--input-shape",
            "input=3",
            "--output-file",
            str(output),
        )
        self.assert_success(result)
        observed = json.loads(self.metadata.read_text())["outputs"]
        self.assertEqual(
            observed,
            [
                {
                    "name": name,
                    "dtype": "int32",
                    "shape": [3],
                    "device": {"type": "cpu", "index": 0},
                    "byte_size": 12,
                }
            ],
        )
        self.assertEqual(output.read_bytes(), struct.pack("=3i", 7, -2, 19))

    def test_legacy_and_backend_package_paths_preserve_bytes_and_metadata(self):
        self.fixture(shape=[-1, -1])
        expected = struct.pack("=4f", 1.25, -2, 3.5, 0)
        relocated = self.root / "model" / "backends" / "aoti.pnm-model"
        backend = relocated.with_name("aoti")
        observations = []
        for index, package in enumerate((self.package, relocated, backend)):
            with self.subTest(package=package.relative_to(self.root)):
                if package != self.package:
                    package.parent.mkdir(parents=True, exist_ok=True)
                    self.package.rename(package)
                    self.package = package
                self.metadata = self.root / f"observed-{index}.json"
                output = self.root / f"output-{index}.bin"
                result = self.run_cli(
                    "--values",
                    "1.25,-2,3.5,0",
                    "--input-shape",
                    "input=2,2",
                    "--output-file",
                    str(output),
                )
                self.assert_success(result)
                self.assertEqual(output.read_bytes(), expected)
                # Backend selection follows the manifest, even at an AOTI-named path.
                self.assertEqual(
                    json.loads(self.metadata.read_text())["backend"], "mock"
                )
                observations.append((output.read_bytes(), self.metadata.read_bytes()))
        self.assertEqual(observations[0], observations[1])
        self.assertEqual(observations[0], observations[2])

    def test_failed_native_execution_does_not_publish_completion(self):
        self.fixture(shape=[2])
        result = self.run_cli("--values", "1,2,3")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.metadata.exists())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnmir", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    OutputMetadataTests.executable = args.pnmir.resolve()
    unittest.main(argv=[__file__, *remaining])
