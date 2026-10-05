"""Reject colliding native CLI outputs before loading CUDA or writing any files."""

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CliOutputConflictsTest(unittest.TestCase):
    executable = None

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def run_cli(self, outputs):
        # An invalid device makes the test independent of CUDA and model assets.
        # Output validation must happen before device checks or preprocessing.
        command = [
            str(self.executable),
            "--package",
            "missing-package",
            "--mesh",
            "missing-mesh",
            "--stl",
            "missing-stl",
            "--domain",
            "surface",
            "--device",
            "cpu",
        ]
        for option, path in outputs.items():
            command.extend([option, str(path)])
        return subprocess.run(command, cwd=self.root, capture_output=True, text=True)

    def assert_conflict(self, outputs, first, second):
        result = self.run_cli(outputs)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("output paths must be distinct", result.stderr)
        self.assertIn(first, result.stderr)
        self.assertIn(second, result.stderr)

    def test_rejects_each_output_pair_without_writing(self):
        for first, second in (
            ("--standardized-output", "--physical-output"),
            ("--standardized-output", "--metadata"),
            ("--physical-output", "--metadata"),
        ):
            with self.subTest(first=first, second=second):
                self.assert_conflict(
                    {first: "result.f32", second: "result.f32"}, first, second
                )
                self.assertEqual(list(self.root.iterdir()), [])

    def test_rejects_default_metadata_overlapping_other_output(self):
        self.assert_conflict(
            {
                "--physical-output": "result.f32",
                "--standardized-output": "result.f32.json",
            },
            "--standardized-output",
            "--metadata",
        )
        self.assertEqual(list(self.root.iterdir()), [])

    def test_rejects_normalized_relative_and_absolute_paths(self):
        self.assert_conflict(
            {
                "--standardized-output": "results/../result.f32",
                "--physical-output": self.root / "result.f32",
            },
            "--standardized-output",
            "--physical-output",
        )
        self.assertEqual(list(self.root.iterdir()), [])

    def test_rejects_symlinked_parent_for_new_outputs(self):
        (self.root / "results").mkdir()
        (self.root / "alias").symlink_to(
            self.root / "results", target_is_directory=True
        )
        for metadata in ("alias/new.f32", "new/../alias/new.f32"):
            with self.subTest(metadata=metadata):
                self.assert_conflict(
                    {"--physical-output": "results/new.f32", "--metadata": metadata},
                    "--physical-output",
                    "--metadata",
                )
                self.assertEqual(list((self.root / "results").iterdir()), [])

    def test_rejects_existing_symlink_and_hardlink_without_truncating(self):
        original = self.root / "result.f32"
        original.write_bytes(b"existing output must survive")
        symlink = self.root / "symlink.f32"
        symlink.symlink_to(original)
        hardlink = self.root / "hardlink.f32"
        hardlink.hardlink_to(original)
        for alias in (symlink, hardlink):
            with self.subTest(alias=alias.name):
                self.assert_conflict(
                    {"--standardized-output": original, "--metadata": alias},
                    "--standardized-output",
                    "--metadata",
                )
                self.assertEqual(original.read_bytes(), b"existing output must survive")
                self.assertEqual(alias.read_bytes(), b"existing output must survive")

    def test_rejects_dangling_output_symlinks_before_creating_target(self):
        target = self.root / "result.f32"
        alias = self.root / "metadata.json"
        alias.symlink_to(target)
        result = self.run_cli({"--physical-output": target, "--metadata": alias})
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(
            "output path contains a dangling symlink: --metadata", result.stderr
        )
        self.assertFalse(target.exists())
        self.assertTrue(alias.is_symlink())

    def test_rejects_output_under_dangling_directory_symlink(self):
        target = self.root / "results"
        alias = self.root / "alias"
        alias.symlink_to(target, target_is_directory=True)
        for output in ("alias/result.f32", "new/../alias/result.f32"):
            with self.subTest(output=output):
                result = self.run_cli(
                    {
                        "--standardized-output": "results/result.f32",
                        "--physical-output": output,
                    }
                )
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(
                    "output path contains a dangling symlink: --physical-output",
                    result.stderr,
                )
                self.assertFalse(target.exists())
                self.assertTrue(alias.is_symlink())

    def test_distinct_and_omitted_outputs_reach_device_validation(self):
        for outputs in (
            {
                "--standardized-output": "results/standardized.f32",
                "--physical-output": "results/physical.f32",
                "--metadata": "sidecars/metadata.json",
            },
            {"--standardized-output": "result.f32"},
            {"--physical-output": "result.f32"},
            {
                "--standardized-output": "first/result.f32",
                "--physical-output": "second/result.f32",
            },
        ):
            with self.subTest(outputs=outputs):
                result = self.run_cli(outputs)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("this workflow requires --device cuda", result.stderr)
                self.assertEqual(list(self.root.iterdir()), [])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=lambda value: Path(value).resolve())
    args, remaining = parser.parse_known_args()
    CliOutputConflictsTest.executable = args.executable
    unittest.main(argv=[sys.argv[0], *remaining])
