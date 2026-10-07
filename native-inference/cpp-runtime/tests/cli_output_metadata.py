"""Exercise the native CLI's observed-output contract without a Python model."""

from __future__ import annotations

import argparse
import json
import os
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

    def fixture(self, *, dtype="float32", name="output", shape=None, names=None):
        shape = shape or [-1]
        (self.package / "model.json").write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "model": {"name": "metadata-fixture", "version": "1"},
                    "inputs": [{"name": "input", "dtype": dtype, "shape": shape}],
                    "outputs": [
                        {"name": output, "dtype": dtype, "shape": shape}
                        for output in (names if names is not None else [name])
                    ],
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
            cwd=self.root,
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

    def test_relative_paths_use_fixture_directory(self):
        self.fixture(shape=[3])
        expected = struct.pack("=3f", 1.25, -2, 3.5)
        source = self.root / "relative input.bin"
        source.write_bytes(expected)
        output, relative_output = self.aliased_destination(
            self.root / "relative outputs", "relative", False
        )
        self.assertFalse(relative_output.is_absolute())
        result = self.run_cli(
            "--input-file",
            f"input={source.name}",
            "--output-file",
            str(relative_output),
        )
        self.assert_success(result)
        self.assertEqual(output.read_bytes(), expected)

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

    def test_values_preserve_decimal_exponent_and_whitespace_inputs(self):
        self.fixture(shape=[3])
        for values in ("1.5,-2,+3", "1.5e0,-2E+0,+3e0", " 1.5 , -2\t, +3 "):
            with self.subTest(values=values):
                result = self.run_cli("--values", values)
                self.assert_success(result)
                self.assertEqual(result.stdout.strip(), "output: 1.5 -2 3")

    def test_values_reject_partial_numbers_before_writing_outputs(self):
        self.fixture(shape=[3])
        for index, values in enumerate(
            ("1.5e,2,3", "1,2junk,3", "1,2,3 4", "1.5e,2junk,3 4")
        ):
            with self.subTest(values=values):
                output = self.root / f"output-{index}.bin"
                self.metadata = self.root / f"observed-{index}.json"
                result = self.run_cli("--values", values, "--output-file", str(output))
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("invalid input value:", result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertFalse(output.exists())
                self.assertFalse(self.metadata.exists())

    def test_values_reject_trailing_comma_before_writing_outputs(self):
        self.fixture(shape=[3])
        output = self.root / "output.bin"
        result = self.run_cli("--values", "1,2,3,", "--output-file", str(output))
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("empty input value", result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertFalse(output.exists())
        self.assertFalse(self.metadata.exists())

    def aliased_destination(self, directory, kind, existing):
        directory.mkdir()
        destination = directory / "output.bin"
        if existing:
            destination.write_bytes(b"preserve existing output")
        if kind == "direct":
            alias = destination
        elif kind == "relative":
            # Use the CLI's cwd, independent of CTest's directory or drive.
            alias = destination.relative_to(self.root)
        elif kind == "lexical":
            child = directory / "child"
            child.mkdir()
            alias = child / ".." / destination.name
        elif kind == "symlink":
            alias = directory / "alias.bin"
            alias.symlink_to(destination.name)
        elif kind == "symlink_parent":
            link = directory / "alias-directory"
            link.symlink_to(directory, target_is_directory=True)
            alias = link / destination.name
        elif kind == "hardlink":
            alias = directory / "alias.bin"
            os.link(destination, alias)
        else:
            raise AssertionError(f"unknown alias kind: {kind}")
        return destination, alias

    def collision_cases(self):
        for kind in (
            "direct",
            "relative",
            "lexical",
            "symlink",
            "symlink_parent",
            "hardlink",
        ):
            for existing in (False, True):
                if kind != "hardlink" or existing:
                    yield kind, existing

    def test_tensor_and_metadata_destinations_must_be_distinct(self):
        self.fixture(shape=[3])
        for kind, existing in self.collision_cases():
            with self.subTest(alias=kind, existing=existing):
                destination, self.metadata = self.aliased_destination(
                    self.root / f"{kind}-{existing}", kind, existing
                )
                result = self.run_cli(
                    "--values", "1,2,3", "--output-file", str(destination)
                )
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("output paths must be distinct:", result.stderr)
                self.assertNotIn("wrote", result.stdout)
                if existing:
                    self.assertEqual(
                        destination.read_bytes(), b"preserve existing output"
                    )
                else:
                    self.assertFalse(destination.exists())

    def test_named_tensor_destinations_are_checked_before_any_writes(self):
        # The mock backend has one output, so collisions must be caught before
        # backend execution for this otherwise valid multi-output manifest.
        self.fixture(shape=[3], names=["before", "first", "second"])
        for kind, existing in self.collision_cases():
            with self.subTest(alias=kind, existing=existing):
                directory = self.root / f"{kind}-{existing}"
                destination, alias = self.aliased_destination(directory, kind, existing)
                before = directory / "before.bin"
                before.write_bytes(b"preserve unrelated output")
                self.metadata = directory / "observed.json"
                self.metadata.write_bytes(b"preserve existing metadata")
                result = self.run_cli(
                    "--values",
                    "1,2,3",
                    "--output-file",
                    f"before={before}",
                    "--output-file",
                    f"first={destination}",
                    "--output-file",
                    f"second={alias}",
                )
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("output paths must be distinct:", result.stderr)
                self.assertNotIn("wrote", result.stdout)
                self.assertEqual(before.read_bytes(), b"preserve unrelated output")
                self.assertEqual(
                    self.metadata.read_bytes(), b"preserve existing metadata"
                )
                if existing:
                    self.assertEqual(
                        destination.read_bytes(), b"preserve existing output"
                    )
                else:
                    self.assertFalse(destination.exists())

    def test_distinct_destinations_allow_existing_files_and_dangling_symlinks(self):
        self.fixture(shape=[3])
        for existing in (False, True):
            with self.subTest(existing=existing):
                destination, alias = self.aliased_destination(
                    self.root / str(existing), "symlink", existing
                )
                self.metadata.write_bytes(b"replace existing metadata")
                result = self.run_cli("--values", "1,2,3", "--output-file", str(alias))
                self.assert_success(result)
                self.assertEqual(destination.read_bytes(), struct.pack("=3f", 1, 2, 3))
                self.assertTrue(json.loads(self.metadata.read_text())["completed"])

    def test_case_only_destinations_follow_filesystem_case_sensitivity(self):
        probe = self.root / "case-sensitivity-probe"
        probe.write_bytes(b"probe")
        case_sensitive = not probe.with_name(probe.name.upper()).exists()
        self.fixture(shape=[3])
        for existing in (False, True):
            with self.subTest(existing=existing, case_sensitive=case_sensitive):
                output = self.root / f"output-{existing}.bin"
                self.metadata = output.with_name(output.name.upper())
                if existing:
                    output.write_bytes(b"preserve existing output")
                result = self.run_cli("--values", "1,2,3", "--output-file", str(output))
                if case_sensitive:
                    self.assert_success(result)
                    self.assertEqual(output.read_bytes(), struct.pack("=3f", 1, 2, 3))
                else:
                    self.assertNotEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                    self.assertIn("output paths must be distinct:", result.stderr)
                    # An initially absent alias may only be identifiable after
                    # creating the tensor file; metadata must not overwrite it.
                    self.assertEqual(
                        output.read_bytes(),
                        b"preserve existing output"
                        if existing
                        else struct.pack("=3f", 1, 2, 3),
                    )

    @unittest.skipUnless(os.name == "posix", "requires POSIX file-size limits")
    def test_buffered_output_write_failure(self):
        import resource
        import signal

        def reject_file_writes():
            signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
            resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))

        self.fixture(shape=[3])
        output = self.root / "output.bin"
        for metadata_arguments in ([], ["--output-metadata", str(self.metadata)]):
            with self.subTest(metadata=bool(metadata_arguments)):
                result = subprocess.run(
                    [
                        str(self.executable),
                        "run",
                        str(self.package),
                        "--values",
                        "1,2,3",
                        "--output-file",
                        str(output),
                        *metadata_arguments,
                    ],
                    capture_output=True,
                    text=True,
                    preexec_fn=reject_file_writes,
                )
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("cannot write output file:", result.stderr)
                self.assertNotIn("wrote", result.stdout)
                self.assertFalse(self.metadata.exists())
                self.assertEqual(output.stat().st_size, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnmir", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    OutputMetadataTests.executable = args.pnmir.resolve()
    unittest.main(argv=[__file__, *remaining])
