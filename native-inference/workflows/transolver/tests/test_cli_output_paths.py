"""Exercise native output files and metadata using a built surface package."""

import argparse
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]


class OutputWriteCompletionTest(unittest.TestCase):
    """Check actual C++ file writers without CUDA or serialization dependencies."""

    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if compiler is None:
            raise unittest.SkipTest("a C++ compiler is required for file writer checks")
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        source = (WORKFLOW_ROOT / "src" / "main.cpp").read_text()
        helpers = []
        for name in ("write_f32", "write_metadata"):
            start = source.index(f"void {name}(")
            end = source.index("\n}\n", start) + 3
            helpers.append(source[start:end])
        # Only the tensor bytes and serialized JSON are stand-ins. Compile the
        # production helper bodies unchanged so their stream handling is tested.
        probe = cls.root / "writers.cpp"
        probe.write_text(
            r"""
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
namespace torch {
constexpr int kCPU = 0, kFloat32 = 1;
struct Tensor {
  std::array<float, 300> values{};
  const Tensor& detach() const { return *this; }
  const Tensor& to(int) const { return *this; }
  const Tensor& contiguous() const { return *this; }
  int scalar_type() const { return kFloat32; }
  const void* const_data_ptr() const { return values.data(); }
  std::size_t numel() const { return values.size(); }
  std::size_t element_size() const { return sizeof(float); }
};
}
struct Json {
  std::string dump(int) const { return "\"" + std::string(3998, 'x') + "\""; }
};
"""
            + "\n".join(helpers)
            + r"""
int main(int argc, char** argv) {
  try {
    if (std::string(argv[1]) == "f32") write_f32(argv[2], torch::Tensor{});
    else write_metadata(argv[2], Json{});
  } catch (const std::exception& error) {
    std::cerr << error.what();
    return 2;
  }
}
"""
        )
        cls.executable = cls.root / "writers"
        subprocess.run(
            [compiler, "-std=c++20", str(probe), "-o", str(cls.executable)],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_writes_complete_files(self):
        for kind, expected in (
            ("f32", b"\0" * 1200),
            ("metadata", ('"' + "x" * 3998 + '"\n').encode()),
        ):
            with self.subTest(kind=kind):
                output = self.root / "complete" / kind
                result = subprocess.run(
                    [str(self.executable), kind, str(output)],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(output.read_bytes(), expected)

    def test_rejects_buffered_write_failures(self):
        try:
            import resource
        except ImportError:
            self.skipTest("POSIX file size limits are required")
        if not hasattr(resource, "RLIMIT_FSIZE") or not hasattr(signal, "SIGXFSZ"):
            self.skipTest("RLIMIT_FSIZE and SIGXFSZ are required")

        def limit_file_size():
            signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
            resource.setrlimit(resource.RLIMIT_FSIZE, (512, 512))

        for kind, message in (
            ("f32", "failed to write output file:"),
            ("metadata", "failed to write metadata file:"),
        ):
            with self.subTest(kind=kind):
                output = self.root / "limited" / kind
                result = subprocess.run(
                    [str(self.executable), kind, str(output)],
                    preexec_fn=limit_file_size,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(output.stat().st_size, 512)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn(message, result.stderr)
                self.assertIn(str(output), result.stderr)


class CliOutputPathsTest(unittest.TestCase):
    fixture = None

    @classmethod
    def setUpClass(cls):
        if cls.fixture is None:
            raise unittest.SkipTest(
                "run this script with native CLI and surface fixture paths"
            )

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def run_workflow(self, outputs, metadata=None):
        fixture = self.fixture
        command = [
            str(fixture.executable),
            "--backend",
            fixture.backend,
            "--package",
            str(fixture.package),
            "--mesh",
            str(fixture.mesh),
            "--stl",
            str(fixture.stl),
            "--stats",
            str(fixture.stats),
            "--domain",
            "surface",
            "--point-limit",
            str(fixture.point_count),
            "--block-size",
            str(fixture.point_count),
            "--seed",
            "0",
        ]
        for name, path in outputs.items():
            command.extend(["--" + name.replace("_", "-"), str(path)])
        if metadata is not None:
            command.extend(["--metadata", str(metadata)])
        result = subprocess.run(command, cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        if metadata is None:
            metadata = (
                str(outputs.get("physical_output", outputs.get("standardized_output")))
                + ".json"
            )
        return self.root / metadata

    def test_metadata_resolves_outputs_with_default_and_custom_metadata(self):
        outputs = {
            "physical_output": Path("results/physical.f32"),
            "standardized_output": Path("results/standardized.f32"),
        }
        for metadata in (None, Path("sidecars/nested/metadata.json")):
            with self.subTest(metadata=metadata):
                metadata_path = self.run_workflow(outputs, metadata)
                payload = json.loads(metadata_path.read_text())
                self.assertEqual(
                    payload["output_shape"], [1, self.fixture.point_count, 4]
                )
                self.assertEqual(payload["output_dtype"], "float32")
                for name, path in outputs.items():
                    expected_path = self.root / path
                    self.assertEqual(Path(payload[name]), expected_path)
                    self.assertEqual(
                        len(expected_path.read_bytes()),
                        self.fixture.point_count * 4 * 4,
                    )

    def test_omitted_output_stays_empty(self):
        for present, omitted in (
            ("physical_output", "standardized_output"),
            ("standardized_output", "physical_output"),
        ):
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
        parser.add_argument(
            "--" + name, required=True, type=lambda value: Path(value).resolve()
        )
    parser.add_argument("--backend", choices=("aoti", "tensorrt"), default="aoti")
    parser.add_argument(
        "--point-count",
        type=int,
        default=75,
        help="Point count supported by the supplied package (default: 75)",
    )
    CliOutputPathsTest.fixture, remaining = parser.parse_known_args()
    unittest.main(argv=[sys.argv[0], *remaining])
