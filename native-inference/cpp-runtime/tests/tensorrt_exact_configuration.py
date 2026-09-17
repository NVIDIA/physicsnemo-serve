"""Fail early on invalid exact-plugin configurations without requiring CUDA."""

import argparse
from pathlib import Path
import subprocess
import tempfile
import unittest


class ExactConfigurationTests(unittest.TestCase):
    sdk_source: Path

    def configure(self, *options):
        with tempfile.TemporaryDirectory(prefix="pnmir-exact-config-") as build:
            return subprocess.run(
                ["cmake", "-S", str(self.sdk_source), "-B", build,
                 "-DPNMIR_BUILD_TESTS=OFF", *options],
                capture_output=True, text=True,
            )

    def test_exact_requires_tensorrt_backend(self):
        result = self.configure("-DPNMIR_ENABLE_TENSORRT_EXACT=ON")
        self.assertNotEqual(result.returncode, 0,
                            "exact plugins must not silently disable themselves")
        self.assertIn("requires PNMIR_ENABLE_TENSORRT=ON", result.stdout + result.stderr)

    def test_exact_requires_explicit_pytorch_headers(self):
        result = self.configure("-DPNMIR_ENABLE_TENSORRT_EXACT=ON",
                                "-DPNMIR_ENABLE_TENSORRT=ON")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("PNMIR_PYTORCH_SOURCE_ROOT must contain", result.stdout + result.stderr)

    def test_domino_sidecar_requires_aoti_backend(self):
        result = self.configure("-DPNMIR_BUILD_DOMINO_EXACT_OPS=ON")
        self.assertNotEqual(result.returncode, 0,
                            "DoMINO exact operators must not be silently disabled")
        self.assertIn("requires PNMIR_ENABLE_AOTI=ON", result.stdout + result.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdk-source", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    ExactConfigurationTests.sdk_source = args.sdk_source.resolve()
    unittest.main(argv=[__file__, *remaining])
