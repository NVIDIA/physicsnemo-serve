"""Check the TensorRT integration's CUDA skip without a GPU or Python backends."""

import builtins
from pathlib import Path
import re
import runpy
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


SDK_SOURCE = Path(__file__).resolve().parents[1]
INTEGRATION = SDK_SOURCE / "tests" / "device_runner_integration.py"
TEST_NAME = "pnmir_tensorrt_device_runner"


class DeviceRunnerCudaSkipTests(unittest.TestCase):
    def run_integration(self, backend, *, cuda_available):
        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: cuda_available)
        )
        with (
            mock.patch.dict(sys.modules, {"torch": torch}),
            mock.patch.object(
                sys,
                "argv",
                [str(INTEGRATION), "--runner", "unused", "--backend", backend],
            ),
            mock.patch("unittest.main") as run_suite,
        ):
            try:
                runpy.run_path(str(INTEGRATION), run_name="__main__")
            except SystemExit as error:
                return error.code, run_suite
        return 0, run_suite

    def test_unavailable_cuda_exits_before_fixture_suite(self):
        code, run_suite = self.run_integration("tensorrt", cuda_available=False)
        self.assertEqual(code, 77, "unavailable CUDA must produce CTest's skip code")
        run_suite.assert_not_called()

    def test_available_cuda_runs_fixture_suite(self):
        code, run_suite = self.run_integration("tensorrt", cuda_available=True)
        self.assertEqual(code, 0)
        run_suite.assert_called_once()

    def test_onnx_does_not_import_torch(self):
        original_import = builtins.__import__

        def import_without_torch(name, *args, **kwargs):
            self.assertNotEqual(name, "torch", "ONNX must not require Torch")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=import_without_torch):
            code, run_suite = self.run_integration("onnxruntime", cuda_available=False)
        self.assertEqual(code, 0)
        run_suite.assert_called_once()

    def test_ctest_reports_unavailable_cuda_as_skipped(self):
        # Execute the SDK's real test registration and skip properties in a tiny
        # CMake project, without configuring or building any GPU dependencies.
        cmake = (SDK_SOURCE / "cmake" / "Tests.cmake").read_text(encoding="utf-8")
        commands = re.findall(r"\b(?:add_test|set_tests_properties)\s*\([^)]*\)", cmake)
        registration = "\n".join(
            command for command in commands if TEST_NAME in command.split()
        ).replace("${CMAKE_CURRENT_SOURCE_DIR}", SDK_SOURCE.as_posix())
        with tempfile.TemporaryDirectory(prefix="pnmir-cuda-skip-") as temporary:
            source = Path(temporary)
            driver = source / "without_cuda.py"
            driver.write_text(
                "import runpy, sys, unittest\n"
                "from types import SimpleNamespace\n"
                "sys.modules['torch'] = SimpleNamespace(\n"
                "    cuda=SimpleNamespace(is_available=lambda: False))\n"
                "def unexpected_suite(*args, **kwargs):\n"
                "    raise AssertionError('fixture suite ran without CUDA')\n"
                "unittest.main = unexpected_suite\n"
                "sys.argv = sys.argv[1:]\n"
                "runpy.run_path(sys.argv[0], run_name='__main__')\n",
                encoding="utf-8",
            )
            (source / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.20)\n"
                "project(DeviceRunnerCudaSkip NONE)\n"
                "enable_testing()\n"
                "add_executable(pnmir_device_runner IMPORTED)\n"
                "set_target_properties(pnmir_device_runner PROPERTIES\n"
                f'  IMPORTED_LOCATION "{Path(sys.executable).as_posix()}")\n'
                f'set(Python3_EXECUTABLE "{Path(sys.executable).as_posix()}"\n'
                f'  "-S" "{driver.as_posix()}")\n' + registration + "\n",
                encoding="utf-8",
            )
            build = source / "build"
            configured = subprocess.run(
                ["cmake", "-S", str(source), "-B", str(build)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                configured.returncode, 0, configured.stdout + configured.stderr
            )
            result = subprocess.run(
                [
                    "ctest",
                    "--test-dir",
                    str(build),
                    "-C",
                    "Release",
                    "--output-on-failure",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertRegex(result.stdout, rf"{TEST_NAME}[^\n]*\*\*\*Skipped")


if __name__ == "__main__":
    unittest.main()
