"""Host-independent checks for the installed SDK's Windows paths."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path, PureWindowsPath
from unittest import mock

import install_consumer


class WindowsCMakePathTests(unittest.TestCase):
    def test_dependency_paths_survive_cmake_macro_reparsing(self):
        root = PureWindowsPath(r"C:\Users\Builder SDKs")
        case = install_consumer.InstallConsumerTests()
        case.onnxruntime_root = root / "ONNX Runtime"
        case.tensorrt_root = root / "TensorRT"
        case.cuda_root = root / "CUDA"
        case.cuda_architectures = "89,90"
        with mock.patch.object(install_consumer.sys, "platform", "win32"):
            options = case.dependency_options()
        with tempfile.TemporaryDirectory() as temporary:
            script = Path(temporary) / "check-paths.cmake"
            script.write_text("""\
cmake_minimum_required(VERSION 3.20)
# FindCUDA reparses macro arguments, including the caller's CUDA toolkit root.
macro(assert_path actual expected)
  if(NOT "${actual}" STREQUAL "${expected}")
    message(FATAL_ERROR "Path changed: ${actual} != ${expected}")
  endif()
endmacro()
assert_path("${PNMIR_ONNXRUNTIME_ROOT}" "C:/Users/Builder SDKs/ONNX Runtime")
assert_path("${PNMIR_TENSORRT_ROOT}" "C:/Users/Builder SDKs/TensorRT")
assert_path("${CUDAToolkit_ROOT}" "C:/Users/Builder SDKs/CUDA")
assert_path("${CUDA_TOOLKIT_ROOT_DIR}" "C:/Users/Builder SDKs/CUDA")
assert_path("${CMAKE_CUDA_COMPILER}" "C:/Users/Builder SDKs/CUDA/bin/nvcc.exe")
assert_path("${CMAKE_CUDA_ARCHITECTURES}" "89;90")
""")
            result = subprocess.run(
                ["cmake", *options, "-P", str(script)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class WindowsRuntimeEnvironmentTests(unittest.TestCase):
    def test_relocated_dll_and_external_dependencies_use_child_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            relocated = root / "relocated-sdk"
            dependency = root / "dependency"
            for directory in (
                relocated / "bin",
                dependency / "bin",
                dependency / "lib",
            ):
                directory.mkdir(parents=True)
            case = install_consumer.InstallConsumerTests()
            case.runtime_prefix = relocated
            case.torch_root = case.tensorrt_root = case.cuda_root = None
            case.onnxruntime_root = dependency
            completed = subprocess.CompletedProcess([], 0, "", "")
            with (
                mock.patch.object(install_consumer.sys, "platform", "win32"),
                mock.patch.dict(os.environ, {"PATH": "existing-tool-path"}),
                mock.patch.object(
                    install_consumer.subprocess, "run", return_value=completed
                ) as run,
            ):
                case.command("consumer.exe", runtime=True)
                environment = run.call_args.kwargs["env"]
                directories = environment["PATH"].split(os.pathsep)
                self.assertIn(str(relocated / "bin"), directories)
                self.assertIn(str(dependency / "bin"), directories)
                self.assertIn(str(dependency / "lib"), directories)
                self.assertEqual(directories[-1], "existing-tool-path")
                self.assertNotIn("LD_LIBRARY_PATH", environment)
                self.assertEqual(os.environ["PATH"], "existing-tool-path")


if __name__ == "__main__":
    unittest.main()
