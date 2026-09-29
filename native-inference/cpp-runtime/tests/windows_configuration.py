"""Exercise Windows CMake decisions with the host compiler (no GPU required).

These configuration tests do not replace the native MSVC build and relocation
checks in Windows CI. A wrapper sets WIN32 only after compiler detection.
"""

import argparse
from pathlib import Path
import subprocess
import tempfile
import unittest


class WindowsConfigurationTests(unittest.TestCase):
    sdk_source: Path

    def configure(self, *options, assertions="", before_sdk="", tensorrt=False):
        with tempfile.TemporaryDirectory(prefix="pnmir-windows-config-") as temp:
            source = Path(temp) / "source"
            source.mkdir()
            if tensorrt:
                sdk = source / "TensorRT"
                (sdk / "include").mkdir(parents=True)
                (sdk / "lib").mkdir()
                (sdk / "include" / "NvInfer.h").write_text("", encoding="utf-8")
                for name in ("nvinfer_10", "nvinfer_plugin_10"):
                    (sdk / "lib" / f"{name}.lib").write_bytes(b"")
                modules = source / "cmake"
                modules.mkdir()
                (modules / "FindCUDAToolkit.cmake").write_text(
                    "set(CUDAToolkit_FOUND TRUE)\n"
                    "if(NOT TARGET CUDA::cudart)\n"
                    "  add_library(CUDA::cudart INTERFACE IMPORTED)\n"
                    "endif()\n",
                    encoding="utf-8",
                )
                before_sdk += (
                    f'list(PREPEND CMAKE_MODULE_PATH "{modules.as_posix()}")\n'
                    f'set(PNMIR_TENSORRT_ROOT "{sdk.as_posix()}" CACHE PATH "")\n'
                    'set(CMAKE_FIND_LIBRARY_PREFIXES "")\n'
                    'set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib")\n'
                )
            (source / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.20)\n"
                "project(WindowsConfiguration LANGUAGES CXX)\n"
                "set(WIN32 TRUE)\n"
                + before_sdk
                + f'add_subdirectory("{self.sdk_source.as_posix()}" sdk)\n'
                + assertions,
                encoding="utf-8",
            )
            return subprocess.run(
                [
                    "cmake",
                    "-S",
                    str(source),
                    "-B",
                    str(Path(temp) / "build"),
                    "-DPNMIR_BUILD_TESTS=OFF",
                    *options,
                ],
                capture_output=True,
                text=True,
            )

    def test_windows_shared_core_exports_symbols(self):
        result = self.configure(
            "-DBUILD_SHARED_LIBS=ON",
            assertions="""
get_target_property(exports pnmir WINDOWS_EXPORT_ALL_SYMBOLS)
if(NOT exports)
  message(FATAL_ERROR "Windows shared runtime must export public symbols")
endif()
""",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_windows_tensorrt_versioned_import_libraries(self):
        result = self.configure("-DPNMIR_ENABLE_TENSORRT=ON", tensorrt=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_windows_exact_tensorrt_requires_pinned_pytorch_headers(self):
        result = self.configure(
            "-DPNMIR_ENABLE_TENSORRT=ON", "-DPNMIR_ENABLE_TENSORRT_EXACT=ON"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "PNMIR_PYTORCH_SOURCE_ROOT must contain",
            result.stdout + result.stderr,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdk-source", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    WindowsConfigurationTests.sdk_source = args.sdk_source.resolve()
    unittest.main(argv=[__file__, *remaining])
