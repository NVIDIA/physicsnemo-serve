"""Host-only CMake configuration and installed-interface checks.

Windows decision tests use a host compiler wrapper; they do not replace native
MSVC build and relocation checks. Exact-plugin tests need no CUDA toolchain.
"""

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class ExactConfigurationTests(unittest.TestCase):
    sdk_source: Path

    def command(self, *argv):
        result = subprocess.run(argv, capture_output=True, text=True)
        self.assertEqual(
            result.returncode, 0, f"command: {argv!r}\n{result.stdout}\n{result.stderr}"
        )
        return result

    def configure(self, *options):
        with tempfile.TemporaryDirectory(prefix="pnmir-exact-config-") as build:
            return subprocess.run(
                [
                    "cmake",
                    "-S",
                    str(self.sdk_source),
                    "-B",
                    build,
                    "-DPNMIR_BUILD_TESTS=OFF",
                    *options,
                ],
                capture_output=True,
                text=True,
            )

    def test_exact_requires_tensorrt_backend(self):
        result = self.configure("-DPNMIR_ENABLE_TENSORRT_EXACT=ON")
        self.assertNotEqual(
            result.returncode, 0, "exact plugins must not silently disable themselves"
        )
        self.assertIn(
            "requires PNMIR_ENABLE_TENSORRT=ON", result.stdout + result.stderr
        )

    def test_exact_requires_explicit_pytorch_headers(self):
        result = self.configure(
            "-DPNMIR_ENABLE_TENSORRT_EXACT=ON", "-DPNMIR_ENABLE_TENSORRT=ON"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "PNMIR_PYTORCH_SOURCE_ROOT must contain", result.stdout + result.stderr
        )

    def test_domino_sidecar_requires_aoti_backend(self):
        result = self.configure("-DPNMIR_BUILD_DOMINO_EXACT_OPS=ON")
        self.assertNotEqual(
            result.returncode, 0, "DoMINO exact operators must not be silently disabled"
        )
        self.assertIn("requires PNMIR_ENABLE_AOTI=ON", result.stdout + result.stderr)

    def test_exact_plugins_use_custom_installed_include_directory(self):
        # Exercise the production target helper with host-only bodies. This
        # checks exported usage requirements, not CUDA/TensorRT plugin behavior.
        with tempfile.TemporaryDirectory(prefix="pnmir-exact-install-") as temporary:
            root = Path(temporary)
            source = root / "source"
            plugin_sources = source / "src/tensorrt"
            plugin_sources.mkdir(parents=True)
            shutil.copytree(self.sdk_source / "include", source / "include")
            plugin_targets = []
            for plugin in (self.sdk_source / "src/tensorrt").glob(
                "tensorrt_exact_*_plugin.*"
            ):
                if plugin.suffix in (".cpp", ".cu"):
                    plugin_targets.append(f"PhysicsNeMoInference::{plugin.stem}")
                    (plugin_sources / plugin.name).write_text(
                        "int exact_plugin_stub() { return 0; }\n", encoding="utf-8"
                    )
            (source / "empty.cpp").write_text("", encoding="utf-8")
            (source / "CMakeLists.txt").write_text(
                """cmake_minimum_required(VERSION 3.20)
project(ExactInstallTest LANGUAGES CXX)
include(GNUInstallDirs)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
add_library(pnmir_tensorrt STATIC empty.cpp)
add_library(native_trt INTERFACE)
foreach(library IN ITEMS cudart cublas cublasLt)
  add_library(CUDA::${library} INTERFACE IMPORTED)
endforeach()
set(PNMIR_TENSORRT_LIBRARY native_trt)
set(PNMIR_TENSORRT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(PNMIR_PYTORCH_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(PNMIR_CUTLASS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(PNMIR_TORCH_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
file(GLOB plugin_sources src/tensorrt/*)
set_source_files_properties(${plugin_sources} PROPERTIES LANGUAGE CXX)
include("${SDK_SOURCE}/cmake/TensorRTExact.cmake")
install(TARGETS ${pnmir_tensorrt_exact_targets} EXPORT ExactTargets
  RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(EXPORT ExactTargets NAMESPACE PhysicsNeMoInference:: DESTINATION cmake)
""",
                encoding="utf-8",
            )
            build = root / "build"
            prefix = root / "prefix"
            self.command(
                "cmake",
                "-S",
                str(source),
                "-B",
                str(build),
                f"-DSDK_SOURCE={self.sdk_source.as_posix()}",
                f"-DCMAKE_INSTALL_PREFIX={prefix.as_posix()}",
                "-DCMAKE_INSTALL_INCLUDEDIR=headers",
                "-DCMAKE_BUILD_TYPE=Release",
            )
            self.command(
                "cmake", "--build", str(build), "--config", "Release", "--parallel", "2"
            )
            self.command("cmake", "--install", str(build), "--config", "Release")
            relocated = root / "relocated"
            prefix.rename(relocated)
            shutil.rmtree(source)
            shutil.rmtree(build)

            consumer = root / "consumer"
            consumer.mkdir()
            (consumer / "main.cpp").write_text(
                "#include <physicsnemo/inference/backends/tensorrt.hpp>\n"
                "int main() { return 0; }\n",
                encoding="utf-8",
            )
            (consumer / "CMakeLists.txt").write_text(
                """cmake_minimum_required(VERSION 3.20)
project(ExactConsumer LANGUAGES CXX)
include("${SDK_PREFIX}/cmake/ExactTargets.cmake")
add_executable(consumer main.cpp)
target_compile_features(consumer PRIVATE cxx_std_20)
"""
                + f"target_link_libraries(consumer PRIVATE {' '.join(plugin_targets)})\n",
                encoding="utf-8",
            )
            consumer_build = root / "consumer-build"
            self.command(
                "cmake",
                "-S",
                str(consumer),
                "-B",
                str(consumer_build),
                f"-DSDK_PREFIX={relocated.as_posix()}",
                "-DCMAKE_BUILD_TYPE=Release",
            )
            self.command("cmake", "--build", str(consumer_build), "--config", "Release")


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
    for case in (ExactConfigurationTests, WindowsConfigurationTests):
        case.sdk_source = args.sdk_source.resolve()
    unittest.main(argv=[__file__, *remaining])
