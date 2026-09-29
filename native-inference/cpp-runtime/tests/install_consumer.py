"""Build a clean SDK, relocate it, and consume it without its source/build tree."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class InstallConsumerTests(unittest.TestCase):
    sdk_source: Path
    shared: bool
    component: str
    torch_root: Path | None
    onnxruntime_root: Path | None
    tensorrt_root: Path | None
    cuda_root: Path | None
    cuda_architectures: str | None
    runtime_prefix: Path | None = None

    def command(self, *argv, runtime=False):
        env = os.environ.copy()
        for name in (
            "PYTHONPATH",
            "CMAKE_PREFIX_PATH",
            "LD_LIBRARY_PATH",
            "DYLD_LIBRARY_PATH",
        ):
            env.pop(name, None)
        if runtime:
            # Unix SDK libraries must resolve via install RPATH. Windows has no
            # RPATH: resolve the relocated DLLs through this child process' PATH.
            library_dirs = []
            for root in (
                self.runtime_prefix if sys.platform == "win32" else None,
                self.torch_root,
                self.onnxruntime_root,
                self.tensorrt_root,
                self.cuda_root,
            ):
                if root is not None:
                    library_dirs.extend(
                        str(path)
                        for path in (
                            root / "lib",
                            root / "lib64",
                            root / "lib/x86_64-linux-gnu",
                            *((root / "bin",) if sys.platform == "win32" else ()),
                        )
                        if path.is_dir()
                    )
            if library_dirs:
                if sys.platform == "win32":
                    env["PATH"] = os.pathsep.join([*library_dirs, env.get("PATH", "")])
                else:
                    variable = (
                        "DYLD_LIBRARY_PATH"
                        if sys.platform == "darwin"
                        else "LD_LIBRARY_PATH"
                    )
                    env[variable] = os.pathsep.join(library_dirs)
        result = subprocess.run(
            argv, capture_output=True, text=True, env=env, check=False
        )
        self.assertEqual(
            result.returncode, 0, f"command: {argv!r}\n{result.stdout}\n{result.stderr}"
        )
        return result

    def dependency_options(self):
        options = []
        if self.onnxruntime_root is not None:
            options.append(
                f"-DPNMIR_ONNXRUNTIME_ROOT={self.onnxruntime_root.as_posix()}"
            )
        if self.tensorrt_root is not None:
            options.append(f"-DPNMIR_TENSORRT_ROOT={self.tensorrt_root.as_posix()}")
        if self.cuda_root is not None:
            nvcc = "nvcc.exe" if sys.platform == "win32" else "nvcc"
            options.extend(
                [
                    f"-DCUDAToolkit_ROOT={self.cuda_root.as_posix()}",
                    f"-DCUDA_TOOLKIT_ROOT_DIR={self.cuda_root.as_posix()}",
                    f"-DCMAKE_CUDA_COMPILER={(self.cuda_root / 'bin' / nvcc).as_posix()}",
                ]
            )
        if self.cuda_architectures is not None:
            architectures = self.cuda_architectures.replace(",", ";")
            options.append(f"-DCMAKE_CUDA_ARCHITECTURES={architectures}")
        return options

    def test_relocated_prefix_is_a_standalone_cpp_dependency(self):
        with tempfile.TemporaryDirectory(prefix="pnmir-install-") as temporary:
            root = Path(temporary)
            source = root / "source"
            shutil.copytree(
                self.sdk_source,
                source,
                ignore=shutil.ignore_patterns("__pycache__", "build*", "_build"),
            )
            build = root / "build"
            prefix = root / "initial-prefix"
            fixture = root / "identity.pnmir"
            shutil.copytree(source / "tests/fixtures/identity", fixture)
            backend_options = []
            if self.component != "core":
                backend_options.append(f"-DPNMIR_ENABLE_{self.component.upper()}=ON")
            self.command(
                "cmake",
                "-S",
                str(source),
                "-B",
                str(build),
                "-DPNMIR_BUILD_TESTS=OFF",
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DBUILD_SHARED_LIBS={'ON' if self.shared else 'OFF'}",
                f"-DCMAKE_INSTALL_PREFIX={prefix}",
                f"-DPython3_EXECUTABLE={sys.executable}",
                *backend_options,
                *self.dependency_options(),
            )
            self.command(
                "cmake", "--build", str(build), "--config", "Release", "--parallel", "2"
            )
            self.command("cmake", "--install", str(build), "--config", "Release")
            configs = list(prefix.rglob("PhysicsNeMoInferenceConfig.cmake"))
            self.assertEqual(
                len(configs),
                1,
                "installed SDK must expose one find_package(PhysicsNeMoInference CONFIG) entry point",
            )

            relocated = root / "relocated-prefix"
            prefix.rename(relocated)
            self.runtime_prefix = relocated
            shutil.rmtree(source)
            shutil.rmtree(build)
            consumer = root / "consumer"
            consumer.mkdir()
            target = "runtime" if self.component == "core" else self.component
            (consumer / "CMakeLists.txt").write_text(f"""\
cmake_minimum_required(VERSION 3.20)
project(pnmir_external_consumer LANGUAGES CXX)
find_package(PhysicsNeMoInference 0.1 CONFIG REQUIRED COMPONENTS {self.component})
add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE PhysicsNeMoInference::{target})
file(GENERATE OUTPUT "${{CMAKE_BINARY_DIR}}/consumer-$<CONFIG>.txt"
  CONTENT "$<TARGET_FILE:consumer>")
""")
            backend_include = ""
            backend_check = ""
            if self.component != "core":
                backend_include = (
                    f'#include "physicsnemo/inference/backends/{self.component}.hpp"'
                )
                backend_check = f"""\
  auto backend = physicsnemo::inference::create_{self.component}_backend();
  if (backend->name() != "{self.component}") return 5;
  engine.register_backend(std::move(backend));
"""
            (consumer / "main.cpp").write_text(
                """\
#include <iostream>
#include <utility>
#include <vector>
#include "physicsnemo/inference/backend.hpp"
#include "physicsnemo/inference/v1/api.hpp"
@BACKEND_INCLUDE@
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  physicsnemo::inference::v1::Engine engine;
@BACKEND_CHECK@
  engine.register_backend(physicsnemo::inference::create_mock_backend());
  auto executor = engine.load_model(argv[1]).create_executor();
  std::vector<float> input{3.0f, -1.0f, 4.0f};
  physicsnemo::inference::v1::Request request;
  request.bind_input({"input", physicsnemo::inference::DType::kFloat32, {}, {3},
                      input.data(), input.size() * sizeof(float)});
  auto result = executor.run(request);
  auto output = result.output("output");
  if (output.shape != physicsnemo::inference::Shape{3} || output.byte_size != 12) return 3;
  auto values = static_cast<const float*>(output.data);
  for (std::size_t i = 0; i < input.size(); ++i) if (values[i] != input[i]) return 4;
  std::cout << "installed API returned 3 -1 4\\n";
}
""".replace("@BACKEND_INCLUDE@", backend_include).replace(
                    "@BACKEND_CHECK@", backend_check
                )
            )
            downstream_build = root / "consumer-build"
            consumer_options = self.dependency_options()
            if self.torch_root is not None:
                consumer_options.append(
                    f"-DTorch_DIR={self.torch_root / 'share/cmake/Torch'}"
                )
            self.command(
                "cmake",
                "-S",
                str(consumer),
                "-B",
                str(downstream_build),
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DCMAKE_PREFIX_PATH={relocated}",
                *consumer_options,
            )
            self.command(
                "cmake",
                "--build",
                str(downstream_build),
                "--config",
                "Release",
                "--parallel",
                "2",
            )
            consumer_executable = (
                (downstream_build / "consumer-Release.txt").read_text().strip()
            )
            result = self.command(consumer_executable, str(fixture), runtime=True)
            self.assertEqual(result.stdout.strip(), "installed API returned 3 -1 4")
            cli = (
                "physicsnemo-infer.exe"
                if sys.platform == "win32"
                else "physicsnemo-infer"
            )
            result = self.command(
                str(relocated / "bin" / cli),
                "run",
                str(fixture),
                "--values",
                "3,-1,4",
                runtime=True,
            )
            self.assertEqual(result.stdout.strip(), "output: 3 -1 4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdk-source", required=True, type=Path)
    parser.add_argument("--shared", action="store_true")
    parser.add_argument(
        "--component",
        choices=("core", "aoti", "onnxruntime", "tensorrt"),
        default="core",
    )
    parser.add_argument("--torch-root", type=Path)
    parser.add_argument("--onnxruntime-root", type=Path)
    parser.add_argument("--tensorrt-root", type=Path)
    parser.add_argument("--cuda-root", type=Path)
    parser.add_argument("--cuda-architectures")
    args, remaining = parser.parse_known_args()
    InstallConsumerTests.sdk_source = args.sdk_source.resolve()
    InstallConsumerTests.shared = args.shared
    InstallConsumerTests.component = args.component
    InstallConsumerTests.torch_root = args.torch_root
    InstallConsumerTests.onnxruntime_root = args.onnxruntime_root
    InstallConsumerTests.tensorrt_root = args.tensorrt_root
    InstallConsumerTests.cuda_root = args.cuda_root
    InstallConsumerTests.cuda_architectures = args.cuda_architectures
    unittest.main(argv=[__file__, *remaining])
