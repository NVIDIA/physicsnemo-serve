"""Build a clean SDK, relocate it, and consume it without its source/build tree."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class InstallConsumerTests(unittest.TestCase):
    sdk_source: Path
    shared: bool

    def command(self, *argv):
        env = os.environ.copy()
        for name in (
            "PYTHONPATH",
            "CMAKE_PREFIX_PATH",
            "LD_LIBRARY_PATH",
            "DYLD_LIBRARY_PATH",
        ):
            env.pop(name, None)
        result = subprocess.run(argv, capture_output=True, text=True, env=env)
        self.assertEqual(
            result.returncode, 0, f"command: {argv!r}\n{result.stdout}\n{result.stderr}"
        )
        return result

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
            )
            self.command("cmake", "--build", str(build), "--parallel", "2")
            self.command("cmake", "--install", str(build))
            configs = list(prefix.rglob("PhysicsNeMoInferenceConfig.cmake"))
            self.assertEqual(
                len(configs),
                1,
                "installed SDK must expose one find_package(PhysicsNeMoInference CONFIG) entry point",
            )

            relocated = root / "relocated-prefix"
            prefix.rename(relocated)
            shutil.rmtree(source)
            shutil.rmtree(build)
            consumer = root / "consumer"
            consumer.mkdir()
            (consumer / "CMakeLists.txt").write_text("""\
cmake_minimum_required(VERSION 3.20)
project(pnmir_external_consumer LANGUAGES CXX)
find_package(PhysicsNeMoInference 0.1 CONFIG REQUIRED COMPONENTS core)
add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE PhysicsNeMoInference::runtime)
""")
            (consumer / "main.cpp").write_text("""\
#include <iostream>
#include <vector>
#include "physicsnemo/inference/backend.hpp"
#include "physicsnemo/inference/v1/api.hpp"
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  physicsnemo::inference::v1::Engine engine;
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
""")
            downstream_build = root / "consumer-build"
            self.command(
                "cmake",
                "-S",
                str(consumer),
                "-B",
                str(downstream_build),
                f"-DCMAKE_PREFIX_PATH={relocated}",
            )
            self.command("cmake", "--build", str(downstream_build), "--parallel", "2")
            result = self.command(str(downstream_build / "consumer"), str(fixture))
            self.assertEqual(result.stdout.strip(), "installed API returned 3 -1 4")
            result = self.command(
                str(relocated / "bin/physicsnemo-infer"),
                "run",
                str(fixture),
                "--values",
                "3,-1,4",
            )
            self.assertEqual(result.stdout.strip(), "output: 3 -1 4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdk-source", required=True, type=Path)
    parser.add_argument("--shared", action="store_true")
    args, remaining = parser.parse_known_args()
    InstallConsumerTests.sdk_source = args.sdk_source.resolve()
    InstallConsumerTests.shared = args.shared
    unittest.main(argv=[__file__, *remaining])
