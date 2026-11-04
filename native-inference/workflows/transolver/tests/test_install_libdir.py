"""Check production install rules with CPU targets and relocated install paths."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]


class InstallLibraryDirectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.name == "nt":
            raise unittest.SkipTest("POSIX RPATH checks do not apply on Windows")
        if shutil.which("cmake") is None or shutil.which("c++") is None:
            raise unittest.SkipTest("CMake and a C++ compiler are required")

    def check_install(self, libdir, bindir="bin", workflow="transolver"):
        production = (WORKFLOW_ROOT.parent / workflow / "CMakeLists.txt").read_text()

        def block(start, end):
            return production[production.index(start) : production.index(end)]

        # Replace only the CUDA/Torch target setup. Execute the production
        # RPATH properties and install commands unchanged on real CPU targets.
        if workflow == "transolver":
            library = "transolver_workflow"
            executables = {
                "physicsnemo-transolver": "physicsnemo-transolver",
                "transolver_runner_tests": "physicsnemo-transolver-runner",
            }
            install_rules = production.rsplit("endforeach()", 1)[1].split(
                "include(CTest)", 1
            )[0] + block(
                "  set_target_properties(transolver_runner_tests PROPERTIES",
                "  add_test(NAME transolver_runner_tests",
            )
        else:
            library = "physicsnemo_workflow"
            executables = {"pnms_native_run": "pnms-native-run"}
            install_rules = (
                block("include(GNUInstallDirs)", "option(PNMS_ENABLE_AOTI")
                + block(
                    "set_target_properties(physicsnemo_workflow PROPERTIES EXPORT_NAME",
                    "if(PNMS_ENABLE_MOCK_BACKEND)",
                )
                + block(
                    "set_target_properties(pnms_native_run PROPERTIES OUTPUT_NAME",
                    "target_link_libraries(pnms_native_run",
                )
                + block(
                    "install(TARGETS physicsnemo_workflow physicsnemo_workflow_authoring",
                    "install(DIRECTORY include/",
                )
            )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, build = root / "source", root / "build"
            prefix, relocated = root / "installed", root / "relocated"
            source.mkdir()
            headers = source / "include" / "transolver_workflow"
            headers.mkdir(parents=True)
            (headers / "probe.h").write_text("int workflow_answer();\n")
            (source / "workflow.cpp").write_text(
                "int workflow_answer() { return 42; }\n"
            )
            (source / "main.cpp").write_text(
                "int workflow_answer();\n"
                "int main() { return workflow_answer() == 42 ? 0 : 1; }\n"
            )
            (source / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.24)\n"
                "project(TransolverInstallProbe LANGUAGES CXX)\n"
                f"add_library({library} SHARED workflow.cpp)\n"
                "add_library(physicsnemo_workflow_authoring INTERFACE)\n"
                + "".join(
                    f"add_executable({target} main.cpp)\n"
                    f"target_link_libraries({target} PRIVATE {library})\n"
                    for target in executables
                )
                + install_rules
                + f"\nforeach(target IN ITEMS {library} {' '.join(executables)})\n"
                + """
  file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/${target}.rpath"
    CONTENT "$<TARGET_PROPERTY:${target},INSTALL_RPATH>")
endforeach()
"""
            )
            commands = (
                [
                    "cmake",
                    "-S",
                    str(source),
                    "-B",
                    str(build),
                    f"-DCMAKE_INSTALL_PREFIX={prefix}",
                    f"-DCMAKE_INSTALL_LIBDIR={libdir}",
                    f"-DCMAKE_INSTALL_BINDIR={bindir}",
                ],
                ["cmake", "--build", str(build)],
                ["cmake", "--install", str(build)],
            )
            for command in commands:
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            manifest = (build / "install_manifest.txt").read_text().splitlines()
            installed = {
                library: next(
                    Path(path)
                    for path in manifest
                    if Path(path).name.startswith(f"lib{library}.")
                ),
                **{
                    target: prefix / bindir / name
                    for target, name in executables.items()
                },
            }
            rpaths = {
                target: (build / f"{target}.rpath").read_text().split(";")
                for target in installed
            }
            shutil.move(prefix, relocated)
            installed = {
                target: relocated / path.relative_to(prefix)
                for target, path in installed.items()
            }
            # Removing the build tree ensures Linux cannot fall back to its
            # build RPATH when the installed CLI loads the workflow library.
            shutil.rmtree(build)
            library_dir = installed[library].parent.resolve()
            self.assertEqual(library_dir, (relocated / libdir).resolve())
            for target, path in installed.items():
                self.assertTrue(path.is_file(), str(path))
                search_dirs = [
                    Path(
                        entry.replace("$ORIGIN", str(path.parent)).replace(
                            "@loader_path", str(path.parent)
                        )
                    ).resolve()
                    for entry in rpaths[target]
                ]
                self.assertIn(
                    library_dir,
                    search_dirs,
                    f"{target} cannot find its installed library directory via INSTALL_RPATH",
                )

            if sys.platform.startswith("linux") or workflow == "runtime":
                environment = os.environ.copy()
                for name in (
                    "LD_LIBRARY_PATH",
                    "LD_PRELOAD",
                    "DYLD_LIBRARY_PATH",
                    "DYLD_FALLBACK_LIBRARY_PATH",
                    "DYLD_INSERT_LIBRARIES",
                ):
                    environment.pop(name, None)
                for target in executables:
                    result = subprocess.run(
                        [str(installed[target])],
                        env=environment,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )

    def test_default_library_directory(self):
        self.check_install("lib")

    def test_lib64_library_directory(self):
        self.check_install("lib64")

    def test_custom_binary_directory(self):
        self.check_install("lib64", "tools/bin")

    def test_runtime_default_library_directory(self):
        self.check_install("lib", workflow="runtime")

    def test_runtime_lib64_library_directory(self):
        self.check_install("lib64", workflow="runtime")

    def test_runtime_custom_binary_directory(self):
        self.check_install("lib64", "tools/bin", workflow="runtime")


if __name__ == "__main__":
    unittest.main()
