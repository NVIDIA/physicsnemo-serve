"""Environment bootstrap must select the requested interpreter without Docker."""

import contextlib
from importlib import metadata
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_builder.build import cli, environment, scaffold


class SetupEnvironmentCommandTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        # These command tests model POSIX by default; Windows cases override
        # the platform explicitly. External tools are simulated on every host.
        platform = mock.patch.object(sys, "platform", "linux")
        platform.start()
        self.addCleanup(platform.stop)
        self.cmake = self.root / "cmake"
        self.make_executable(self.cmake)
        which = mock.patch(
            "shutil.which",
            side_effect=lambda name: (
                str(self.cmake)
                if name == "cmake"
                else (str(name) if Path(name).is_file() else None)
            ),
        )
        which.start()
        self.addCleanup(which.stop)
        self.destination = self.root / "new environment"
        self.selected_python = self.root / "customer python"
        self.make_executable(self.selected_python)
        self.runtime = self.root / "prebuilt runtime" / "physicsnemo-infer"
        self.make_executable(self.runtime)
        self.project = self.root / "customer model"
        scaffold.initialize(self.project)
        self.project_file = self.project / "model-build.json"
        document = json.loads(self.project_file.read_text())
        document["builder_image"] = "sha256:" + "a" * 64
        document["backends"] = ["aoti", "tensorrt"]
        self.project_file.write_text(json.dumps(document))
        self.project_bytes = self.project_file.read_bytes()

    @staticmethod
    def make_executable(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)

    def invoke(self, *options, behavior=None):
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            mock.patch("subprocess.run", side_effect=behavior or self.success) as run,
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            code = cli.main(["setup-env", str(self.destination), *options, "--json"])
        return code, json.loads(stdout.getvalue()), stderr.getvalue(), run

    def success(self, command, **kwargs):
        # Simulate external installers/build tools without downloading packages
        # or requiring Torch, a compiler, or a GPU in this command-level suite.
        command = [str(value) for value in command]
        if "venv" in command:
            self.make_executable(self.destination / "bin" / "python")
        if "pip" in command and "install" in command:
            self.make_executable(self.destination / "bin" / "pnms-model-builder")
        if "--build" in command and "cmake" in Path(command[0]).name:
            self.make_executable(
                self.destination / ".physicsnemo" / "runtime" / "physicsnemo-infer"
            )
        if "build" in command:
            return subprocess.CompletedProcess(
                command, 0, stdout=json.dumps({"status": "complete"}), stderr=""
            )
        output = kwargs.get("stdout")
        if hasattr(output, "write"):
            output.write("installer/build-tool diagnostic\n")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    def test_setup_env_is_available_as_a_single_command(self):
        code, result, _, run = self.invoke("--runtime", str(self.runtime))
        self.assertEqual(code, 0, result)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["backends"], ["aoti"])
        self.assertEqual(result["python"], str(self.destination / "bin" / "python"))
        self.assertEqual(
            result["project_settings"],
            {"executor": "local", "runtime": str(self.runtime)},
        )
        self.assertTrue(
            (self.destination / ".physicsnemo" / "environment.json").is_file()
        )
        self.assertFalse(
            any(
                Path(str(call.args[0][0])).name == "docker"
                for call in run.call_args_list
            )
        )

    def test_selected_python_creates_venv_and_installs_with_venv_python(self):
        requirements = self.root / "customer requirements.txt"
        requirements.write_text("customer-model-dependency==1.0\n")
        code, result, _, run = self.invoke(
            "--python",
            str(self.selected_python),
            "--requirements",
            str(requirements),
            "--runtime",
            str(self.runtime),
        )
        self.assertEqual(code, 0, result)
        commands = [
            [str(value) for value in call.args[0]] for call in run.call_args_list
        ]
        creation = next(command for command in commands if "venv" in command)
        self.assertEqual(creation[0], str(self.selected_python))
        self.assertEqual(creation[-1], str(self.destination))
        installs = [command for command in commands if "pip" in command]
        self.assertTrue(installs)
        self.assertTrue(
            all(
                command[0] == str(self.destination / "bin" / "python")
                for command in installs
            )
        )
        self.assertTrue(any(str(requirements) in command for command in installs))

    def test_supplied_runtime_skips_cpp_bootstrap(self):
        code, result, _, run = self.invoke("--runtime", str(self.runtime))
        self.assertEqual(code, 0, result)
        self.assertEqual(result["runtime"], str(self.runtime))
        self.assertFalse(
            any(
                "cmake" in Path(str(call.args[0][0])).name
                for call in run.call_args_list
            )
        )

    def test_source_staging_preserves_nested_build_and_dist_packages(self):
        package = self.root / "builder source"
        retained = {
            "pyproject.toml": '[project]\nname = "model-builder"\nversion = "1.0"\n',
            "src/model_builder/__init__.py": "# package\n",
            "src/model_builder/build/__init__.py": "# build package\n",
            "src/model_builder/build/module.py": "VALUE = 'build source'\n",
            "src/model_builder/dist/module.py": "VALUE = 'nested source'\n",
        }
        excluded = {
            "build/generated.py": "# generated build output\n",
            "dist/package.whl": "generated wheel",
            ".pytest_cache/state": "cached result",
            "src/model_builder/build/__pycache__/module.pyc": "cached bytecode",
            "src/model_builder/package.egg-info/PKG-INFO": "generated metadata",
        }
        for relative, content in (retained | excluded).items():
            path = package / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

        code, result, _, _ = self.invoke(
            "--builder-package", str(package), "--runtime", str(self.runtime)
        )
        self.assertEqual(code, 0, result)
        staged = self.destination / ".physicsnemo" / "builder-source"
        self.assertEqual(
            {
                path.relative_to(staged).as_posix()
                for path in staged.rglob("*")
                if path.is_file()
            },
            set(retained),
        )
        for relative, content in retained.items():
            self.assertEqual((staged / relative).read_text(), content)
        for relative, content in (retained | excluded).items():
            self.assertEqual((package / relative).read_text(), content)

    def test_checkout_bootstraps_sdk_once_with_venv_python(self):
        code, result, _, run = self.invoke("--python", str(self.selected_python))
        self.assertEqual(code, 0, result)
        commands = [
            [str(value) for value in call.args[0]] for call in run.call_args_list
        ]
        cmake = [command for command in commands if "cmake" in Path(command[0]).name]
        configure = next(command for command in cmake if "-S" in command)
        self.assertIn(
            f"-DPython3_EXECUTABLE={self.destination / 'bin' / 'python'}", configure
        )
        self.assertIn("-DPNMIR_ENABLE_AOTI=ON", configure)
        self.assertIn("-DPNMIR_ENABLE_TENSORRT=OFF", configure)
        self.assertEqual(sum("--build" in command for command in cmake), 1)
        self.assertFalse(any("--install" in command for command in cmake))
        expected_runtime = (
            self.destination / ".physicsnemo" / "runtime" / "physicsnemo-infer"
        )
        self.assertEqual(result["runtime"], str(expected_runtime))

    def fake_tensorrt_sdk(self):
        trt_root = self.root / "TensorRT SDK"
        include = trt_root / "include"
        include.mkdir(parents=True)
        (include / "NvInfer.h").write_text("// test SDK header\n")
        (include / "NvInferVersion.h").write_text(
            "#define NV_TENSORRT_MAJOR 10\n"
            "#define NV_TENSORRT_MINOR 14\n"
            "#define NV_TENSORRT_PATCH 1\n"
            "#define NV_TENSORRT_BUILD 48\n"
        )
        return trt_root

    def invoke_windows(self, *options, behavior=None):
        python = sys.executable
        cmake = self.root / "cmake.exe"
        self.make_executable(cmake)
        with (
            mock.patch.object(sys, "platform", "win32"),
            mock.patch(
                "shutil.which",
                side_effect=lambda name: str(cmake) if name == "cmake" else python,
            ),
        ):
            return self.invoke(*options, behavior=behavior or self.windows_success)

    def windows_success(self, command, **kwargs):
        command = [str(value) for value in command]
        if "venv" in command:
            self.make_executable(self.destination / "Scripts" / "python.exe")
            (self.destination / "Scripts" / "Activate.ps1").write_text(
                "# venv activation\n"
            )
        if "pip" in command and "install" in command:
            self.make_executable(
                self.destination / "Scripts" / "pnms-model-builder.exe"
            )
        if "--build" in command and "cmake" in Path(command[0]).name:
            self.make_executable(
                self.destination
                / ".physicsnemo"
                / "runtime"
                / "Release"
                / "physicsnemo-infer.exe"
            )
        if "build" in command:
            return subprocess.CompletedProcess(
                command, 0, stdout=json.dumps({"status": "complete"}), stderr=""
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    def test_windows_tensorrt_builds_release_executable(self):
        trt_root = self.fake_tensorrt_sdk()
        (trt_root / "bin").mkdir()
        (trt_root / "lib").mkdir()
        cuda_root = self.root / "CUDA Toolkit"
        (cuda_root / "bin").mkdir(parents=True)
        with mock.patch.dict("os.environ", {"CUDA_PATH": str(cuda_root)}):
            code, result, _, run = self.invoke_windows(
                "--backend", "tensorrt", "--tensorrt-root", str(trt_root)
            )
        self.assertEqual(code, 0, result)
        self.assertEqual(result["backends"], ["tensorrt"])
        self.assertEqual(
            result["python"], str(self.destination / "Scripts" / "python.exe")
        )
        self.assertEqual(
            result["runtime"],
            str(
                self.destination / ".physicsnemo/runtime/Release/physicsnemo-infer.exe"
            ),
        )
        commands = [call.args[0] for call in run.call_args_list]
        configure = next(command for command in commands if "-S" in command)
        self.assertEqual(configure[configure.index("-G") + 1], "Visual Studio 17 2022")
        self.assertEqual(configure[configure.index("-A") + 1], "x64")
        self.assertIn("-DPNMIR_ENABLE_AOTI=OFF", configure)
        build = next(command for command in commands if "--build" in command)
        self.assertEqual(build[build.index("--config") + 1], "Release")
        self.assertEqual(
            result["build_command"][:3],
            [result["python"], "-m", "model_builder.build.cli"],
        )
        for call in run.call_args_list[1:]:
            self.assertIn(str(trt_root / "bin"), call.kwargs["env"]["PATH"])
            self.assertIn(str(trt_root / "lib"), call.kwargs["env"]["PATH"])
            self.assertIn(str(cuda_root / "bin"), call.kwargs["env"]["PATH"])
            self.assertIn(
                str(self.destination / "Lib/site-packages/torch/lib"),
                call.kwargs["env"]["PATH"],
            )
        activation = self.destination / ".physicsnemo" / "activate.ps1"
        self.assertEqual(result["activate"], f". '{activation}'")
        self.assertIn(
            str(self.destination / "Scripts" / "Activate.ps1"), activation.read_text()
        )
        self.assertIn(str(trt_root / "bin"), activation.read_text())
        self.assertIn(str(cuda_root / "bin"), activation.read_text())
        # Windows PowerShell 5.1 needs a BOM to decode Unicode paths as UTF-8.
        self.assertTrue(activation.read_bytes().startswith(b"\xef\xbb\xbf"))

    def test_windows_default_aoti_builds_matching_runtime(self):
        code, result, _, run = self.invoke_windows()
        self.assertEqual(code, 0, result)
        self.assertEqual(result["backends"], ["aoti"])
        configure = next(
            call.args[0] for call in run.call_args_list if "-S" in call.args[0]
        )
        self.assertIn("-DPNMIR_ENABLE_AOTI=ON", configure)
        self.assertIn("-DPNMIR_ENABLE_TENSORRT=OFF", configure)

    def test_windows_forwards_caller_generator_toolset_and_instance(self):
        toolset = r"cuda=C:\CUDA SDK\cuda-12.8,host=x64"
        instance = r"C:\Build Tools\2022"
        with mock.patch.dict(
            "os.environ",
            {
                "CMAKE_GENERATOR_TOOLSET": toolset,
                "CMAKE_GENERATOR_INSTANCE": instance,
            },
        ):
            code, result, _, run = self.invoke_windows()
        self.assertEqual(code, 0, result)
        configure = next(
            call.args[0] for call in run.call_args_list if "-S" in call.args[0]
        )
        self.assertIn("-T", configure)
        self.assertEqual(configure[configure.index("-T") + 1], toolset)
        self.assertIn(f"-DCMAKE_GENERATOR_INSTANCE={instance}", configure)

    def test_windows_keeps_generator_defaults_without_caller_preferences(self):
        for index, preferences in enumerate(
            (
                {},
                {
                    "CMAKE_GENERATOR_TOOLSET": "",
                    "CMAKE_GENERATOR_INSTANCE": "",
                },
            )
        ):
            with self.subTest(preferences=preferences):
                self.destination = self.root / f"environment-{index}"
                with mock.patch.dict("os.environ", preferences, clear=True):
                    code, result, _, run = self.invoke_windows()
                self.assertEqual(code, 0, result)
                configure = next(
                    call.args[0] for call in run.call_args_list if "-S" in call.args[0]
                )
                self.assertNotIn("-T", configure)
                self.assertFalse(
                    any(
                        str(value).startswith("-DCMAKE_GENERATOR_INSTANCE=")
                        for value in configure
                    )
                )

    def test_windows_preserves_both_backends_from_project_settings(self):
        code, result, _, run = self.invoke_windows(
            "--build", str(self.project), "--runtime", str(self.runtime)
        )
        self.assertEqual(code, 0, result)
        self.assertEqual(result["backends"], ["aoti", "tensorrt"])
        build = next(
            call.args[0] for call in run.call_args_list if "build" in call.args[0]
        )
        self.assertIn("aoti", build)
        self.assertIn("tensorrt", build)

    def test_windows_project_build_uses_python_module_and_explicit_tensorrt(self):
        code, result, _, run = self.invoke_windows(
            "--build",
            str(self.project),
            "--runtime",
            str(self.runtime),
            "--backend",
            "tensorrt",
        )
        self.assertEqual(code, 0, result)
        build = next(
            call.args[0] for call in run.call_args_list if "build" in call.args[0]
        )
        self.assertEqual(
            build[:3],
            [
                str(self.destination / "Scripts" / "python.exe"),
                "-m",
                "model_builder.build.cli",
            ],
        )
        self.assertEqual(result["build"]["status"], "complete")
        self.assertEqual(self.project_file.read_bytes(), self.project_bytes)

    def test_tensorrt_sdk_version_pins_python_bindings_and_enables_both_backends(self):
        trt_root = self.fake_tensorrt_sdk()
        code, result, _, run = self.invoke(
            "--backend",
            "aoti",
            "--backend",
            "tensorrt",
            "--tensorrt-root",
            str(trt_root),
        )
        self.assertEqual(code, 0, result)
        commands = [
            [str(value) for value in call.args[0]] for call in run.call_args_list
        ]
        install = next(
            command for command in commands if "pip" in command and "install" in command
        )
        self.assertIn("tensorrt-cu13==10.14.1.48", install)
        self.assertTrue(any(value.endswith("[export]") for value in install))
        self.assertFalse(any(value.endswith("[tensorrt]") for value in install))
        configure = next(
            command
            for command in commands
            if "cmake" in Path(command[0]).name and "-S" in command
        )
        self.assertIn("-DPNMIR_ENABLE_AOTI=ON", configure)
        self.assertIn("-DPNMIR_ENABLE_TENSORRT=ON", configure)
        self.assertIn(f"-DPNMIR_TENSORRT_ROOT={trt_root}", configure)

    def test_explicit_cuda12_selects_only_cuda12_tensorrt_wheels(self):
        trt_root = self.fake_tensorrt_sdk()
        code, result, _, run = self.invoke(
            "--backend",
            "tensorrt",
            "--tensorrt-root",
            str(trt_root),
            "--tensorrt-cuda-major",
            "12",
        )
        self.assertEqual(code, 0, result)
        commands = [
            [str(value) for value in call.args[0]] for call in run.call_args_list
        ]
        install = next(
            command for command in commands if "pip" in command and "install" in command
        )
        self.assertIn("tensorrt-cu12==10.14.1.48", install)
        self.assertFalse(any("tensorrt-cu13" in value for value in install))
        self.assertFalse(any(value.startswith("tensorrt==") for value in install))
        self.assertFalse(any(value.endswith("[tensorrt]") for value in install))

    def test_windows_tensorrt_enterprise_version_aliases_pin_matching_wheels(self):
        trt_root = self.fake_tensorrt_sdk()
        (trt_root / "include" / "NvInferVersion.h").write_text(
            "#define TRT_MAJOR_ENTERPRISE 10\n"
            "#define TRT_MINOR_ENTERPRISE 13\n"
            "#define TRT_PATCH_ENTERPRISE 3\n"
            "#define TRT_BUILD_ENTERPRISE 9\n"
            "#define NV_TENSORRT_MAJOR TRT_MAJOR_ENTERPRISE //!< TensorRT major version.\n"
            "#define NV_TENSORRT_MINOR TRT_MINOR_ENTERPRISE //!< TensorRT minor version.\n"
            "#define NV_TENSORRT_PATCH TRT_PATCH_ENTERPRISE //!< TensorRT patch version.\n"
            "#define NV_TENSORRT_BUILD TRT_BUILD_ENTERPRISE //!< TensorRT build number.\n"
        )
        code, result, _, run = self.invoke_windows(
            "--backend",
            "tensorrt",
            "--tensorrt-root",
            str(trt_root),
            "--tensorrt-cuda-major",
            "12",
        )
        self.assertEqual(code, 0, result)
        commands = [call.args[0] for call in run.call_args_list]
        install = next(
            command for command in commands if "pip" in command and "install" in command
        )
        self.assertIn("tensorrt-cu12==10.13.3.9", install)
        probe = next(command for command in commands if "-c" in command)
        self.assertIn("10.13.3.9", probe)

    def test_invalid_tensorrt_version_macros_fail_before_environment_creation(self):
        trt_root = self.fake_tensorrt_sdk()
        header = trt_root / "include" / "NvInferVersion.h"
        valid = header.read_text()
        for definition in (
            "#define NV_TENSORRT_MAJOR MISSING_MAJOR",
            "#define NV_TENSORRT_MAJOR TRT_MAJOR\n#define TRT_MAJOR NV_TENSORRT_MAJOR",
            "#define NV_TENSORRT_MAJOR 10 + 1",
        ):
            with self.subTest(definition=definition):
                header.write_text(
                    valid.replace("#define NV_TENSORRT_MAJOR 10", definition)
                )
                code, result, _, run = self.invoke_windows(
                    "--backend", "tensorrt", "--tensorrt-root", str(trt_root)
                )
                self.assertEqual(code, 2, result)
                self.assertIn(
                    "TensorRT 10 or newer", result["diagnostics"][0]["message"]
                )
                run.assert_not_called()
                self.assertFalse(self.destination.exists())

    def test_failed_dependency_probe_prevents_sdk_bootstrap(self):
        def fail_probe(command, **kwargs):
            if "-c" in command:
                kwargs["stdout"].write("mixed TensorRT CUDA distributions\n")
                return subprocess.CompletedProcess(command, 1)
            return self.success(command, **kwargs)

        code, result, _, run = self.invoke(behavior=fail_probe)
        self.assertNotEqual(code, 0, result)
        self.assertEqual(result["status"], "failed")
        probes = [call.args[0] for call in run.call_args_list if "-c" in call.args[0]]
        self.assertTrue(probes)
        self.assertTrue(
            all(
                command[0] == str(self.destination / "bin" / "python")
                for command in probes
            )
        )
        self.assertFalse(
            any(
                "cmake" in Path(str(call.args[0][0])).name
                for call in run.call_args_list
            )
        )
        self.assertFalse(
            (self.destination / ".physicsnemo" / "environment.json").exists()
        )

    def execute_dependency_probe(
        self,
        installed_variants,
        *,
        cuda="12.8",
        sdk_version="",
        tensorrt=True,
        aoti=False,
        triton=True,
    ):
        modules = {}
        for name in (
            "model_builder.build.cli",
            "torch",
            "numpy",
            "onnx",
            "onnxscript",
            "tensorrt",
            "triton",
        ):
            module = ModuleType(name)
            module.__version__ = "10.14.1.48"
            modules[name] = module
        modules["torch"].version = SimpleNamespace(cuda=cuda)
        if not triton:
            modules["triton"] = None

        def installed_version(name):
            normalized = name.lower().replace("_", "-")
            if normalized in installed_variants:
                return "10.14.1.48"
            raise metadata.PackageNotFoundError(name)

        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.object(
                sys,
                "argv",
                ["-c", "12" if tensorrt else "", sdk_version, "aoti" if aoti else ""],
            ),
            mock.patch.object(metadata, "version", side_effect=installed_version),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            exec(compile(environment._DEPENDENCY_PROBE, "dependency-probe", "exec"), {})

    def test_actual_dependency_probe_rejects_mixed_tensorrt_cuda_variants(self):
        with self.assertRaisesRegex(RuntimeError, "Conflicting TensorRT CUDA packages"):
            self.execute_dependency_probe({"tensorrt-cu12", "tensorrt-cu13"})

    def test_actual_dependency_probe_accepts_only_the_selected_cuda_variant(self):
        self.execute_dependency_probe({"tensorrt-cu12"})

    def test_windows_dependency_probe_rejects_cpu_only_torch(self):
        with mock.patch.object(sys, "platform", "win32"):
            with self.assertRaisesRegex(RuntimeError, "CUDA-enabled PyTorch"):
                self.execute_dependency_probe({"tensorrt-cu12"}, cuda=None)

    def test_windows_cuda_aoti_requires_triton_in_new_environment(self):
        with mock.patch.object(sys, "platform", "win32"):
            with self.assertRaisesRegex(RuntimeError, "triton-windows"):
                self.execute_dependency_probe(
                    set(), tensorrt=False, aoti=True, triton=False
                )

    def test_windows_cpu_aoti_does_not_require_cuda_or_triton(self):
        with mock.patch.object(sys, "platform", "win32"):
            self.execute_dependency_probe(
                set(), tensorrt=False, aoti=True, cuda=None, triton=False
            )

    def test_dependency_probe_rejects_tensorrt_sdk_version_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "TensorRT Python version.*SDK"):
            self.execute_dependency_probe({"tensorrt-cu12"}, sdk_version="10.13.0.35")

    def test_missing_installed_builder_entrypoint_cannot_report_ready(self):
        unrelated = self.root / "unrelated package"
        unrelated.mkdir()
        (unrelated / "pyproject.toml").write_text(
            '[project]\nname = "unrelated-package"\nversion = "1.0"\n'
        )

        def missing_entrypoint(command, **kwargs):
            completed = self.success(command, **kwargs)
            if "pip" in command and "install" in command:
                (self.destination / "bin" / "pnms-model-builder").unlink()
            return completed

        code, result, _, _ = self.invoke(
            "--runtime",
            str(self.runtime),
            "--builder-package",
            str(unrelated),
            behavior=missing_entrypoint,
        )
        self.assertNotEqual(code, 0, result)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(
            (self.destination / ".physicsnemo" / "environment.json").exists()
        )

    def test_missing_tensorrt_headers_fail_before_creating_environment(self):
        trt_root = self.root / "TensorRT without headers"
        trt_root.mkdir()
        code, result, _, run = self.invoke(
            "--backend", "tensorrt", "--tensorrt-root", str(trt_root)
        )
        self.assertEqual(code, 2, result)
        self.assertIn("NvInfer.h", result["diagnostics"][0]["message"])
        run.assert_not_called()
        self.assertFalse(self.destination.exists())

    def test_outer_python_import_settings_cannot_redirect_new_environment(self):
        with mock.patch.dict(
            "os.environ",
            {"PYTHONPATH": "/outer/python/packages", "PYTHONHOME": "/outer/python"},
        ):
            code, result, _, run = self.invoke("--runtime", str(self.runtime))
        self.assertEqual(code, 0, result)
        for call in run.call_args_list:
            self.assertNotIn("PYTHONPATH", call.kwargs["env"])
            self.assertNotIn("PYTHONHOME", call.kwargs["env"])

    def test_existing_environment_is_not_overwritten(self):
        self.destination.mkdir()
        retained = self.destination / "customer.txt"
        retained.write_text("keep my environment")
        code, result, _, run = self.invoke("--runtime", str(self.runtime))
        self.assertEqual(code, 2, result)
        run.assert_not_called()
        self.assertEqual(retained.read_text(), "keep my environment")

    def test_missing_supplied_runtime_is_rejected_before_environment_creation(self):
        self.runtime.unlink()
        code, result, _, run = self.invoke("--runtime", str(self.runtime))
        self.assertEqual(code, 2, result)
        run.assert_not_called()
        self.assertFalse(self.destination.exists())

    def test_dangling_environment_symlink_is_rejected(self):
        self.destination.symlink_to(
            self.root / "unrelated destination", target_is_directory=True
        )
        code, result, _, run = self.invoke("--runtime", str(self.runtime))
        self.assertEqual(code, 2, result)
        run.assert_not_called()
        self.assertTrue(self.destination.is_symlink())

    def test_build_uses_project_backends_and_explicit_local_executor_without_edits(
        self,
    ):
        code, result, _, run = self.invoke(
            "--runtime", str(self.runtime), "--build", str(self.project)
        )
        self.assertEqual(code, 0, result)
        self.assertEqual(result["backends"], ["aoti", "tensorrt"])
        self.assertEqual(result["build"]["status"], "complete")
        commands = [
            [str(value) for value in call.args[0]] for call in run.call_args_list
        ]
        build = next(command for command in commands if "build" in command)
        self.assertEqual(
            build[:2],
            [
                str(self.destination / "bin" / "python"),
                str(self.destination / "bin" / "pnms-model-builder"),
            ],
        )
        self.assertEqual(build[build.index("--executor") + 1], "local")
        self.assertEqual(build[build.index("--runtime") + 1], str(self.runtime))
        self.assertIn(str(self.project), build)
        self.assertEqual(self.project_file.read_bytes(), self.project_bytes)
        self.assertFalse((self.project / "model-build.lock.json").exists())

    def test_explicit_backend_selection_overrides_build_project_backends(self):
        code, result, _, run = self.invoke(
            "--runtime",
            str(self.runtime),
            "--build",
            str(self.project),
            "--backend",
            "aoti",
        )
        self.assertEqual(code, 0, result)
        self.assertEqual(result["backends"], ["aoti"])
        build = next(
            call.args[0] for call in run.call_args_list if "build" in call.args[0]
        )
        self.assertEqual(build[build.index("--backend") + 1], "aoti")
        self.assertNotIn("tensorrt", build)
        self.assertEqual(self.project_file.read_bytes(), self.project_bytes)

    def test_failed_dependency_install_does_not_claim_ready_or_start_build(self):
        def fail_install(command, **kwargs):
            if "pip" in command and "install" in command:
                kwargs["stdout"].write("dependency resolution failed\n")
                return subprocess.CompletedProcess(command, 1)
            return self.success(command, **kwargs)

        code, result, _, run = self.invoke(
            "--runtime",
            str(self.runtime),
            "--build",
            str(self.project),
            behavior=fail_install,
        )
        self.assertNotEqual(code, 0)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(any("build" in call.args[0] for call in run.call_args_list))
        self.assertFalse(
            (self.destination / ".physicsnemo" / "environment.json").exists()
        )
        self.assertEqual(self.project_file.read_bytes(), self.project_bytes)

    def test_failed_model_build_keeps_child_diagnostics_in_json_result(self):
        def fail_build(command, **kwargs):
            if "build" in command:
                return subprocess.CompletedProcess(
                    command,
                    2,
                    stdout=json.dumps(
                        {
                            "status": "incomplete",
                            "diagnostics": [
                                {
                                    "code": "AUTHORING_INPUT_REQUIRED",
                                    "field": "checkpoint",
                                }
                            ],
                        }
                    ),
                    stderr="",
                )
            return self.success(command, **kwargs)

        code, result, _, _ = self.invoke(
            "--runtime",
            str(self.runtime),
            "--build",
            str(self.project),
            behavior=fail_build,
        )
        self.assertNotEqual(code, 0)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["build"]["diagnostics"][0]["field"], "checkpoint")
        self.assertEqual(self.project_file.read_bytes(), self.project_bytes)


if __name__ == "__main__":
    unittest.main()
