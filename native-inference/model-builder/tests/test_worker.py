from __future__ import annotations

from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import ModuleType, SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pnmir_build import worker


RUNTIME = """#!{python}
import json, struct, sys
from pathlib import Path
args=sys.argv[1:]
def option(flag): return args[args.index(flag)+1]
if {mode!r} == 'fail':
    print('native intentionally failed', file=sys.stderr)
    sys.exit(7)
source=Path(option('--input-file').split('=',1)[1]).read_bytes()
values=struct.unpack('<4f', source)
out=[2*x+1 for x in values]
if {mode!r} == 'nan': out[0]=float('nan')
Path(option('--output-file').split('=',1)[1]).write_bytes(struct.pack('<4f', *out))
metadata={{'schema_version':1,'completed':True,'backend':option('--backend'),
'execution_device':{{'type':'cpu','index':0}},
'outputs':[{{'name':'output','dtype':'float32','shape':[4],
'device':{{'type':'cpu','index':0}},'byte_size':16}}]}}
if {mode!r} == 'shape': metadata['outputs'][0]['shape']=[2,2]
if {mode!r} == 'backend': metadata['backend']='wrong'
if {mode!r} == 'device': metadata['execution_device']['type']='cuda'
if {mode!r} == 'bytes': metadata['outputs'][0]['byte_size']=20
Path(option('--output-metadata')).write_text(json.dumps(metadata))
"""


def tensor(name, values):
    return {
        "name": name,
        "dtype": "float32",
        "shape": [4],
        "data": struct.pack("<4f", *values),
    }


class WorkerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.recipe_path = self.root / "recipe.json"
        self.recipe = {
            "format_version": 1,
            "name": "affine",
            "version": "0.1.0",
            "adapter": "export.py",
            "factory": "create_model",
            "cases": "create_cases",
            "input_names": ["input"],
            "output_names": ["output"],
            "supported_backends": ["aoti", "tensorrt"],
            "default_backend": "aoti",
            "dtype": "float32",
            "shape": [4],
        }
        self.recipe_path.write_text(json.dumps(self.recipe))
        (self.root / "export.py").write_text("# frozen recipe adapter\n")
        self.runtime = self.root / "physicsnemo-infer"
        self.set_runtime("ok")
        if sys.platform == "win32":
            real_run = subprocess.run
            runtime_path = str(self.runtime.resolve())

            def run_script(command, *args, **kwargs):
                if command[0] == runtime_path:
                    command = [sys.executable, *command]
                return real_run(command, *args, **kwargs)

            launch = mock.patch.object(
                worker,
                "subprocess",
                SimpleNamespace(run=run_script, STDOUT=subprocess.STDOUT),
            )
            launch.start()
            self.addCleanup(launch.stop)
        self.output = self.root / "build"
        values = [[-2.0, 0.0, 1.0, 2.0], [0.25, 4.0, 10.0, -4.0]]
        self.prepared = {
            "model": object(),
            "cases": [(object(),), (object(),)],
            "inputs": [(tensor("input", v),) for v in values],
            "references": [(tensor("output", [2 * x + 1 for x in v]),) for v in values],
            "weights": {"kind": "embedded", "state_sha256": "0" * 64},
            "environment": {"torch_version": "fake"},
        }
        self.calls = []

    def set_runtime(self, mode):
        self.runtime.write_text(RUNTIME.format(python=sys.executable, mode=mode))
        self.runtime.chmod(0o755)

    def backend(self, backend, prepared, recipe, device, package, exported):
        self.calls.append(backend)
        package.mkdir(parents=True)
        exported.mkdir(parents=True)
        artifact = "model.pt2" if backend == "aoti" else "model.plan"
        (package / artifact).write_bytes(b"compiled " + backend.encode())
        (package / "model.json").write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "inputs": [{"name": "input", "dtype": "float32", "shape": [4]}],
                    "outputs": [{"name": "output", "dtype": "float32", "shape": [4]}],
                    "artifacts": [{"backend": backend, "path": artifact}],
                }
            )
        )
        (exported / ("program.pt2" if backend == "aoti" else "model.onnx")).write_bytes(
            b"raw graph"
        )
        if backend == "tensorrt":
            (exported / "model.onnx.data").write_bytes(b"external weights")
        return {"format": "exported_program" if backend == "aoti" else "onnx"}

    def run_build(self, backends=None):
        with (
            mock.patch.object(worker, "_prepare_model", return_value=self.prepared),
            mock.patch.object(worker, "_build_backend", side_effect=self.backend),
        ):
            return worker.execute_build(
                self.recipe_path,
                self.output,
                backends or ["aoti", "tensorrt"],
                "cpu",
                self.runtime,
            )

    def test_build_runs_native_for_every_case_and_requested_backend(self):
        report = self.run_build()
        self.assertEqual(report["status"], "complete")
        self.assertEqual(self.calls, ["aoti", "tensorrt"])
        self.assertEqual(set(report["variants"]), {"aoti", "tensorrt"})
        for backend in self.calls:
            check = json.loads((self.output / "checks" / f"{backend}.json").read_text())
            self.assertTrue(check["passed"])
            self.assertEqual(len(check["cases"]), 2)
            self.assertEqual(
                [c["outputs"][0]["max_abs"] for c in check["cases"]], [0.0, 0.0]
            )
            self.assertEqual(
                check["runtime"]["sha256"],
                hashlib.sha256(self.runtime.read_bytes()).hexdigest(),
            )
            self.assertTrue(
                (self.output / "model" / "backends" / backend / "model.json").is_file()
            )
        self.assertTrue((self.output / "model" / "model-release.json").is_file())
        self.assertEqual(
            report["recipe"]["sha256"],
            hashlib.sha256(self.recipe_path.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            report["source"]["adapter"]["sha256"],
            hashlib.sha256((self.root / "export.py").read_bytes()).hexdigest(),
        )
        self.assertIn(
            "exported/tensorrt/model.onnx.data",
            [f["path"] for f in report["variants"]["tensorrt"]["graphs"]],
        )

    def test_build_logs_preserve_unicode_with_legacy_locale(self):
        real_open = Path.open
        backend = self.backend
        stdout_message = "export \u2705 \u6a21\u578b"
        stderr_message = "diagnostic \u26a0 \u6a21\u578b"

        def legacy_open(
            path, mode="r", buffering=-1, encoding=None, errors=None, newline=None
        ):
            # Simulate the Windows ANSI default on every test host.
            if "b" not in mode and encoding is None:
                encoding = "cp1252"
            return real_open(path, mode, buffering, encoding, errors, newline)

        def export_with_progress(*args):
            print(stdout_message)
            print(stderr_message, file=sys.stderr)
            return backend(*args)

        with (
            mock.patch.object(Path, "open", legacy_open),
            mock.patch.object(self, "backend", side_effect=export_with_progress),
        ):
            report = self.run_build()

        self.assertEqual(report["status"], "complete", report)
        for name in ("aoti", "tensorrt"):
            content = (self.output / "logs" / f"{name}-build.log").read_bytes()
            self.assertIn(stdout_message.encode("utf-8"), content)
            self.assertIn(stderr_message.encode("utf-8"), content)

    def test_backend_packages_are_direct_siblings_and_independently_portable(self):
        report = self.run_build()
        model = self.output / "model"
        release = json.loads((model / "model-release.json").read_text())
        self.assertEqual(report["format_version"], 1)
        self.assertEqual(release["format_version"], 1)
        self.assertEqual(set(release["variants"]), {"aoti", "tensorrt"})
        inventory_paths = {}
        for backend in ("aoti", "tensorrt"):
            with self.subTest(backend=backend):
                expected = f"backends/{backend}"
                self.assertEqual(report["variants"][backend]["package"], expected)
                self.assertEqual(release["variants"][backend]["package"], expected)
                self.assertEqual(
                    report["variants"][backend]["files"],
                    release["variants"][backend]["files"],
                )
                inventory_paths[backend] = {
                    item["path"] for item in release["variants"][backend]["files"]
                }
                self.assertIn(f"{expected}/model.json", inventory_paths[backend])
                for path in inventory_paths[backend]:
                    self.assertTrue(Path(path).is_relative_to(expected), path)
                    self.assertNotIn("..", Path(path).parts)
        self.assertEqual(set(inventory_paths), {"aoti", "tensorrt"})
        self.assertTrue(inventory_paths["aoti"].isdisjoint(inventory_paths["tensorrt"]))
        self.assertEqual(
            {path.name for path in (model / "backends").iterdir()},
            {"aoti", "tensorrt"},
        )
        relocated = self.root / "relocated-bundle"
        shutil.copytree(model, relocated)
        standalone = self.root / "standalone"
        for backend, variant in release["variants"].items():
            shutil.copytree(model / variant["package"], standalone / backend)
        shutil.rmtree(self.output)
        for variant in release["variants"].values():
            for item in variant["files"]:
                data = (relocated / item["path"]).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), item["sha256"])
                self.assertEqual(len(data), item["size_bytes"])
        shutil.rmtree(relocated)
        for backend in ("aoti", "tensorrt"):
            package = standalone / backend
            manifest = json.loads((package / "model.json").read_text())
            self.assertEqual(manifest["format_version"], 1)
            self.assertEqual(len(manifest["artifacts"]), 1)
            artifact = manifest["artifacts"][0]
            self.assertEqual(artifact["backend"], backend)
            self.assertFalse(Path(artifact["path"]).is_absolute())
            self.assertNotIn("..", Path(artifact["path"]).parts)
            self.assertEqual(
                (package / artifact["path"]).read_bytes(),
                b"compiled " + backend.encode(),
            )

    def test_model_directory_is_a_portable_self_contained_package_bundle(self):
        self.run_build()
        copied = self.root / "relocated-model"
        shutil.copytree(self.output / "model", copied)
        release = json.loads((copied / "model-release.json").read_text())
        self.assertNotIn("path", release["runtime"])
        self.assertIn("external", release["runtime"]["dependency_scope"])
        for variant in release["variants"].values():
            package = copied / variant["package"]
            self.assertTrue((package / "model.json").is_file())
            for artifact in variant["files"]:
                path = copied / artifact["path"]
                self.assertTrue(path.is_file())
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(), artifact["sha256"]
                )
            self.assertNotIn("graphs", variant)
            self.assertNotIn("checks", variant)

    def test_retained_aoti_graph_is_the_exact_program_given_to_compiler(self):
        class Tensor:
            dtype = "float32"
            shape = (4,)

            def detach(self):
                return self

            def cpu(self):
                return self

            def contiguous(self):
                return self

            def to(self, *args):
                return self

        value = Tensor()

        class Model:
            def eval(self):
                return self

            def to(self, device):
                return self

            def __call__(self, *args):
                return value

        fake = ModuleType("torch")
        for dtype in ["float32", "float16", "bfloat16", "int32", "int64", "uint8"]:
            setattr(fake, dtype, dtype)
        fake.Tensor = Tensor
        fake.device = lambda target: SimpleNamespace(type="cpu")
        fake.inference_mode = nullcontext
        fake.get_float32_matmul_precision = lambda: "highest"
        fake.__version__ = "1.0.0"
        exported = SimpleNamespace(graph=SimpleNamespace(nodes=[]))
        captured = []
        compiled = []
        compiled_paths = []

        def save(program, path):
            captured.append(program)
            Path(path).write_bytes(b"retained export")

        def compile_program(program, package_path):
            compiled.append(program)
            compiled_paths.append(Path(package_path))
            Path(package_path).write_bytes(b"compiled package")

        fake.export = SimpleNamespace(export=lambda *a, **kw: exported, save=save)
        fake._inductor = SimpleNamespace(aoti_compile_and_package=compile_program)
        fake.testing = SimpleNamespace(assert_close=lambda *a, **kw: None)
        source = Path(worker.__file__).parents[1] / "pnmir_export" / "exporter.py"
        spec = importlib.util.spec_from_file_location("_retained_export_test", source)
        module = importlib.util.module_from_spec(spec)
        graph = self.root / "exported" / "program.pt2"
        with mock.patch.dict(sys.modules, {"torch": fake}):
            spec.loader.exec_module(module)
            with mock.patch.object(
                module, "_run_isolated_aoti_package", return_value=(value,)
            ) as validate:
                module.export_package(
                    Model(),
                    (value,),
                    self.root / "stage.pnmir",
                    model_name="affine",
                    model_version="1",
                    input_names=("input",),
                    output_names=("output",),
                    exported_program_path=graph,
                )
        self.assertTrue(graph.is_file(), "raw exported program was discarded")
        self.assertEqual(captured, [exported])
        self.assertEqual(compiled, [exported])
        self.assertEqual(validate.call_args.args[0], compiled_paths[0])
        package = self.root / "stage.pnmir"
        manifest = json.loads((package / "model.json").read_text())
        self.assertEqual(manifest["artifacts"][0]["path"], "model.pt2")
        self.assertEqual(
            {path.name for path in package.iterdir()}, {"model.json", "model.pt2"}
        )
        self.assertEqual(manifest["producer"]["aoti_validation"], "isolated_process")
        self.assertNotEqual(
            graph.read_bytes(),
            (package / "model.pt2").read_bytes(),
        )

    def test_recipe_and_adapter_sources_are_retained_outside_deployable_model(self):
        original_recipe = self.recipe_path.read_bytes()
        original_adapter = (self.root / "export.py").read_bytes()
        report = self.run_build(["aoti"])
        self.recipe_path.unlink()
        (self.root / "export.py").unlink()
        self.assertEqual(
            (self.output / "source" / "recipe.json").read_bytes(), original_recipe
        )
        self.assertEqual(
            (self.output / "source" / "export.py").read_bytes(), original_adapter
        )
        sources = report["source"]["files"]
        self.assertEqual(
            {item["path"] for item in sources},
            {"source/recipe.json", "source/export.py"},
        )
        for item in sources:
            self.assertEqual(
                hashlib.sha256((self.output / item["path"]).read_bytes()).hexdigest(),
                item["sha256"],
            )
        self.assertFalse(list((self.output / "model").rglob("*.py")))

    def test_only_selected_backend_is_built(self):
        report = self.run_build(["aoti"])
        self.assertEqual(self.calls, ["aoti"])
        self.assertEqual(set(report["variants"]), {"aoti"})
        self.assertFalse((self.output / "exported" / "tensorrt").exists())

    def test_backend_failure_retains_first_variant_and_failed_receipt(self):
        original = self.backend

        def fail(backend, *args):
            if backend == "tensorrt":
                raise RuntimeError("compiler failed")
            return original(backend, *args)

        with (
            mock.patch.object(worker, "_prepare_model", return_value=self.prepared),
            mock.patch.object(worker, "_build_backend", side_effect=fail),
        ):
            with self.assertRaisesRegex(RuntimeError, "compiler failed"):
                worker.execute_build(
                    self.recipe_path,
                    self.output,
                    ["aoti", "tensorrt"],
                    "cpu",
                    self.runtime,
                )
        report = json.loads((self.output / "build.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["variants"]["aoti"]["status"], "complete")
        self.assertEqual(report["variants"]["tensorrt"]["status"], "failed")
        self.assertFalse((self.output / "model" / "model-release.json").exists())

    def test_native_failure_retains_failed_check_and_log(self):
        self.set_runtime("fail")
        with self.assertRaisesRegex(RuntimeError, "native.*7"):
            self.run_build(["aoti"])
        self.assertEqual(
            json.loads((self.output / "build.json").read_text())["status"], "failed"
        )
        check = json.loads((self.output / "checks" / "aoti.json").read_text())
        self.assertFalse(check["passed"])
        self.assertIn(
            "native intentionally failed",
            (self.output / "logs" / "aoti-case-0.log").read_text(),
        )

    def test_matching_bytes_do_not_override_wrong_native_metadata(self):
        for mode in ["shape", "backend", "device", "bytes"]:
            with self.subTest(mode=mode):
                self.output = self.root / f"build-{mode}"
                self.set_runtime(mode)
                with self.assertRaisesRegex(
                    (RuntimeError, ValueError), "metadata|shape|backend|device|byte"
                ):
                    self.run_build(["aoti"])
                self.assertEqual(
                    json.loads((self.output / "build.json").read_text())["status"],
                    "failed",
                )

    def test_nonfinite_native_outputs_fail(self):
        self.set_runtime("nan")
        with self.assertRaisesRegex((RuntimeError, ValueError), "finite|NaN"):
            self.run_build(["aoti"])

    def test_no_cases_cannot_pass(self):
        self.prepared["cases"] = []
        self.prepared["inputs"] = []
        self.prepared["references"] = []
        with self.assertRaisesRegex(ValueError, "case"):
            self.run_build(["aoti"])

    def test_existing_output_is_never_overwritten(self):
        self.output.mkdir()
        marker = self.output / "owned"
        marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            self.run_build()
        self.assertEqual(marker.read_text(), "keep")
        self.assertEqual(list(self.output.iterdir()), [marker])

    def test_invalid_backends_and_adapter_paths_fail_before_model_loading(self):
        for backends in [[], ["aoti", "aoti"], ["onnxruntime"]]:
            with (
                self.subTest(backends=backends),
                mock.patch.object(worker, "_prepare_model") as prepare,
            ):
                with self.assertRaisesRegex(ValueError, "backend"):
                    worker.execute_build(
                        self.recipe_path, self.output, backends, "cpu", self.runtime
                    )
                prepare.assert_not_called()
        self.recipe["adapter"] = "../outside.py"
        self.recipe_path.write_text(json.dumps(self.recipe))
        with mock.patch.object(worker, "_prepare_model") as prepare:
            with self.assertRaisesRegex(ValueError, "adapter"):
                worker.execute_build(
                    self.recipe_path, self.output, ["aoti"], "cpu", self.runtime
                )
            prepare.assert_not_called()

    def test_worker_module_import_has_no_torch_requirement(self):
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                'import sys; from pnmir_build import worker; assert "torch" not in sys.modules',
            ],
            cwd=Path(__file__).resolve().parents[1] / "src",
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
