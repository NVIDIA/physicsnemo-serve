"""Worker builds, backend doubles and receipt publication without ML dependencies."""

import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from unittest import mock

from model_builder.build import worker

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


def backend_modules():
    """Neutral module doubles for observing worker export calls."""
    modules = {"torch": ModuleType("torch")}
    for module_name, symbol in (
        ("exporter", "export_package"),
        ("onnx_exporter", "export_onnx_model"),
        ("tensorrt_builder", "build_tensorrt_package"),
        ("tensorrt_exporter", "tensorrt_ieee_fp32"),
    ):
        name = f"model_builder.export.{module_name}"
        modules[name] = ModuleType(name)
        setattr(modules[name], symbol, mock.Mock())
    return modules


def publish_build_receipts(output, build, release, backend, *, check_path=None):
    """Rehash receipt envelopes after explicit edits; preserve evidence contents."""
    variant = build["variants"][backend]
    variant["files"] = worker._inventory(
        output / "model" / variant["package"], output / "model"
    )
    release["variants"][backend]["files"] = variant["files"]
    if check_path is not None:
        variant["checks"] = worker._file_identity(check_path, output)
    release_path = output / "model/model-release.json"
    release_path.write_text(json.dumps(release))
    build["release"] = worker._file_identity(release_path, output)
    (output / "build.json").write_text(json.dumps(build))


class WorkerFixture:
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
