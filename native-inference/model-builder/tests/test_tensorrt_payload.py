"""Package publication with explicit TensorRT/CUDA API doubles, not GPU execution."""

import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None,
    "requires the optional Torch dependency",
)
class TensorRTPayloadTests(unittest.TestCase):
    def setUp(self):
        from pnmir_export import tensorrt_builder

        self.module = tensorrt_builder
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source.onnx"
        self.source.write_bytes(b"test graph; not executable ONNX")
        self.output = self.root / "tensorrt"
        self.onnx = SimpleNamespace(
            __version__="test-onnx",
            load_model=mock.Mock(return_value=object()),
            checker=SimpleNamespace(check_model=mock.Mock()),
        )
        self.config = SimpleNamespace(
            set_memory_pool_limit=mock.Mock(), clear_flag=mock.Mock()
        )
        self.builder = SimpleNamespace(
            create_network=mock.Mock(return_value=object()),
            create_builder_config=mock.Mock(return_value=self.config),
            build_serialized_network=mock.Mock(return_value=b"test compiled plan"),
        )
        self.trt = SimpleNamespace(
            __version__="test-tensorrt",
            Logger=mock.Mock(WARNING=2),
            Builder=mock.Mock(return_value=self.builder),
            OnnxParser=mock.Mock(
                return_value=SimpleNamespace(
                    parse_from_file=mock.Mock(return_value=True)
                )
            ),
            init_libnvinfer_plugins=mock.Mock(return_value=True),
            NetworkDefinitionCreationFlag=SimpleNamespace(STRONGLY_TYPED=1),
            MemoryPoolType=SimpleNamespace(WORKSPACE="workspace"),
            BuilderFlag=SimpleNamespace(TF32="tf32"),
        )
        self.contract = (
            [{"name": "input", "dtype": "float32", "shape": [4]}],
            [{"name": "output", "dtype": "float32", "shape": [4]}],
        )

    def build(self, **kwargs):
        cuda = SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
            set_device=mock.Mock(),
            get_device_capability=lambda index: (9, 0),
            get_device_name=lambda index: "test GPU",
        )
        with (
            mock.patch.object(self.module.torch, "cuda", cuda),
            mock.patch.object(self.module, "_onnx_module", return_value=self.onnx),
            mock.patch.object(
                self.module, "_graph_contract", return_value=self.contract
            ),
            mock.patch.object(self.module, "_tensorrt_module", return_value=self.trt),
        ):
            result = self.module.build_tensorrt_package(
                self.source,
                self.output,
                model_name="affine",
                model_version="test",
                device=1,
                workspace_size=4096,
                **kwargs,
            )
        cuda.set_device.assert_called_once_with(1)
        return result

    def test_plan_is_published_at_package_root_and_remains_portable(self):
        self.assertEqual(self.build(), self.output.resolve())
        manifest = json.loads((self.output / "model.json").read_text())
        self.assertEqual(manifest["format_version"], 1)
        self.assertEqual(manifest["artifacts"][0]["path"], "model.plan")
        self.assertEqual(
            {path.name for path in self.output.iterdir()}, {"model.json", "model.plan"}
        )
        self.assertEqual(manifest["artifacts"][0]["precision"], "fp32")
        self.assertEqual(manifest["artifacts"][0]["target"], "cuda")
        self.assertEqual(manifest["artifacts"][0]["runtime_version"], "test-tensorrt")
        self.assertEqual(manifest["producer"]["compute_capability"], "9.0")
        self.assertFalse(manifest["producer"]["tf32"])
        self.assertEqual(manifest["inputs"], self.contract[0])
        self.assertEqual(manifest["outputs"], self.contract[1])
        self.builder.create_network.assert_called_once_with(2)
        self.config.clear_flag.assert_called_once_with("tf32")
        self.config.set_memory_pool_limit.assert_called_once_with("workspace", 4096)
        relocated = self.root / "relocated"
        shutil.copytree(self.output, relocated)
        shutil.rmtree(self.output)
        self.source.unlink()
        self.assertEqual(
            (relocated / manifest["artifacts"][0]["path"]).read_bytes(),
            b"test compiled plan",
        )

    def test_failed_engine_build_preserves_existing_package_and_removes_staging(self):
        self.output.mkdir()
        (self.output / "model.json").write_text('{"existing":true}')
        (self.output / "model.plan").write_bytes(b"existing plan")
        before = {path.name: path.read_bytes() for path in self.output.iterdir()}
        self.builder.build_serialized_network.return_value = None
        with self.assertRaisesRegex(RuntimeError, "engine build failed"):
            self.build(force=True)
        self.assertEqual(
            {path.name: path.read_bytes() for path in self.output.iterdir()}, before
        )
        self.assertEqual(
            {path.name for path in self.root.iterdir()}, {"source.onnx", "tensorrt"}
        )

    def test_exact_profile_requires_all_plugin_libraries(self):
        # Exercise the old default when the selector is absent, establishing
        # that the baseline currently succeeds without the required libraries.
        parameters = inspect.signature(self.module.build_tensorrt_package).parameters
        options = {"profile": "layout-order-exact"} if "profile" in parameters else {}
        with self.assertRaisesRegex(ValueError, "requires.*plugin"):
            self.build(**options)

    def test_baseline_rejects_silently_unused_plugin_libraries(self):
        parameters = inspect.signature(self.module.build_tensorrt_package).parameters
        options = {"plugin_libraries": {"exact_linear": "unused.so"}} if "plugin_libraries" in parameters else {}
        with self.assertRaisesRegex(ValueError, "baseline.*plugin"):
            self.build(**options)


if __name__ == "__main__":
    unittest.main()
