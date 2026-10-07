"""Package publication with explicit TensorRT/CUDA API doubles, not GPU execution."""

import importlib.util
import json
from pathlib import Path
import shutil
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tensorrt_test_support import TensorRTPayloadFixture


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None,
    "requires the optional Torch dependency",
)
class TensorRTPayloadTests(TensorRTPayloadFixture, unittest.TestCase):
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
        with self.assertRaisesRegex(ValueError, "requires.*plugin"):
            self.build(profile="layout-order-exact")

    def test_baseline_rejects_silently_unused_plugin_libraries(self):
        with self.assertRaisesRegex(ValueError, "baseline.*plugin"):
            self.build(plugin_libraries={"exact_linear": "unused.so"})


if __name__ == "__main__":
    unittest.main()
