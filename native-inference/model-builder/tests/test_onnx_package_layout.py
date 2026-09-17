"""Real ONNX import, external-data relocation, and atomic collision rejection."""

import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@unittest.skipUnless(
    importlib.util.find_spec("onnx") and importlib.util.find_spec("torch"),
    "requires optional ONNX and Torch dependencies",
)
class OnnxPackageLayoutTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import onnx
        from pnmir_export.onnx_importer import import_onnx_package

        self.np = np
        self.onnx = onnx
        self.import_package = import_onnx_package
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def source(self, directory, location=None):
        onnx = self.onnx
        directory.mkdir()
        source = directory / "source.onnx"
        tensor = onnx.helper.make_tensor_value_info
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node("Add", ["input", "bias"], ["output"])],
            "external-bias",
            [tensor("input", onnx.TensorProto.FLOAT, [3])],
            [tensor("output", onnx.TensorProto.FLOAT, [3])],
            [
                onnx.numpy_helper.from_array(
                    self.np.array([1, 2, 3], dtype="float32"), "bias"
                )
            ],
        )
        model = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 18)], ir_version=8
        )
        options = {}
        if location:
            (directory / location).parent.mkdir(parents=True, exist_ok=True)
            options = dict(
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=location,
                size_threshold=0,
            )
        onnx.save_model(model, source, **options)
        return source

    def test_flat_graph_and_external_weights_survive_relocation(self):
        from onnx.reference import ReferenceEvaluator

        for index, location in enumerate((None, "weights.bin", "weights/shard.bin")):
            with self.subTest(location=location):
                source = self.source(self.root / f"source-{index}", location)
                source_bytes = source.read_bytes()
                weights = (source.parent / location).read_bytes() if location else None
                package = self.import_package(
                    source,
                    self.root / f"package-{index}",
                    model_name="bias",
                    model_version="1",
                )
                manifest = json.loads((package / "model.json").read_text())
                self.assertEqual(manifest["artifacts"][0]["path"], "model.onnx")
                expected = {"model.json", "model.onnx"}
                if location:
                    expected.add(location)
                self.assertEqual(
                    {
                        str(p.relative_to(package))
                        for p in package.rglob("*")
                        if p.is_file()
                    },
                    expected,
                )
                moved = self.root / f"deployed-{index}"
                package.rename(moved)
                shutil.rmtree(source.parent)
                self.assertEqual((moved / "model.onnx").read_bytes(), source_bytes)
                if location:
                    self.assertEqual((moved / location).read_bytes(), weights)
                loaded = self.onnx.load_model(moved / manifest["artifacts"][0]["path"])
                actual = ReferenceEvaluator(loaded).run(
                    None, {"input": self.np.array([4, 5, 6], dtype="float32")}
                )
                self.np.testing.assert_array_equal(actual[0], [5, 7, 9])

    def test_reserved_external_paths_cannot_replace_existing_package(self):
        package = self.root / "existing"
        package.mkdir()
        (package / "model.json").write_text('{"existing": true}\n')
        (package / "payload.bin").write_bytes(b"previous qualified payload")
        original = {p.name: p.read_bytes() for p in package.iterdir()}
        for index, location in enumerate(
            (
                "model.json",
                "model.onnx",
                "model.json/shard.bin",
                "model.onnx/shard.bin",
                "MODEL.JSON",
                "MODEL.ONNX/shard.bin",
            )
        ):
            with self.subTest(location=location):
                source = self.source(self.root / f"collision-{index}", location)
                with self.assertRaisesRegex(ValueError, "conflicts with package file"):
                    self.import_package(
                        source,
                        package,
                        model_name="bias",
                        model_version="1",
                        force=True,
                    )
                self.assertEqual(
                    {p.name: p.read_bytes() for p in package.iterdir()}, original
                )
                self.assertEqual(list(self.root.glob(".existing-*")), [])


if __name__ == "__main__":
    unittest.main()
