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
                        p.relative_to(package).as_posix()
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

    def subgraph_source(self, directory):
        onnx = self.onnx
        directory.mkdir()
        value_info = onnx.helper.make_tensor_value_info
        bias = onnx.numpy_helper.from_array(
            self.np.array([1, 2, 3], dtype="float32"), "branch-bias.bin"
        )
        offset = onnx.numpy_helper.from_array(
            self.np.array([10, 20, 30], dtype="float32"), "branch-offset.bin"
        )
        branch = onnx.helper.make_graph(
            [
                onnx.helper.make_node("Constant", [], ["offset"], value=offset),
                onnx.helper.make_node("Add", ["input", bias.name], ["biased"]),
                onnx.helper.make_node("Add", ["biased", "offset"], ["result"]),
            ],
            "branch",
            [],
            [value_info("result", onnx.TensorProto.FLOAT, [3])],
            [bias],
        )
        graph = onnx.helper.make_graph(
            [
                onnx.helper.make_node(
                    "If",
                    ["condition"],
                    ["output"],
                    then_branch=branch,
                    else_branch=branch,
                )
            ],
            "external-branch",
            [value_info("input", onnx.TensorProto.FLOAT, [3])],
            [value_info("output", onnx.TensorProto.FLOAT, [3])],
            [onnx.helper.make_tensor("condition", onnx.TensorProto.BOOL, [], [True])],
        )
        source = directory / "source.onnx"
        onnx.save_model(
            onnx.helper.make_model(
                graph, opset_imports=[onnx.helper.make_opsetid("", 18)], ir_version=8
            ),
            source,
            save_as_external_data=True,
            all_tensors_to_one_file=False,
            size_threshold=0,
            convert_attribute=True,
        )
        return source

    def test_subgraph_initializers_and_tensor_attributes_survive_relocation(self):
        from onnx.reference import ReferenceEvaluator

        source = self.subgraph_source(self.root / "subgraph-source")
        source_bytes = source.read_bytes()
        weights = {
            name: (source.parent / name).read_bytes()
            for name in ("branch-bias.bin", "branch-offset.bin")
        }
        package = self.import_package(
            source, self.root / "subgraph-package", model_name="branch", model_version="1"
        )
        moved = self.root / "subgraph-deployed"
        package.rename(moved)
        shutil.rmtree(source.parent)
        self.assertEqual((moved / "model.onnx").read_bytes(), source_bytes)
        self.assertTrue((moved / "branch-bias.bin").is_file())
        self.assertTrue((moved / "branch-offset.bin").is_file())
        for name, contents in weights.items():
            self.assertEqual((moved / name).read_bytes(), contents)
        loaded = self.onnx.load_model(moved / "model.onnx")
        actual = ReferenceEvaluator(loaded).run(
            None, {"input": self.np.array([4, 5, 6], dtype="float32")}
        )
        self.np.testing.assert_array_equal(actual[0], [15, 27, 39])

    def test_external_data_in_sparse_tensors_and_attribute_lists(self):
        from pnmir_export.onnx_importer import _external_data_locations

        onnx = self.onnx
        locations = []

        def external_tensor(name, dtype="float32"):
            values = self.np.array([1], dtype=dtype)
            tensor = onnx.numpy_helper.from_array(values, name)
            onnx.external_data_helper.set_external_data(tensor, name)
            tensor.ClearField("raw_data")
            (self.root / name).write_bytes(values.tobytes())
            locations.append(name)
            return tensor

        def sparse_tensor(name):
            return onnx.helper.make_sparse_tensor(
                external_tensor(f"{name}-values.bin"),
                external_tensor(f"{name}-indices.bin", "int64"),
                [3],
            )

        nested_graph = onnx.helper.make_graph(
            [], "nested", [], [], [external_tensor("graph-list.bin")]
        )
        node = onnx.helper.make_node(
            "Custom",
            [],
            [],
            branches=[nested_graph],
            tensors=[external_tensor("tensor-list.bin")],
            sparse=sparse_tensor("attribute"),
            sparse_list=[sparse_tensor("attribute-list")],
        )
        graph = onnx.helper.make_graph(
            [node], "graph", [], [], sparse_initializer=[sparse_tensor("initializer")]
        )
        actual = _external_data_locations(
            onnx.helper.make_model(graph), self.root / "source.onnx"
        )
        self.assertEqual(
            actual,
            [(self.root.resolve() / name, Path(name)) for name in sorted(locations)],
        )

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
