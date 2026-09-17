"""Characterize the qualified GeoTransolver weighted-blend ONNX substitution."""

import unittest

try:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
except ImportError as error:
    raise unittest.SkipTest("weighted blend tests require NumPy and ONNX") from error

from pnmir_export.tensorrt_exact_graphs import _replace_weighted_blends


class WeightedBlendGraphTests(unittest.TestCase):
    def model(self):
        shape = (1, 8, 128, 56)
        return helper.make_model(helper.make_graph(
            [
                helper.make_node("Mul", ("left_weight", "left"), ("scaled_left",)),
                helper.make_node("Mul", ("right_weight", "right"), ("scaled_right",)),
                helper.make_node("Add", ("scaled_left", "scaled_right"), ("output",)),
            ], "weighted-blend",
            [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name in ("left", "right")],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
            [numpy_helper.from_array(np.array(value, dtype=np.float32), name=name) for name, value in (("left_weight", 0.492), ("right_weight", 0.508))],
        ))

    def test_scalar_fp32_blend_becomes_one_four_input_operator(self):
        model = self.model()
        self.assertEqual(_replace_weighted_blends(onnx, model), 1)
        self.assertEqual(len(model.graph.node), 1)
        plugin = model.graph.node[0]
        self.assertEqual(plugin.op_type, "PNMIRExactWeightedBlend")
        self.assertEqual(list(plugin.input), ["left", "left_weight", "right", "right_weight"])
        self.assertEqual(list(plugin.output), ["output"])

    def test_broadcasts_non_fp32_and_shared_intermediates_remain_unmodified(self):
        for unsupported in ("shape", "dtype", "scalar-rank", "shared", "graph-output"):
            with self.subTest(unsupported=unsupported):
                model = self.model()
                if unsupported == "shape":
                    model.graph.input[1].type.tensor_type.shape.dim[-1].dim_value = 1
                elif unsupported == "dtype":
                    model.graph.input[1].type.tensor_type.elem_type = TensorProto.FLOAT16
                elif unsupported == "scalar-rank":
                    model.graph.initializer[0].CopyFrom(numpy_helper.from_array(np.array([0.492], dtype=np.float32), name="left_weight"))
                elif unsupported == "shared":
                    model.graph.node.append(helper.make_node("Identity", ["scaled_left"], ["other"]))
                else:
                    model.graph.output.append(helper.make_tensor_value_info("scaled_left", TensorProto.FLOAT, [1, 8, 128, 56]))
                before = model.SerializeToString()
                self.assertEqual(_replace_weighted_blends(onnx, model), 0)
                self.assertEqual(model.SerializeToString(), before)


if __name__ == "__main__":
    unittest.main()
