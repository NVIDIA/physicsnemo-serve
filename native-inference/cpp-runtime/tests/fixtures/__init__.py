"""Frozen packages and tiny model factories used by SDK regression tests."""

from pathlib import Path


def write_affine_onnx(path: Path, *, graph_name: str) -> None:
    import onnx
    from onnx import TensorProto, helper

    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [3])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [3])
    scale = helper.make_tensor("scale", TensorProto.FLOAT, [1], [2.0])
    bias = helper.make_tensor("bias", TensorProto.FLOAT, [1], [1.0])
    graph = helper.make_graph(
        [
            helper.make_node("Mul", ["input", "scale"], ["scaled"]),
            helper.make_node("Add", ["scaled", "bias"], ["output"]),
        ],
        graph_name,
        [input_info],
        [output_info],
        [scale, bias],
    )
    model = helper.make_model(
        graph,
        producer_name="pnm-ir-test",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)
