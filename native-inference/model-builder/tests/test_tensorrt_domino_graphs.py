"""Characterize the qualified DoMINO scalar-division and blend graph ports."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import numpy as np
    import onnx
    import pytest
    from onnx import TensorProto, helper, numpy_helper
except ImportError as error:
    raise unittest.SkipTest(
        "DoMINO graph tests require optional NumPy, ONNX and pytest"
    ) from error
from model_builder.export.tensorrt_exact_graphs import (
    _replace_scalar_divs,
    _replace_inverse_distance_blends,
)
from graph_test_support import _check_model_with_tensorrt_plugins


def _scalar_div_model(
    *,
    scalar: np.ndarray | None = None,
    shape: tuple[int, ...] = (2, 512),
) -> onnx.ModelProto:
    if scalar is None:
        scalar = np.array(10.0, dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("Div", ("input", "scalar"), ("output",))],
        "exact-scalar-div",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
        [numpy_helper.from_array(scalar, name="scalar")],
    )
    return helper.make_model(graph)


def test_replaces_fp32_constant_scalar_division() -> None:
    model = _scalar_div_model()

    assert _replace_scalar_divs(onnx, model) == 1

    assert model.graph.node[0].op_type == "PNMIRExactScalarDiv"
    assert list(model.graph.node[0].input) == ["input", "scalar"]
    assert list(model.graph.node[0].output) == ["output"]


def _inverse_distance_blend_model(
    *, variables: int = 1, samples: int = 3
) -> onnx.ModelProto:
    shape = (1, 8, 1)
    nodes = []
    inputs = []
    outputs = []
    value_info = []
    half = numpy_helper.from_array(np.array(0.5, dtype=np.float32), name="half")
    distances = []
    reciprocals = []
    inverse_sum = None
    for sample in range(samples):
        distance = f"distance_{sample}"
        reciprocal = f"inverse_{sample}"
        distances.append(distance)
        reciprocals.append(reciprocal)
        inputs.append(helper.make_tensor_value_info(distance, TensorProto.FLOAT, shape))
        nodes.append(helper.make_node("Reciprocal", (distance,), (reciprocal,)))
        value_info.append(
            helper.make_tensor_value_info(reciprocal, TensorProto.FLOAT, shape)
        )
        if inverse_sum is None:
            inverse_sum = reciprocal
        else:
            next_sum = f"inverse_sum_{sample}"
            nodes.append(
                helper.make_node("Add", (inverse_sum, reciprocal), (next_sum,))
            )
            value_info.append(
                helper.make_tensor_value_info(next_sum, TensorProto.FLOAT, shape)
            )
            inverse_sum = next_sum

    for variable in range(variables):
        center = f"center_{variable}"
        inputs.append(helper.make_tensor_value_info(center, TensorProto.FLOAT, shape))
        neighbor_sum = None
        for sample, reciprocal in enumerate(reciprocals):
            prediction = f"prediction_{variable}_{sample}"
            weighted = f"weighted_{variable}_{sample}"
            inputs.append(
                helper.make_tensor_value_info(prediction, TensorProto.FLOAT, shape)
            )
            nodes.append(helper.make_node("Mul", (prediction, reciprocal), (weighted,)))
            value_info.append(
                helper.make_tensor_value_info(weighted, TensorProto.FLOAT, shape)
            )
            if neighbor_sum is None:
                neighbor_sum = weighted
            else:
                next_sum = f"neighbor_sum_{variable}_{sample}"
                nodes.append(
                    helper.make_node("Add", (neighbor_sum, weighted), (next_sum,))
                )
                value_info.append(
                    helper.make_tensor_value_info(next_sum, TensorProto.FLOAT, shape)
                )
                neighbor_sum = next_sum
        center_half = f"center_half_{variable}"
        neighbor_half = f"neighbor_half_{variable}"
        normalized = f"normalized_{variable}"
        output = f"output_{variable}"
        nodes.extend(
            (
                helper.make_node("Mul", (center, "half"), (center_half,)),
                helper.make_node("Mul", (neighbor_sum, "half"), (neighbor_half,)),
                helper.make_node("Div", (neighbor_half, inverse_sum), (normalized,)),
                helper.make_node("Add", (center_half, normalized), (output,)),
            )
        )
        value_info.extend(
            helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
            for name in (center_half, neighbor_half, normalized)
        )
        outputs.append(helper.make_tensor_value_info(output, TensorProto.FLOAT, shape))
    return helper.make_model(
        helper.make_graph(
            nodes,
            "inverse-distance-blend",
            inputs,
            outputs,
            [half],
            value_info=value_info,
        )
    )


@pytest.mark.parametrize("variables", (1, 2))
def test_fuses_inverse_distance_blend_with_shared_distances(variables: int) -> None:
    model = _inverse_distance_blend_model(variables=variables)
    onnx.checker.check_model(model, full_check=True)

    assert _replace_inverse_distance_blends(onnx, model) == variables

    plugins = [
        node
        for node in model.graph.node
        if node.op_type == "PNMIRExactInverseDistanceBlend"
    ]
    assert len(plugins) == variables
    assert list(plugins[0].input) == [
        "center_0",
        "prediction_0_0",
        "distance_0",
        "prediction_0_1",
        "distance_1",
        "prediction_0_2",
        "distance_2",
    ]
    assert all(node.op_type != "Reciprocal" for node in model.graph.node)
    _check_model_with_tensorrt_plugins(model)


def test_keeps_externally_consumed_inverse_distance_intermediate() -> None:
    model = _inverse_distance_blend_model()
    model.graph.output.append(
        helper.make_tensor_value_info("inverse_0", TensorProto.FLOAT, (1, 8, 1))
    )
    onnx.checker.check_model(model, full_check=True)

    assert _replace_inverse_distance_blends(onnx, model) == 0

    assert all(
        node.op_type != "PNMIRExactInverseDistanceBlend" for node in model.graph.node
    )


@pytest.mark.parametrize("variables", (1, 2))
@pytest.mark.parametrize("consumer", ("center_half_0", "weighted_0_1", "inverse_1"))
def test_keeps_inverse_distance_producer_needed_by_plugin_input(
    variables: int, consumer: str
) -> None:
    model = _inverse_distance_blend_model(variables=variables, samples=2)
    for node in model.graph.node:
        if list(node.output) == [consumer]:
            node.input[0] = "inverse_0"
    onnx.checker.check_model(model, full_check=True)
    original = model.SerializeToString()

    replacements = _replace_inverse_distance_blends(onnx, model)

    _check_model_with_tensorrt_plugins(model)
    assert replacements == 0
    assert model.SerializeToString() == original


@pytest.mark.parametrize(
    "scalar",
    (
        np.array([10.0], dtype=np.float32),
        np.array(10.0, dtype=np.float64),
        np.array(2.0, dtype=np.float32),
    ),
)
def test_keeps_non_fp32_or_nonscalar_division_native(scalar: np.ndarray) -> None:
    model = _scalar_div_model(scalar=scalar)

    assert _replace_scalar_divs(onnx, model) == 0
    assert model.graph.node[0].op_type == "Div"
