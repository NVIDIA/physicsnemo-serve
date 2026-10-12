"""Characterization tests ported with the bounded exact TensorRT graph rewrites."""

from __future__ import annotations

import unittest

try:
    import numpy as np
    import onnx
    import pytest
    from onnx import TensorProto, helper, numpy_helper
except ImportError as error:
    raise unittest.SkipTest(
        "exact graph characterization requires optional NumPy, ONNX and pytest"
    ) from error
from graph_test_support import (
    _with_rematerialized_nodes,
    _reserve_value_name,
    _check_model_with_tensorrt_plugins,
)
from model_builder.export.tensorrt_exact_graphs import (
    _replace_attention_subgraphs,
    _replace_constant_rhs_matmuls,
    _replace_gelu_subgraphs,
    _replace_layer_norms,
    _replace_linear_subgraphs,
    _replace_slice_bmms,
    _replace_softmaxes,
    _replace_token_sums,
)

_ATTENTION_INPUT_SCALE = np.float32(32.0**-0.25)


_GELU_SQRT_TWO = np.float32(np.sqrt(2.0))


_GELU_ONE = np.float32(1.0)


_GELU_HALF = np.float32(0.5)


_VALUE_NAMESPACES = ("input", "output", "value_info", "initializer", "node_output")


def _with_non_fp32_cast_boundary(
    model: onnx.ModelProto,
    *,
    input_names: tuple[str, ...],
    output_name: str,
    initializer_names: tuple[str, ...],
    dtype: int,
    numpy_dtype: type[np.generic],
) -> onnx.ModelProto:
    typed_inputs = {name: f"{name}__typed" for name in input_names}
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name in typed_inputs:
                node.input[index] = typed_inputs[name]

    cast_inputs = []
    graph_inputs = {value.name: value for value in model.graph.input}
    for name, typed_name in typed_inputs.items():
        cast_inputs.append(helper.make_node("Cast", (name,), (typed_name,), to=dtype))
        typed_value = onnx.ValueInfoProto()
        typed_value.CopyFrom(graph_inputs[name])
        typed_value.name = typed_name
        typed_value.type.tensor_type.elem_type = dtype
        model.graph.value_info.append(typed_value)

    typed_output_name = f"{output_name}__typed"
    for node in model.graph.node:
        for index, name in enumerate(node.output):
            if name == output_name:
                node.output[index] = typed_output_name
    graph_outputs = {value.name: value for value in model.graph.output}
    typed_output = onnx.ValueInfoProto()
    typed_output.CopyFrom(graph_outputs[output_name])
    typed_output.name = typed_output_name
    typed_output.type.tensor_type.elem_type = dtype
    model.graph.value_info.append(typed_output)

    for value in model.graph.value_info:
        if value.name not in {typed_output_name, *typed_inputs.values()}:
            value.type.tensor_type.elem_type = dtype
    initializers = {
        initializer.name: initializer for initializer in model.graph.initializer
    }
    for name in initializer_names:
        initializer = initializers[name]
        value = numpy_helper.to_array(initializer).astype(numpy_dtype)
        initializer.CopyFrom(numpy_helper.from_array(value, name=name))

    for node in reversed(cast_inputs):
        model.graph.node.insert(0, node)
    model.graph.node.append(
        helper.make_node(
            "Cast", (typed_output_name,), (output_name,), to=TensorProto.FLOAT
        )
    )
    return model


def _shape_only_value_info(name: str, shape: tuple[int, ...]) -> onnx.ValueInfoProto:
    value = onnx.ValueInfoProto()
    value.name = name
    for dimension in shape:
        value.type.tensor_type.shape.dim.add().dim_value = dimension
    return value


def _append_subgraph_capture(
    model: onnx.ModelProto,
    name: str,
    shape: tuple[int, ...],
    *,
    nested_loop: bool,
) -> None:
    branch = helper.make_graph(
        [helper.make_node("Identity", (name,), ("branch_result",))],
        "capture-branch",
        [],
        [helper.make_tensor_value_info("branch_result", TensorProto.FLOAT, shape)],
    )
    model.graph.input.append(
        helper.make_tensor_value_info("condition", TensorProto.BOOL, ())
    )
    if nested_loop:
        body = helper.make_graph(
            [
                helper.make_node("Identity", ("loop_condition",), ("next_condition",)),
                helper.make_node(
                    "If",
                    ("loop_condition",),
                    ("scan_value",),
                    then_branch=branch,
                    else_branch=branch,
                ),
            ],
            "capture-loop-body",
            [
                helper.make_tensor_value_info("iteration", TensorProto.INT64, ()),
                helper.make_tensor_value_info("loop_condition", TensorProto.BOOL, ()),
            ],
            [
                helper.make_tensor_value_info("next_condition", TensorProto.BOOL, ()),
                helper.make_tensor_value_info("scan_value", TensorProto.FLOAT, shape),
            ],
        )
        model.graph.initializer.append(
            numpy_helper.from_array(np.array(1, dtype=np.int64), name="trip_count")
        )
        model.graph.node.append(
            helper.make_node(
                "Loop", ("trip_count", "condition"), ("retained",), body=body
            )
        )
        retained_shape = (None, *shape)
    else:
        model.graph.node.append(
            helper.make_node(
                "If",
                ("condition",),
                ("retained",),
                then_branch=branch,
                else_branch=branch,
            )
        )
        retained_shape = shape
    model.graph.output.append(
        helper.make_tensor_value_info("retained", TensorProto.FLOAT, retained_shape)
    )


def _linear_capture_model(*, transpose: bool = False) -> onnx.ModelProto:
    weight = np.arange(12, dtype=np.float32).reshape(3, 4)
    nodes = []
    if transpose:
        weight = weight.T.copy()
        nodes.append(
            helper.make_node("Transpose", ("weight",), ("weight_t",), perm=(1, 0))
        )
    nodes.extend(
        [
            helper.make_node(
                "MatMul",
                ("input", "weight_t" if transpose else "weight"),
                ("projected",),
            ),
            helper.make_node("Add", ("projected", "bias"), ("output",)),
        ]
    )
    return helper.make_model(
        helper.make_graph(
            nodes,
            "linear-with-subgraph-capture",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, (2, 3))],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, (2, 4))],
            [
                numpy_helper.from_array(weight, name="weight"),
                numpy_helper.from_array(np.arange(4, dtype=np.float32), name="bias"),
            ],
            value_info=[
                helper.make_tensor_value_info("projected", TensorProto.FLOAT, (2, 4))
            ],
        )
    )


@pytest.mark.parametrize("nested_loop", (False, True))
def test_linear_keeps_intermediate_captured_by_subgraph(nested_loop: bool) -> None:
    model = _linear_capture_model()
    _append_subgraph_capture(model, "projected", (2, 4), nested_loop=nested_loop)
    onnx.checker.check_model(model, full_check=True)

    assert _replace_linear_subgraphs(onnx, _with_rematerialized_nodes(model)) == 0

    onnx.checker.check_model(model, full_check=True)
    assert model.graph.node[0].op_type == "MatMul"


@pytest.mark.parametrize("nested_loop", (False, True))
@pytest.mark.parametrize(
    "rewrite", (_replace_linear_subgraphs, _replace_constant_rhs_matmuls)
)
def test_exact_projection_preserves_weight_captured_by_subgraph(
    rewrite, nested_loop: bool
) -> None:
    model = _linear_capture_model()
    weight = numpy_helper.to_array(model.graph.initializer[0]).copy()
    _append_subgraph_capture(model, "weight", (3, 4), nested_loop=nested_loop)
    onnx.checker.check_model(model, full_check=True)

    assert rewrite(onnx, model) == 1

    np.testing.assert_array_equal(
        numpy_helper.to_array(model.graph.initializer[0]), weight
    )
    assert model.graph.node[0].input[1] != "weight"
    _check_model_with_tensorrt_plugins(model)


@pytest.mark.parametrize("nested_loop", (False, True))
def test_linear_preserves_transpose_captured_by_subgraph(nested_loop: bool) -> None:
    model = _linear_capture_model(transpose=True)
    _append_subgraph_capture(model, "weight_t", (3, 4), nested_loop=nested_loop)
    onnx.checker.check_model(model, full_check=True)

    assert _replace_linear_subgraphs(onnx, _with_rematerialized_nodes(model)) == 1

    assert model.graph.node[0].op_type == "Transpose"
    _check_model_with_tensorrt_plugins(model)


@pytest.mark.parametrize("nested_loop", (False, True))
@pytest.mark.parametrize("intermediate", ("divided", "erf", "plus_one", "scaled"))
def test_gelu_keeps_intermediate_captured_by_subgraph(
    intermediate: str, nested_loop: bool
) -> None:
    model = _gelu_model()
    _append_subgraph_capture(model, intermediate, (2, 512), nested_loop=nested_loop)
    onnx.checker.check_model(model, full_check=True)

    assert _replace_gelu_subgraphs(onnx, _with_rematerialized_nodes(model)) == 0

    onnx.checker.check_model(model, full_check=True)
    assert model.graph.node[0].op_type == "Div"


def test_replaces_static_fp32_matmul_with_transposed_plugin_weight() -> None:
    weight = np.arange(12, dtype=np.float32).reshape(3, 4)
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ("input", "weight"), ("projected",)),
            helper.make_node("Identity", ("weight",), ("retained",)),
        ],
        "exact-gemm",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, (2, 3))],
        [
            helper.make_tensor_value_info("projected", TensorProto.FLOAT, (2, 4)),
            helper.make_tensor_value_info("retained", TensorProto.FLOAT, (3, 4)),
        ],
        [numpy_helper.from_array(weight, name="weight")],
    )
    model = helper.make_model(graph)

    assert _replace_constant_rhs_matmuls(onnx, model) == 1

    node = model.graph.node[0]
    assert node.op_type == "PNMIRExactGemm"
    assert {attribute.name for attribute in node.attribute} == {
        "plugin_namespace",
        "plugin_version",
    }
    assert node.input[1] != "weight"
    initializers = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    np.testing.assert_array_equal(initializers["weight"], weight)
    np.testing.assert_array_equal(initializers[node.input[1]], weight.T)


def test_static_matmul_preserves_rhs_exposed_as_graph_output() -> None:
    weight = np.arange(12, dtype=np.float32).reshape(3, 4)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ("input", "weight"), ("projected",))],
        "exact-gemm-with-exposed-weight",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, (2, 3))],
        [
            helper.make_tensor_value_info("projected", TensorProto.FLOAT, (2, 4)),
            helper.make_tensor_value_info("weight", TensorProto.FLOAT, (3, 4)),
        ],
        [numpy_helper.from_array(weight, name="weight")],
    )
    model = helper.make_model(graph)
    onnx.checker.check_model(model)

    assert _replace_constant_rhs_matmuls(onnx, model) == 1

    node = model.graph.node[0]
    assert node.op_type == "PNMIRExactGemm"
    assert node.input[1] != "weight"
    initializers = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    assert model.graph.output[1].name == "weight"
    assert [
        dimension.dim_value
        for dimension in model.graph.output[1].type.tensor_type.shape.dim
    ] == [3, 4]
    np.testing.assert_array_equal(initializers["weight"], weight)
    np.testing.assert_array_equal(initializers[node.input[1]], weight.T)


@pytest.mark.parametrize("namespace", _VALUE_NAMESPACES)
def test_static_gemm_allocates_unique_shared_weight_name(namespace: str) -> None:
    weight = np.arange(12, dtype=np.float32).reshape(3, 4)
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ("input", "weight"), ("projected",)),
            helper.make_node("Identity", ("weight",), ("retained",)),
        ],
        "collision-safe-exact-gemm",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, (2, 3))],
        [
            helper.make_tensor_value_info("projected", TensorProto.FLOAT, (2, 4)),
            helper.make_tensor_value_info("retained", TensorProto.FLOAT, (3, 4)),
        ],
        [numpy_helper.from_array(weight, name="weight")],
    )
    model = helper.make_model(graph)
    reserved = "weight__pnmir_exact_gemm_0"
    _reserve_value_name(model, reserved, namespace)
    onnx.checker.check_model(model, full_check=True)

    assert _replace_constant_rhs_matmuls(onnx, model) == 1

    plugin = next(node for node in model.graph.node if node.op_type == "PNMIRExactGemm")
    assert plugin.input[1] == f"{reserved}_1"
    _check_model_with_tensorrt_plugins(model)


def test_keeps_rank_one_static_gemm_native() -> None:
    graph = helper.make_graph(
        [helper.make_node("MatMul", ("input", "weight"), ("output",))],
        "rank-one-native-gemm",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, (3,))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, (4,))],
        [
            numpy_helper.from_array(
                np.arange(12, dtype=np.float32).reshape(3, 4), name="weight"
            )
        ],
    )
    model = helper.make_model(graph)
    onnx.checker.check_model(model, full_check=True)

    assert _replace_constant_rhs_matmuls(onnx, model) == 0

    onnx.checker.check_model(model, full_check=True)
    assert model.graph.node[0].op_type == "MatMul"


@pytest.mark.parametrize("unknown_rank", ("activation", "output"))
def test_keeps_static_gemm_with_unknown_plugin_rank_native(
    unknown_rank: str,
) -> None:
    nodes = []
    activation = "input"
    plugin_output = "output"
    if unknown_rank == "activation":
        activation = "activation"
        nodes.append(helper.make_node("Identity", ("input",), (activation,)))
    else:
        plugin_output = "projected"
    nodes.append(helper.make_node("MatMul", (activation, "weight"), (plugin_output,)))
    if unknown_rank == "output":
        nodes.append(helper.make_node("Identity", (plugin_output,), ("output",)))
    graph = helper.make_graph(
        nodes,
        "unknown-rank-native-gemm",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, (2, 3))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, (2, 4))],
        [
            numpy_helper.from_array(
                np.arange(12, dtype=np.float32).reshape(3, 4), name="weight"
            )
        ],
    )
    model = helper.make_model(graph)
    onnx.checker.check_model(model, full_check=True)

    assert _replace_constant_rhs_matmuls(onnx, model) == 0

    onnx.checker.check_model(model, full_check=True)
    assert any(node.op_type == "MatMul" for node in model.graph.node)


@pytest.mark.parametrize(
    ("dtype", "numpy_dtype"),
    (
        (TensorProto.FLOAT16, np.float16),
        (TensorProto.DOUBLE, np.float64),
    ),
)
def test_keeps_non_fp32_cast_boundary_static_gemm_native(
    dtype: int, numpy_dtype: type[np.generic]
) -> None:
    graph = helper.make_graph(
        [helper.make_node("MatMul", ("input", "weight"), ("output",))],
        "native-typed-gemm",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, (2, 3))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, (2, 4))],
        [
            numpy_helper.from_array(
                np.arange(12, dtype=np.float32).reshape(3, 4), name="weight"
            )
        ],
    )
    model = _with_non_fp32_cast_boundary(
        helper.make_model(graph),
        input_names=("input",),
        output_name="output",
        initializer_names=("weight",),
        dtype=dtype,
        numpy_dtype=numpy_dtype,
    )
    onnx.checker.check_model(model, full_check=True)

    assert _replace_constant_rhs_matmuls(onnx, model) == 0

    onnx.checker.check_model(model, full_check=True)
    assert [node.op_type for node in model.graph.node] == [
        "Cast",
        "MatMul",
        "Cast",
    ]


def test_fuses_static_fp32_linear_with_transposed_plugin_weight() -> None:
    weight = np.arange(12, dtype=np.float32).reshape(3, 4)
    bias = np.arange(4, dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ("input", "weight"), ("projected",)),
            helper.make_node("Add", ("projected", "bias"), ("output",)),
        ],
        "exact-linear",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, (2, 3))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, (2, 4))],
        [
            numpy_helper.from_array(weight, name="weight"),
            numpy_helper.from_array(bias, name="bias"),
        ],
    )
    model = helper.make_model(graph)

    assert _replace_linear_subgraphs(onnx, model) == 1

    assert len(model.graph.node) == 1
    node = model.graph.node[0]
    assert node.op_type == "PNMIRExactLinear"
    assert list(node.input) == ["input", "weight", "bias"]
    np.testing.assert_array_equal(
        numpy_helper.to_array(model.graph.initializer[0]), weight.T
    )


def test_fuses_scoped_linear_with_shared_weight_transpose() -> None:
    weight = np.arange(12, dtype=np.float32).reshape(4, 3)
    target_bias = np.arange(4, dtype=np.float32)
    other_bias = target_bias + np.float32(1.0)
    graph = helper.make_graph(
        [
            helper.make_node("Transpose", ("weight",), ("weight_t",), perm=(1, 0)),
            helper.make_node("MatMul", ("input", "weight_t"), ("target_projected",)),
            helper.make_node(
                "Add",
                ("target_projected", "target.layers.0.bias"),
                ("target_output",),
            ),
            helper.make_node("MatMul", ("input", "weight_t"), ("other_projected",)),
            helper.make_node(
                "Add", ("other_projected", "other.bias"), ("other_output",)
            ),
        ],
        "scoped-shared-weight-linear",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, (2, 3))],
        [
            helper.make_tensor_value_info("target_output", TensorProto.FLOAT, (2, 4)),
            helper.make_tensor_value_info("other_output", TensorProto.FLOAT, (2, 4)),
        ],
        [
            numpy_helper.from_array(weight, name="weight"),
            numpy_helper.from_array(target_bias, name="target.layers.0.bias"),
            numpy_helper.from_array(other_bias, name="other.bias"),
        ],
        value_info=[
            helper.make_tensor_value_info("weight_t", TensorProto.FLOAT, (3, 4)),
            helper.make_tensor_value_info(
                "target_projected", TensorProto.FLOAT, (2, 4)
            ),
            helper.make_tensor_value_info("other_projected", TensorProto.FLOAT, (2, 4)),
        ],
    )
    model = helper.make_model(graph)
    onnx.checker.check_model(model, full_check=True)

    assert (
        _replace_linear_subgraphs(onnx, model, bias_name_prefixes=("target.layers.",))
        == 1
    )

    plugin = next(
        node for node in model.graph.node if node.op_type == "PNMIRExactLinear"
    )
    assert list(plugin.input) == ["input", "weight", "target.layers.0.bias"]
    assert any(node.op_type == "Transpose" for node in model.graph.node)
    assert any(node.op_type == "MatMul" for node in model.graph.node)
    _check_model_with_tensorrt_plugins(model)


def test_prunes_fully_replaced_shared_weight_transpose() -> None:
    weight = np.arange(12, dtype=np.float32).reshape(4, 3)
    bias = np.arange(4, dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("Transpose", ("weight",), ("weight_t",), perm=(1, 0)),
            helper.make_node("MatMul", ("input", "weight_t"), ("projected",)),
            helper.make_node("Add", ("projected", "bias"), ("output",)),
        ],
        "pruned-shared-weight-linear",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, (2, 3))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, (2, 4))],
        [
            numpy_helper.from_array(weight, name="weight"),
            numpy_helper.from_array(bias, name="bias"),
        ],
        value_info=[
            helper.make_tensor_value_info("weight_t", TensorProto.FLOAT, (3, 4)),
            helper.make_tensor_value_info("projected", TensorProto.FLOAT, (2, 4)),
        ],
    )
    model = helper.make_model(graph)

    assert _replace_linear_subgraphs(onnx, model) == 1

    assert [node.op_type for node in model.graph.node] == ["PNMIRExactLinear"]
    assert list(model.graph.node[0].input) == ["input", "weight", "bias"]
    _check_model_with_tensorrt_plugins(model)


@pytest.mark.parametrize("namespace", _VALUE_NAMESPACES)
def test_exact_linear_allocates_unique_shared_weight_name(namespace: str) -> None:
    weight = np.arange(12, dtype=np.float32).reshape(3, 4)
    bias = np.arange(4, dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ("input", "weight"), ("projected",)),
            helper.make_node("Add", ("projected", "bias"), ("output",)),
            helper.make_node("Identity", ("weight",), ("retained",)),
        ],
        "collision-safe-exact-linear",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, (2, 3))],
        [
            helper.make_tensor_value_info("output", TensorProto.FLOAT, (2, 4)),
            helper.make_tensor_value_info("retained", TensorProto.FLOAT, (3, 4)),
        ],
        [
            numpy_helper.from_array(weight, name="weight"),
            numpy_helper.from_array(bias, name="bias"),
        ],
    )
    model = helper.make_model(graph)
    reserved = "weight__pnmir_exact_linear_0"
    _reserve_value_name(model, reserved, namespace)
    onnx.checker.check_model(model, full_check=True)

    assert _replace_linear_subgraphs(onnx, model) == 1

    plugin = next(
        node for node in model.graph.node if node.op_type == "PNMIRExactLinear"
    )
    assert plugin.input[1] == f"{reserved}_1"
    _check_model_with_tensorrt_plugins(model)


def test_keeps_rank_one_linear_native() -> None:
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ("input", "weight"), ("projected",)),
            helper.make_node("Add", ("projected", "bias"), ("output",)),
        ],
        "rank-one-native-linear",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, (3,))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, (4,))],
        [
            numpy_helper.from_array(
                np.arange(12, dtype=np.float32).reshape(3, 4), name="weight"
            ),
            numpy_helper.from_array(np.arange(4, dtype=np.float32), name="bias"),
        ],
    )
    model = helper.make_model(graph)
    onnx.checker.check_model(model, full_check=True)

    assert _replace_linear_subgraphs(onnx, model) == 0

    onnx.checker.check_model(model, full_check=True)
    assert [node.op_type for node in model.graph.node] == ["MatMul", "Add"]


@pytest.mark.parametrize("unknown_rank", ("activation", "output"))
def test_keeps_linear_with_unknown_plugin_rank_native(unknown_rank: str) -> None:
    nodes = []
    activation = "input"
    plugin_output = "output"
    if unknown_rank == "activation":
        activation = "activation"
        nodes.append(helper.make_node("Identity", ("input",), (activation,)))
    else:
        plugin_output = "linear_output"
    nodes.extend(
        (
            helper.make_node("MatMul", (activation, "weight"), ("projected",)),
            helper.make_node("Add", ("projected", "bias"), (plugin_output,)),
        )
    )
    if unknown_rank == "output":
        nodes.append(helper.make_node("Identity", (plugin_output,), ("output",)))
    graph = helper.make_graph(
        nodes,
        "unknown-rank-native-linear",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, (2, 3))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, (2, 4))],
        [
            numpy_helper.from_array(
                np.arange(12, dtype=np.float32).reshape(3, 4), name="weight"
            ),
            numpy_helper.from_array(np.arange(4, dtype=np.float32), name="bias"),
        ],
    )
    model = helper.make_model(graph)
    onnx.checker.check_model(model, full_check=True)

    assert _replace_linear_subgraphs(onnx, model) == 0

    onnx.checker.check_model(model, full_check=True)
    assert any(node.op_type == "MatMul" for node in model.graph.node)


@pytest.mark.parametrize(
    ("dtype", "numpy_dtype"),
    (
        (TensorProto.FLOAT16, np.float16),
        (TensorProto.DOUBLE, np.float64),
    ),
)
def test_keeps_non_fp32_cast_boundary_linear_native(
    dtype: int, numpy_dtype: type[np.generic]
) -> None:
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ("input", "weight"), ("projected",)),
            helper.make_node("Add", ("projected", "bias"), ("output",)),
        ],
        "native-typed-linear",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, (2, 3))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, (2, 4))],
        [
            numpy_helper.from_array(
                np.arange(12, dtype=np.float32).reshape(3, 4), name="weight"
            ),
            numpy_helper.from_array(np.arange(4, dtype=np.float32), name="bias"),
        ],
        value_info=[
            helper.make_tensor_value_info("projected", TensorProto.FLOAT, (2, 4))
        ],
    )
    model = _with_non_fp32_cast_boundary(
        helper.make_model(graph),
        input_names=("input",),
        output_name="output",
        initializer_names=("weight", "bias"),
        dtype=dtype,
        numpy_dtype=numpy_dtype,
    )
    onnx.checker.check_model(model, full_check=True)

    assert _replace_linear_subgraphs(onnx, model) == 0

    onnx.checker.check_model(model, full_check=True)
    assert [node.op_type for node in model.graph.node] == [
        "Cast",
        "MatMul",
        "Add",
        "Cast",
    ]


def test_linear_fusion_survives_rematerialized_node_wrappers() -> None:
    weight = np.arange(12, dtype=np.float32).reshape(3, 4)
    bias = np.arange(4, dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ("input", "weight"), ("projected",)),
            helper.make_node("Add", ("projected", "bias"), ("output",)),
        ],
        "rematerialized-exact-linear",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, (2, 3))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, (2, 4))],
        [
            numpy_helper.from_array(weight, name="weight"),
            numpy_helper.from_array(bias, name="bias"),
        ],
    )
    model = helper.make_model(graph)
    onnx.checker.check_model(model)
    rematerialized = _with_rematerialized_nodes(model)

    assert _replace_linear_subgraphs(onnx, rematerialized) == 1

    assert [node.op_type for node in model.graph.node] == ["PNMIRExactLinear"]


def test_rejects_ambiguous_node_output_ownership() -> None:
    graph = helper.make_graph(
        [
            helper.make_node("Identity", ("left",), ("shared",)),
            helper.make_node("Identity", ("right",), ("shared",)),
        ],
        "ambiguous-output-ownership",
        [
            helper.make_tensor_value_info("left", TensorProto.FLOAT, (1,)),
            helper.make_tensor_value_info("right", TensorProto.FLOAT, (1,)),
        ],
        [helper.make_tensor_value_info("shared", TensorProto.FLOAT, (1,))],
    )
    model = helper.make_model(graph)

    with pytest.raises(ValueError, match="ambiguous node ownership"):
        _replace_linear_subgraphs(onnx, model)


def test_linear_with_weight_as_activation_preserves_initializer() -> None:
    weight = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    bias = np.array([5.0, 6.0], dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ("weight", "weight"), ("projected",)),
            helper.make_node("Add", ("projected", "bias"), ("output",)),
        ],
        "shared-linear-operand",
        [],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, (2, 2))],
        [
            numpy_helper.from_array(weight, name="weight"),
            numpy_helper.from_array(bias, name="bias"),
        ],
    )
    model = helper.make_model(graph)
    onnx.checker.check_model(model)

    assert _replace_linear_subgraphs(onnx, model) == 1

    assert len(model.graph.node) == 1
    node = model.graph.node[0]
    assert node.op_type == "PNMIRExactLinear"
    assert list(node.input) == [
        "weight",
        "weight__pnmir_exact_linear_0",
        "bias",
    ]
    initializers = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    np.testing.assert_array_equal(initializers["weight"], weight)
    np.testing.assert_array_equal(initializers[node.input[1]], weight.T)


def test_keeps_linear_matmul_when_its_output_is_also_a_graph_output() -> None:
    weight = np.arange(12, dtype=np.float32).reshape(3, 4)
    bias = np.arange(4, dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ("input", "weight"), ("projected",)),
            helper.make_node("Add", ("projected", "bias"), ("output",)),
        ],
        "retained-linear-intermediate",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, (2, 3))],
        [
            helper.make_tensor_value_info("output", TensorProto.FLOAT, (2, 4)),
            helper.make_tensor_value_info("projected", TensorProto.FLOAT, (2, 4)),
        ],
        [
            numpy_helper.from_array(weight, name="weight"),
            numpy_helper.from_array(bias, name="bias"),
        ],
    )
    model = helper.make_model(graph)
    onnx.checker.check_model(model)

    assert _replace_linear_subgraphs(onnx, _with_rematerialized_nodes(model)) == 0
    onnx.checker.check_model(model)
    assert [node.op_type for node in model.graph.node] == ["MatMul", "Add"]


def test_ignores_dynamic_and_non_fp32_matmuls() -> None:
    integer_weight = np.arange(12, dtype=np.int32).reshape(3, 4)
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ("left", "right"), ("dynamic",)),
            helper.make_node(
                "MatMul", ("integer_input", "integer_weight"), ("integer",)
            ),
        ],
        "unchanged-gemms",
        [
            helper.make_tensor_value_info("left", TensorProto.FLOAT, (2, 3)),
            helper.make_tensor_value_info("right", TensorProto.FLOAT, (3, 4)),
            helper.make_tensor_value_info("integer_input", TensorProto.INT32, (2, 3)),
        ],
        [
            helper.make_tensor_value_info("dynamic", TensorProto.FLOAT, (2, 4)),
            helper.make_tensor_value_info("integer", TensorProto.INT32, (2, 4)),
        ],
        [numpy_helper.from_array(integer_weight, name="integer_weight")],
    )
    model = helper.make_model(graph)

    assert _replace_constant_rhs_matmuls(onnx, model) == 0
    assert [node.op_type for node in model.graph.node] == ["MatMul", "MatMul"]


@pytest.mark.parametrize(
    ("input_shape", "output_shape"),
    (
        ((1, 75, 8, 512), (1, 8, 512)),
        ((1, 32, 8, 128), (1, 8, 128)),
    ),
)
def test_replaces_supported_token_sum(
    input_shape: tuple[int, ...], output_shape: tuple[int, ...]
) -> None:
    graph = helper.make_graph(
        [
            helper.make_node(
                "ReduceSum",
                ("input", "axes"),
                ("output",),
                keepdims=0,
                noop_with_empty_axes=0,
            )
        ],
        "exact-token-sum",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
        [numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes")],
    )
    model = helper.make_model(graph)

    assert _replace_token_sums(onnx, model) == 1

    node = model.graph.node[0]
    assert node.op_type == "PNMIRExactTokenSum"
    assert list(node.input) == ["input"]
    assert list(node.output) == ["output"]


@pytest.mark.parametrize("dtype", (TensorProto.FLOAT16, TensorProto.DOUBLE))
def test_keeps_non_fp32_cast_boundary_token_sum_native(dtype: int) -> None:
    graph = helper.make_graph(
        [
            helper.make_node("Cast", ("input",), ("typed_input",), to=dtype),
            helper.make_node(
                "ReduceSum",
                ("typed_input", "axes"),
                ("typed_output",),
                keepdims=0,
                noop_with_empty_axes=0,
            ),
            helper.make_node(
                "Cast", ("typed_output",), ("output",), to=TensorProto.FLOAT
            ),
        ],
        "native-typed-token-sum",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, (1, 75, 8, 512))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, (1, 8, 512))],
        [numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes")],
        value_info=[
            helper.make_tensor_value_info("typed_input", dtype, (1, 75, 8, 512)),
            helper.make_tensor_value_info("typed_output", dtype, (1, 8, 512)),
        ],
    )
    model = helper.make_model(graph)
    onnx.checker.check_model(model, full_check=True)

    assert _replace_token_sums(onnx, model) == 0

    onnx.checker.check_model(model, full_check=True)
    assert [node.op_type for node in model.graph.node] == [
        "Cast",
        "ReduceSum",
        "Cast",
    ]


def test_keeps_token_sum_with_unknown_plugin_tensor_dtypes_native() -> None:
    graph = helper.make_graph(
        [
            helper.make_node(
                "Cast", ("input",), ("typed_input",), to=TensorProto.FLOAT
            ),
            helper.make_node(
                "ReduceSum",
                ("typed_input", "axes"),
                ("typed_output",),
                keepdims=0,
                noop_with_empty_axes=0,
            ),
            helper.make_node(
                "Cast", ("typed_output",), ("output",), to=TensorProto.FLOAT
            ),
        ],
        "unknown-typed-token-sum",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, (1, 75, 8, 512))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, (1, 8, 512))],
        [numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes")],
        value_info=[
            _shape_only_value_info("typed_input", (1, 75, 8, 512)),
            _shape_only_value_info("typed_output", (1, 8, 512)),
        ],
    )
    model = helper.make_model(graph)
    onnx.checker.check_model(model, full_check=True)

    assert _replace_token_sums(onnx, model) == 0

    onnx.checker.check_model(model, full_check=True)
    assert model.graph.node[1].op_type == "ReduceSum"


@pytest.mark.parametrize(
    ("weights_shape", "features_shape", "output_shape"),
    (
        ((1, 75, 8, 512), (1, 75, 8, 32), (1, 8, 512, 32)),
        ((1, 32, 8, 128), (1, 32, 8, 56), (1, 8, 128, 56)),
    ),
)
def test_replaces_supported_strided_slice_bmm(
    weights_shape: tuple[int, ...],
    features_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> None:
    graph = helper.make_graph(
        [
            helper.make_node(
                "Transpose",
                ("weights",),
                ("weights_permuted",),
                perm=(0, 2, 3, 1),
            ),
            helper.make_node(
                "Transpose",
                ("features",),
                ("features_permuted",),
                perm=(0, 2, 1, 3),
            ),
            helper.make_node(
                "MatMul",
                ("weights_permuted", "features_permuted"),
                ("output",),
            ),
        ],
        "exact-slice-bmm",
        [
            helper.make_tensor_value_info("weights", TensorProto.FLOAT, weights_shape),
            helper.make_tensor_value_info(
                "features", TensorProto.FLOAT, features_shape
            ),
        ],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
    )
    model = helper.make_model(graph)

    assert _replace_slice_bmms(onnx, _with_rematerialized_nodes(model)) == 1

    assert len(model.graph.node) == 1
    node = model.graph.node[0]
    assert node.op_type == "PNMIRExactSliceBmm"
    assert list(node.input) == ["weights", "features"]
    assert list(node.output) == ["output"]


@pytest.mark.parametrize("dtype", (TensorProto.FLOAT16, TensorProto.DOUBLE))
def test_keeps_non_fp32_cast_boundary_slice_bmm_native(dtype: int) -> None:
    graph = helper.make_graph(
        [
            helper.make_node("Cast", ("weights",), ("typed_weights",), to=dtype),
            helper.make_node("Cast", ("features",), ("typed_features",), to=dtype),
            helper.make_node(
                "Transpose",
                ("typed_weights",),
                ("weights_permuted",),
                perm=(0, 2, 3, 1),
            ),
            helper.make_node(
                "Transpose",
                ("typed_features",),
                ("features_permuted",),
                perm=(0, 2, 1, 3),
            ),
            helper.make_node(
                "MatMul",
                ("weights_permuted", "features_permuted"),
                ("typed_output",),
            ),
            helper.make_node(
                "Cast", ("typed_output",), ("output",), to=TensorProto.FLOAT
            ),
        ],
        "native-typed-slice-bmm",
        [
            helper.make_tensor_value_info(
                "weights", TensorProto.FLOAT, (1, 75, 8, 512)
            ),
            helper.make_tensor_value_info(
                "features", TensorProto.FLOAT, (1, 75, 8, 32)
            ),
        ],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, (1, 8, 512, 32))],
        value_info=[
            helper.make_tensor_value_info("typed_weights", dtype, (1, 75, 8, 512)),
            helper.make_tensor_value_info("typed_features", dtype, (1, 75, 8, 32)),
            helper.make_tensor_value_info("weights_permuted", dtype, (1, 8, 512, 75)),
            helper.make_tensor_value_info("features_permuted", dtype, (1, 8, 75, 32)),
            helper.make_tensor_value_info("typed_output", dtype, (1, 8, 512, 32)),
        ],
    )
    model = helper.make_model(graph)
    onnx.checker.check_model(model, full_check=True)

    assert _replace_slice_bmms(onnx, model) == 0

    onnx.checker.check_model(model, full_check=True)
    assert [node.op_type for node in model.graph.node] == [
        "Cast",
        "Cast",
        "Transpose",
        "Transpose",
        "MatMul",
        "Cast",
    ]


def test_keeps_slice_bmm_with_unknown_plugin_tensor_dtypes_native() -> None:
    graph = helper.make_graph(
        [
            helper.make_node(
                "Cast", ("weights",), ("typed_weights",), to=TensorProto.FLOAT
            ),
            helper.make_node(
                "Cast", ("features",), ("typed_features",), to=TensorProto.FLOAT
            ),
            helper.make_node(
                "Transpose",
                ("typed_weights",),
                ("weights_permuted",),
                perm=(0, 2, 3, 1),
            ),
            helper.make_node(
                "Transpose",
                ("typed_features",),
                ("features_permuted",),
                perm=(0, 2, 1, 3),
            ),
            helper.make_node(
                "MatMul",
                ("weights_permuted", "features_permuted"),
                ("typed_output",),
            ),
            helper.make_node(
                "Cast", ("typed_output",), ("output",), to=TensorProto.FLOAT
            ),
        ],
        "unknown-typed-slice-bmm",
        [
            helper.make_tensor_value_info(
                "weights", TensorProto.FLOAT, (1, 75, 8, 512)
            ),
            helper.make_tensor_value_info(
                "features", TensorProto.FLOAT, (1, 75, 8, 32)
            ),
        ],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, (1, 8, 512, 32))],
        value_info=[
            _shape_only_value_info("typed_weights", (1, 75, 8, 512)),
            _shape_only_value_info("typed_features", (1, 75, 8, 32)),
            _shape_only_value_info("typed_output", (1, 8, 512, 32)),
        ],
    )
    model = helper.make_model(graph)
    onnx.checker.check_model(model, full_check=True)

    assert _replace_slice_bmms(onnx, model) == 0

    onnx.checker.check_model(model, full_check=True)
    assert model.graph.node[4].op_type == "MatMul"


@pytest.mark.parametrize(
    ("intermediate", "shape"),
    (
        ("weights_permuted", (1, 8, 512, 75)),
        ("features_permuted", (1, 8, 75, 32)),
    ),
)
def test_keeps_slice_bmm_when_transpose_output_is_a_graph_output(
    intermediate: str, shape: tuple[int, ...]
) -> None:
    graph = helper.make_graph(
        [
            helper.make_node(
                "Transpose",
                ("weights",),
                ("weights_permuted",),
                perm=(0, 2, 3, 1),
            ),
            helper.make_node(
                "Transpose",
                ("features",),
                ("features_permuted",),
                perm=(0, 2, 1, 3),
            ),
            helper.make_node(
                "MatMul",
                ("weights_permuted", "features_permuted"),
                ("output",),
            ),
        ],
        "retained-slice-bmm-intermediate",
        [
            helper.make_tensor_value_info(
                "weights", TensorProto.FLOAT, (1, 75, 8, 512)
            ),
            helper.make_tensor_value_info(
                "features", TensorProto.FLOAT, (1, 75, 8, 32)
            ),
        ],
        [
            helper.make_tensor_value_info("output", TensorProto.FLOAT, (1, 8, 512, 32)),
            helper.make_tensor_value_info(intermediate, TensorProto.FLOAT, shape),
        ],
    )
    model = helper.make_model(graph)
    onnx.checker.check_model(model)

    assert _replace_slice_bmms(onnx, _with_rematerialized_nodes(model)) == 0
    onnx.checker.check_model(model)
    assert [node.op_type for node in model.graph.node] == [
        "Transpose",
        "Transpose",
        "MatMul",
    ]


def test_replaces_last_dimension_layer_norm_and_preserves_epsilon() -> None:
    graph = helper.make_graph(
        [
            helper.make_node(
                "LayerNormalization",
                ("input", "gamma", "beta"),
                ("output",),
                axis=-1,
                epsilon=2.5e-5,
            )
        ],
        "exact-layer-norm",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, (2, 256))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, (2, 256))],
        [
            numpy_helper.from_array(np.ones(256, dtype=np.float32), name="gamma"),
            numpy_helper.from_array(np.zeros(256, dtype=np.float32), name="beta"),
        ],
    )
    model = helper.make_model(graph)

    assert _replace_layer_norms(onnx, model) == 1

    node = model.graph.node[0]
    assert node.op_type == "PNMIRExactLayerNorm"
    attributes = {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in node.attribute
    }
    assert attributes["epsilon"] == np.float32(2.5e-5)
    assert attributes["plugin_version"] == b"1"
    assert attributes["plugin_namespace"] == b""


@pytest.mark.parametrize(
    ("dtype", "numpy_dtype"),
    (
        (TensorProto.FLOAT16, np.float16),
        (TensorProto.DOUBLE, np.float64),
    ),
)
def test_keeps_non_fp32_cast_boundary_layer_norm_native(
    dtype: int, numpy_dtype: type[np.generic]
) -> None:
    graph = helper.make_graph(
        [
            helper.make_node("Cast", ("input",), ("typed_input",), to=dtype),
            helper.make_node(
                "LayerNormalization",
                ("typed_input", "gamma", "beta"),
                ("typed_output",),
                axis=-1,
            ),
            helper.make_node(
                "Cast", ("typed_output",), ("output",), to=TensorProto.FLOAT
            ),
        ],
        "native-typed-layer-norm",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, (2, 256))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, (2, 256))],
        [
            numpy_helper.from_array(np.ones(256, dtype=numpy_dtype), name="gamma"),
            numpy_helper.from_array(np.zeros(256, dtype=numpy_dtype), name="beta"),
        ],
        value_info=[
            helper.make_tensor_value_info("typed_input", dtype, (2, 256)),
            helper.make_tensor_value_info("typed_output", dtype, (2, 256)),
        ],
    )
    model = helper.make_model(graph)
    onnx.checker.check_model(model, full_check=True)

    assert _replace_layer_norms(onnx, model) == 0

    onnx.checker.check_model(model, full_check=True)
    assert [node.op_type for node in model.graph.node] == [
        "Cast",
        "LayerNormalization",
        "Cast",
    ]


def test_keeps_layer_norm_with_non_vectorized_width_native() -> None:
    graph = helper.make_graph(
        [
            helper.make_node(
                "LayerNormalization",
                ("input", "gamma", "beta"),
                ("output",),
                axis=-1,
            )
        ],
        "native-layer-norm",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, (2, 6))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, (2, 6))],
        [
            numpy_helper.from_array(np.ones(6, dtype=np.float32), name="gamma"),
            numpy_helper.from_array(np.zeros(6, dtype=np.float32), name="beta"),
        ],
    )
    model = helper.make_model(graph)
    onnx.checker.check_model(model)

    assert _replace_layer_norms(onnx, model) == 0
    assert model.graph.node[0].op_type == "LayerNormalization"


@pytest.mark.parametrize("outputs", (("output", "mean"), ("output", "mean", "inv_std")))
def test_keeps_multi_output_layer_norm_native(outputs: tuple[str, ...]) -> None:
    graph = helper.make_graph(
        [
            helper.make_node(
                "LayerNormalization",
                ("input", "gamma", "beta"),
                outputs,
                axis=-1,
            )
        ],
        "multi-output-layer-norm",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, (2, 256))],
        [
            helper.make_tensor_value_info("output", TensorProto.FLOAT, (2, 256)),
            *(
                helper.make_tensor_value_info(name, TensorProto.FLOAT, (2, 1))
                for name in outputs[1:]
            ),
        ],
        [
            numpy_helper.from_array(np.ones(256, dtype=np.float32), name="gamma"),
            numpy_helper.from_array(np.zeros(256, dtype=np.float32), name="beta"),
        ],
    )
    model = helper.make_model(graph)
    onnx.checker.check_model(model)

    assert _replace_layer_norms(onnx, model) == 0

    onnx.checker.check_model(model)
    assert model.graph.node[0].op_type == "LayerNormalization"
    assert tuple(model.graph.node[0].output) == outputs


@pytest.mark.parametrize("width", (128, 512))
def test_replaces_supported_last_dimension_softmax(width: int) -> None:
    graph = helper.make_graph(
        [helper.make_node("Softmax", ("input",), ("output",), axis=-1)],
        "exact-softmax",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, (8, width))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, (8, width))],
    )
    model = helper.make_model(graph)

    assert _replace_softmaxes(onnx, model) == 1

    node = model.graph.node[0]
    assert node.op_type == "PNMIRExactSoftmax"
    assert {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in node.attribute
    } == {"plugin_version": b"1", "plugin_namespace": b""}


@pytest.mark.parametrize("dtype", (TensorProto.FLOAT16, TensorProto.DOUBLE))
def test_keeps_non_fp32_cast_boundary_softmax_native(dtype: int) -> None:
    graph = helper.make_graph(
        [
            helper.make_node("Cast", ("input",), ("typed_input",), to=dtype),
            helper.make_node("Softmax", ("typed_input",), ("typed_output",), axis=-1),
            helper.make_node(
                "Cast", ("typed_output",), ("output",), to=TensorProto.FLOAT
            ),
        ],
        "native-typed-softmax",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, (8, 512))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, (8, 512))],
        value_info=[
            helper.make_tensor_value_info("typed_input", dtype, (8, 512)),
            helper.make_tensor_value_info("typed_output", dtype, (8, 512)),
        ],
    )
    model = helper.make_model(graph)
    onnx.checker.check_model(model, full_check=True)

    assert _replace_softmaxes(onnx, model) == 0

    onnx.checker.check_model(model, full_check=True)
    assert [node.op_type for node in model.graph.node] == [
        "Cast",
        "Softmax",
        "Cast",
    ]


def test_keeps_unsupported_width_softmax_native() -> None:
    graph = helper.make_graph(
        [helper.make_node("Softmax", ("input",), ("output",), axis=-1)],
        "native-softmax",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, (8, 256))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, (8, 256))],
    )
    model = helper.make_model(graph)
    onnx.checker.check_model(model)

    assert _replace_softmaxes(onnx, model) == 0
    assert model.graph.node[0].op_type == "Softmax"


def _attention_model(
    query_scale: np.float32 = _ATTENTION_INPUT_SCALE,
    key_scale: np.float32 = _ATTENTION_INPUT_SCALE,
    *,
    shape: tuple[int, ...] = (1, 8, 512, 32),
    key_shape_1: tuple[int, ...] = (8, 512, 32),
    transpose_perm: tuple[int, ...] = (0, 2, 1),
    key_shape_2: tuple[int, ...] = (1, 8, 32, 512),
    output_shape: tuple[int, ...] = (1, 8, 512, 32),
) -> onnx.ModelProto:
    transposed_shape = tuple(key_shape_1[index] for index in transpose_perm)
    graph = helper.make_graph(
        [
            helper.make_node("Reshape", ("key", "key_shape_1"), ("key_r1",)),
            helper.make_node("Transpose", ("key_r1",), ("key_t",), perm=transpose_perm),
            helper.make_node("Reshape", ("key_t", "key_shape_2"), ("key_r2",)),
            helper.make_node("Mul", ("query", "query_scale"), ("query_scaled",)),
            helper.make_node("Mul", ("key_r2", "key_scale"), ("key_scaled",)),
            helper.make_node("MatMul", ("query_scaled", "key_scaled"), ("scores",)),
            helper.make_node("Softmax", ("scores",), ("probabilities",), axis=-1),
            helper.make_node(
                "MatMul",
                ("probabilities", "value"),
                ("attention_output",),
                name="attention",
            ),
        ],
        "exact-attention",
        [
            helper.make_tensor_value_info("query", TensorProto.FLOAT, shape),
            helper.make_tensor_value_info("key", TensorProto.FLOAT, shape),
            helper.make_tensor_value_info("value", TensorProto.FLOAT, shape),
        ],
        [
            helper.make_tensor_value_info(
                "attention_output", TensorProto.FLOAT, output_shape
            )
        ],
        [
            numpy_helper.from_array(
                np.array(key_shape_1, dtype=np.int64), name="key_shape_1"
            ),
            numpy_helper.from_array(
                np.array(key_shape_2, dtype=np.int64), name="key_shape_2"
            ),
            numpy_helper.from_array(query_scale, name="query_scale"),
            numpy_helper.from_array(key_scale, name="key_scale"),
        ],
        value_info=[
            helper.make_tensor_value_info("key_r1", TensorProto.FLOAT, key_shape_1),
            helper.make_tensor_value_info("key_t", TensorProto.FLOAT, transposed_shape),
            helper.make_tensor_value_info("key_r2", TensorProto.FLOAT, key_shape_2),
        ],
    )
    model = helper.make_model(graph)
    onnx.checker.check_model(model)
    return model


def test_replaces_fixed_transolver_attention_decomposition() -> None:
    model = _attention_model()

    assert _replace_attention_subgraphs(onnx, _with_rematerialized_nodes(model)) == 1

    assert [node.op_type for node in model.graph.node] == [
        "PNMIRExactAttention",
        "Transpose",
    ]
    assert list(model.graph.node[0].input) == ["query", "key", "value"]
    assert list(model.graph.node[1].output) == ["attention_output"]


def test_replaces_geotransolver_attention_decomposition() -> None:
    scale = np.float32(56.0**-0.25)
    model = _attention_model(
        scale,
        scale,
        shape=(1, 8, 128, 56),
        key_shape_1=(8, 128, 56),
        key_shape_2=(1, 8, 56, 128),
        output_shape=(1, 8, 128, 56),
    )

    assert _replace_attention_subgraphs(onnx, _with_rematerialized_nodes(model)) == 1
    assert [node.op_type for node in model.graph.node] == [
        "PNMIRExactAttention",
        "Transpose",
    ]


@pytest.mark.parametrize("namespace", _VALUE_NAMESPACES)
def test_exact_attention_allocates_unique_intermediate_name(namespace: str) -> None:
    model = _attention_model()
    reserved = "attention_output__pnmir_bmhd"
    _reserve_value_name(model, reserved, namespace)
    onnx.checker.check_model(model, full_check=True)

    assert _replace_attention_subgraphs(onnx, model) == 1

    plugin = next(
        node for node in model.graph.node if node.op_type == "PNMIRExactAttention"
    )
    assert plugin.output[0] == f"{reserved}_1"
    _check_model_with_tensorrt_plugins(model)


@pytest.mark.parametrize(
    ("dtype", "numpy_dtype"),
    (
        (TensorProto.FLOAT16, np.float16),
        (TensorProto.DOUBLE, np.float64),
    ),
)
def test_keeps_non_fp32_cast_boundary_attention_native(
    dtype: int, numpy_dtype: type[np.generic]
) -> None:
    model = _with_non_fp32_cast_boundary(
        _attention_model(),
        input_names=("query", "key", "value"),
        output_name="attention_output",
        initializer_names=("query_scale", "key_scale"),
        dtype=dtype,
        numpy_dtype=numpy_dtype,
    )
    onnx.checker.check_model(model, full_check=True)

    assert _replace_attention_subgraphs(onnx, model) == 0

    onnx.checker.check_model(model, full_check=True)
    assert all(node.op_type != "PNMIRExactAttention" for node in model.graph.node)


def test_replaces_attention_with_inferred_leading_reshape_dimension() -> None:
    model = _attention_model()
    key_shape = next(
        initializer
        for initializer in model.graph.initializer
        if initializer.name == "key_shape_1"
    )
    key_shape.CopyFrom(
        numpy_helper.from_array(
            np.array([-1, 512, 32], dtype=np.int64), name="key_shape_1"
        )
    )
    onnx.checker.check_model(model)

    assert _replace_attention_subgraphs(onnx, _with_rematerialized_nodes(model)) == 1

    assert [node.op_type for node in model.graph.node] == [
        "PNMIRExactAttention",
        "Transpose",
    ]


@pytest.mark.parametrize(
    "target",
    ((-1, -1, 32), (0, 512, 32), (-1, 256, 32)),
)
def test_rejects_attention_with_ambiguous_inferred_reshape(
    target: tuple[int, ...],
) -> None:
    model = _attention_model()
    key_shape = next(
        initializer
        for initializer in model.graph.initializer
        if initializer.name == "key_shape_1"
    )
    key_shape.CopyFrom(
        numpy_helper.from_array(np.array(target, dtype=np.int64), name="key_shape_1")
    )

    assert _replace_attention_subgraphs(onnx, model) == 0
    assert all(node.op_type != "PNMIRExactAttention" for node in model.graph.node)


@pytest.mark.parametrize(
    ("intermediate", "shape"),
    (
        ("key_r1", (8, 512, 32)),
        ("key_t", (8, 32, 512)),
        ("key_r2", (1, 8, 32, 512)),
        ("query_scaled", (1, 8, 512, 32)),
        ("key_scaled", (1, 8, 32, 512)),
        ("scores", (1, 8, 512, 512)),
        ("probabilities", (1, 8, 512, 512)),
    ),
)
@pytest.mark.parametrize("external_use", ("graph_output", "identity"))
def test_keeps_attention_with_externally_used_intermediate(
    intermediate: str, shape: tuple[int, ...], external_use: str
) -> None:
    model = _attention_model()
    if external_use == "identity":
        model.graph.node.append(
            helper.make_node("Identity", (intermediate,), ("retained",))
        )
        retained_name = "retained"
    else:
        retained_name = intermediate
    model.graph.output.append(
        helper.make_tensor_value_info(retained_name, TensorProto.FLOAT, shape)
    )
    onnx.checker.check_model(model)

    assert _replace_attention_subgraphs(onnx, _with_rematerialized_nodes(model)) == 0
    onnx.checker.check_model(model)
    assert all(node.op_type != "PNMIRExactAttention" for node in model.graph.node)


def test_rejects_attention_with_wrong_query_or_key_multiplier() -> None:
    for query_scale, key_scale in (
        (np.float32(0.5), _ATTENTION_INPUT_SCALE),
        (_ATTENTION_INPUT_SCALE, np.float32(0.5)),
    ):
        model = _attention_model(query_scale, key_scale)

        assert _replace_attention_subgraphs(onnx, model) == 0
        assert all(node.op_type != "PNMIRExactAttention" for node in model.graph.node)


def test_rejects_attention_with_identity_or_wrong_key_transpose() -> None:
    for transpose_perm in ((0, 1, 2), (1, 0, 2)):
        model = _attention_model(transpose_perm=transpose_perm)

        assert _replace_attention_subgraphs(onnx, model) == 0
        assert all(node.op_type != "PNMIRExactAttention" for node in model.graph.node)


def test_rejects_attention_with_wrong_key_reshape_constants() -> None:
    for layout in (
        {"key_shape_1": (8, 32, 512)},
        {
            "key_shape_2": (1, 1, 8, 32, 512),
            "output_shape": (1, 1, 8, 512, 32),
        },
    ):
        model = _attention_model(**layout)

        assert _replace_attention_subgraphs(onnx, model) == 0
        assert all(node.op_type != "PNMIRExactAttention" for node in model.graph.node)


def _gelu_model(
    *,
    sqrt_two: np.float32 = _GELU_SQRT_TWO,
    one: np.float32 = _GELU_ONE,
    half: np.float32 = _GELU_HALF,
    shape: tuple[int, ...] = (2, 512),
) -> onnx.ModelProto:
    graph = helper.make_graph(
        [
            helper.make_node("Div", ("input", "sqrt_two"), ("divided",)),
            helper.make_node("Erf", ("divided",), ("erf",)),
            helper.make_node("Add", ("erf", "one"), ("plus_one",)),
            helper.make_node("Mul", ("half", "plus_one"), ("scaled",)),
            helper.make_node("Mul", ("input", "scaled"), ("output",)),
        ],
        "exact-gelu",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
        [
            numpy_helper.from_array(sqrt_two, name="sqrt_two"),
            numpy_helper.from_array(one, name="one"),
            numpy_helper.from_array(half, name="half"),
        ],
    )
    return helper.make_model(graph)


def test_replaces_exact_gelu_expansion() -> None:
    model = _gelu_model()

    assert _replace_gelu_subgraphs(onnx, _with_rematerialized_nodes(model)) == 1

    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == "PNMIRExactGelu"
    assert list(model.graph.node[0].input) == ["input"]
    assert list(model.graph.node[0].output) == ["output"]


def test_scoped_gelu_only_replaces_exact_linear_source() -> None:
    model = _gelu_model()

    assert _replace_gelu_subgraphs(onnx, model, exact_linear_sources_only=True) == 0

    model = _gelu_model()
    model.graph.node.insert(
        0,
        helper.make_node(
            "PNMIRExactLinear",
            ("source", "weight", "bias"),
            ("input",),
        ),
    )
    model.graph.input[0].name = "source"
    model.graph.value_info.append(
        helper.make_tensor_value_info("input", TensorProto.FLOAT, (2, 512))
    )
    model.graph.initializer.extend(
        (
            numpy_helper.from_array(np.eye(512, dtype=np.float32), name="weight"),
            numpy_helper.from_array(np.zeros(512, dtype=np.float32), name="bias"),
        )
    )

    assert _replace_gelu_subgraphs(onnx, model, exact_linear_sources_only=True) == 1
    assert sum(node.op_type == "PNMIRExactGelu" for node in model.graph.node) == 1


def test_keeps_scalar_exact_gelu_expansion_native() -> None:
    model = _gelu_model(shape=())
    onnx.checker.check_model(model, full_check=True)

    assert _replace_gelu_subgraphs(onnx, model) == 0

    onnx.checker.check_model(model, full_check=True)
    assert all(node.op_type != "PNMIRExactGelu" for node in model.graph.node)


@pytest.mark.parametrize(
    ("dtype", "numpy_dtype"),
    (
        (TensorProto.FLOAT16, np.float16),
        (TensorProto.DOUBLE, np.float64),
    ),
)
def test_keeps_non_fp32_cast_boundary_gelu_native(
    dtype: int, numpy_dtype: type[np.generic]
) -> None:
    model = _with_non_fp32_cast_boundary(
        _gelu_model(),
        input_names=("input",),
        output_name="output",
        initializer_names=("sqrt_two", "one", "half"),
        dtype=dtype,
        numpy_dtype=numpy_dtype,
    )
    onnx.checker.check_model(model, full_check=True)

    assert _replace_gelu_subgraphs(onnx, model) == 0

    onnx.checker.check_model(model, full_check=True)
    assert all(node.op_type != "PNMIRExactGelu" for node in model.graph.node)


@pytest.mark.parametrize("intermediate", ("divided", "erf", "plus_one", "scaled"))
def test_keeps_gelu_expansion_with_externally_consumed_intermediate(
    intermediate: str,
) -> None:
    model = _gelu_model()
    model.graph.node.append(
        helper.make_node("Identity", (intermediate,), ("retained",))
    )
    model.graph.output.append(
        helper.make_tensor_value_info("retained", TensorProto.FLOAT, (2, 512))
    )
    onnx.checker.check_model(model)

    assert _replace_gelu_subgraphs(onnx, _with_rematerialized_nodes(model)) == 0
    onnx.checker.check_model(model)
    assert all(node.op_type != "PNMIRExactGelu" for node in model.graph.node)


def test_rejects_exact_gelu_expansion_with_wrong_constants() -> None:
    for constants in (
        {"sqrt_two": np.float32(1.5)},
        {"one": np.float32(2.0)},
        {"half": np.float32(0.25)},
    ):
        model = _gelu_model(**constants)

        assert _replace_gelu_subgraphs(onnx, model) == 0
        assert all(node.op_type != "PNMIRExactGelu" for node in model.graph.node)
