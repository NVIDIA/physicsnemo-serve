"""Semantic and fail-closed tests for the bounded Transolver deslice rewrite."""
from __future__ import annotations

import unittest

try:
    import numpy as np
    import onnx
    import pytest
    from onnx import TensorProto, helper
except ImportError as error:
    raise unittest.SkipTest("deslice graph tests require NumPy, ONNX and pytest") from error

from pnmir_export import tensorrt_exact_graphs as graphs
from test_tensorrt_exact_graphs import (
    _check_model_with_tensorrt_plugins,
    _reserve_value_name,
    _with_rematerialized_nodes,
)


def fixture():
    # Raw attention deliberately has no value_info, matching the real rewrite.
    weights, attention, output = (1, 3, 2, 6), (1, 2, 6, 4), (1, 3, 2, 4)
    value = lambda name, shape: helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
    return helper.make_model(helper.make_graph(
        [
            helper.make_node("PNMIRExactAttention", ("q", "k", "v"), ("raw",), plugin_version="1", plugin_namespace=""),
            helper.make_node("Transpose", ("raw",), ("attention",), perm=(0, 2, 1, 3)),
            helper.make_node("Einsum", ("weights", "attention"), ("output",), equation="bths,bhsd->bthd"),
        ],
        "deslice", [value("weights", weights), *(value(name, attention) for name in ("q", "k", "v"))],
        [value("output", output)], value_info=[value("attention", attention)],
    ))


def evaluate(model):
    # Fixed small integers make all sums exact, isolating tensor layout rather
    # than relying on NumPy's reduction order to reproduce CUDA arithmetic.
    values = {
        "weights": (np.arange(36, dtype=np.float32) % 5).reshape(1, 3, 2, 6),
        "raw": (np.arange(48, dtype=np.float32) % 7).reshape(1, 6, 2, 4),
    }
    for node in model.graph.node:
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if node.op_type == "PNMIRExactAttention":
            continue  # Both graphs consume the identical captured attention result.
        if node.op_type == "Einsum":
            result = np.einsum(attrs["equation"].decode(), *(values[name] for name in node.input))
        elif node.op_type == "PNMIRExactDesliceBmm":
            weights, raw_attention = (values[name] for name in node.input)
            result = np.matmul(weights.transpose(0, 2, 1, 3), raw_attention.transpose(0, 2, 1, 3))
        elif node.op_type == "Transpose":
            result = values[node.input[0]].transpose(attrs["perm"])
        elif node.op_type == "Identity":
            result = values[node.input[0]]
        else:
            raise AssertionError(node.op_type)
        values[node.output[0]] = result
    return {output.name: values[output.name] for output in model.graph.output}


def test_deslice_preserves_semantics_using_raw_attention_without_value_info():
    model = fixture()
    expected = evaluate(model)
    assert graphs._replace_deslice_bmms(onnx, model) == 1
    plugin = next(n for n in model.graph.node if n.op_type == "PNMIRExactDesliceBmm")
    assert list(plugin.input) == ["weights", "raw"]
    assert all(n.op_type != "Einsum" for n in model.graph.node)
    np.testing.assert_array_equal(evaluate(model)["output"], expected["output"])
    _check_model_with_tensorrt_plugins(model)


def test_deslice_preserves_shared_and_exposed_attention_results():
    model = fixture()
    model.graph.node.append(helper.make_node("Identity", ("attention",), ("retained",)))
    for name in ("attention", "retained"):
        model.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, (1, 2, 6, 4)))
    expected = evaluate(model)
    assert graphs._replace_deslice_bmms(onnx, model) == 1
    actual = evaluate(model)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])
    _check_model_with_tensorrt_plugins(model)


@pytest.mark.parametrize("namespace", ("input", "output", "value_info", "initializer", "node_output"))
def test_deslice_allocates_collision_free_names(namespace):
    model = fixture()
    _reserve_value_name(model, "output__pnmir_bhtd", namespace)
    assert graphs._replace_deslice_bmms(onnx, model) == 1
    plugin = next(n for n in model.graph.node if n.op_type == "PNMIRExactDesliceBmm")
    assert plugin.output[0] != "output__pnmir_bhtd"
    _check_model_with_tensorrt_plugins(model)


def test_deslice_uses_stable_protobuf_node_identity():
    model = fixture()
    assert graphs._replace_deslice_bmms(onnx, _with_rematerialized_nodes(model)) == 1
    assert len([n for n in model.graph.node if n.op_type == "PNMIRExactDesliceBmm"]) == 1
    _check_model_with_tensorrt_plugins(model)


@pytest.mark.parametrize("invalid", (
    "dtype", "symbolic", "batch", "heads", "single_head", "slices", "output", "permutation",
    "producer", "version", "namespace", "qkv", "raw_shape", "raw_dtype", "int_limit", "leading_dim",
))
def test_targeted_unsupported_deslice_is_rejected_without_mutation(invalid):
    model = fixture()
    weights = model.graph.input[0].type.tensor_type
    if invalid == "dtype":
        weights.elem_type = TensorProto.DOUBLE
    elif invalid == "symbolic":
        weights.shape.dim[1].dim_param = "tokens"
    elif invalid == "batch":
        weights.shape.dim[0].dim_value = 2
    elif invalid == "heads":
        weights.shape.dim[2].dim_value = 3
    elif invalid == "single_head":
        # Torch's batch-one GEMM dispatch is outside this strided BMM profile.
        weights.shape.dim[2].dim_value = 1
        model.graph.output[0].type.tensor_type.shape.dim[2].dim_value = 1
        for value in (*model.graph.input[1:], model.graph.value_info[0]):
            value.type.tensor_type.shape.dim[1].dim_value = 1
    elif invalid == "slices":
        weights.shape.dim[3].dim_value = 7
    elif invalid == "output":
        model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 4
    elif invalid == "permutation":
        model.graph.node[1].attribute[0].ints[:] = [0, 1, 2, 3]
    elif invalid == "producer":
        model.graph.node[0].op_type = "OtherAttention"
    elif invalid in ("version", "namespace"):
        attribute = "plugin_version" if invalid == "version" else "plugin_namespace"
        next(a for a in model.graph.node[0].attribute if a.name == attribute).s = b"other"
    elif invalid == "qkv":
        model.graph.input[2].type.tensor_type.shape.dim[2].dim_value = 7
    elif invalid == "raw_shape":
        model.graph.value_info.append(helper.make_tensor_value_info("raw", TensorProto.FLOAT, (1, 2, 6, 4)))
    elif invalid == "raw_dtype":
        model.graph.value_info.append(helper.make_tensor_value_info("raw", TensorProto.DOUBLE, (1, 6, 2, 4)))
    elif invalid == "int_limit":
        weights.shape.dim[1].dim_value = 2**31
        model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 2**31
    elif invalid == "leading_dim":
        # Shared shapes remain consistent while H*S exceeds signed-int lda.
        weights.shape.dim[3].dim_value = 2**30
        for value in (*model.graph.input[1:], model.graph.value_info[0]):
            value.type.tensor_type.shape.dim[2].dim_value = 2**30
    original = model.SerializeToString()
    with pytest.raises(ValueError, match="unsupported.*deslice"):
        graphs._replace_deslice_bmms(onnx, model)
    assert model.SerializeToString() == original


def test_partial_targeted_rewrite_fails_before_changing_any_nodes():
    model = fixture()
    model.graph.node.append(helper.make_node("Einsum", ("weights", "weights"), ("bad",), equation="bths,bhsd->bthd"))
    model.graph.output.append(helper.make_tensor_value_info("bad", TensorProto.FLOAT, (1, 3, 2, 4)))
    original = model.SerializeToString()
    with pytest.raises(ValueError, match="unsupported.*deslice"):
        graphs._replace_deslice_bmms(onnx, model)
    assert model.SerializeToString() == original


def test_other_equations_are_left_unchanged():
    model = fixture()
    model.graph.node[-1].attribute[0].s = b"bths,bhsd->bhtd"
    original = model.SerializeToString()
    assert graphs._replace_deslice_bmms(onnx, model) == 0
    assert model.SerializeToString() == original
