"""Geo deslicing must retain the eager layout after mixing two exact attentions."""
from __future__ import annotations

import unittest

try:
    import numpy as np
    import onnx
    import pytest
    from onnx import TensorProto, helper, numpy_helper
except ImportError as error:
    raise unittest.SkipTest("Geo deslice tests require NumPy, ONNX and pytest") from error

from pnmir_export import tensorrt_exact_graphs as graphs
from test_tensorrt_deslice_graphs import fixture as attention_fixture
from test_tensorrt_exact_graphs import (
    _check_model_with_tensorrt_plugins,
    _reserve_value_name,
    _with_rematerialized_nodes,
)


def fixture(*, split=False):
    model = attention_fixture()
    nodes = list(model.graph.node[:2])
    nodes.extend((
        helper.make_node("PNMIRExactAttention", ("cq", "ck", "cv"), ("cross_raw",), plugin_version="1", plugin_namespace=""),
        helper.make_node("Transpose", ("cross_raw",), ("cross_attention",), perm=(0, 2, 1, 3)),
        helper.make_node("PNMIRExactWeightedBlend", ("attention", "alpha", "cross_attention", "beta"), ("mixed",), plugin_version="1", plugin_namespace=""),
        helper.make_node("Einsum", ("weights", "mixed"), ("output",), equation="bths,bhsd->bthd"),
    ))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    for name in ("cq", "ck", "cv"):
        model.graph.input.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, (1, 2, 6, 4)))
    for name in ("cross_attention", "mixed"):
        model.graph.value_info.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, (1, 2, 6, 4)))
    model.graph.initializer.extend((
        numpy_helper.from_array(np.array(0.25, dtype=np.float32), "alpha"),
        numpy_helper.from_array(np.array(0.75, dtype=np.float32), "beta"),
    ))
    if split:
        # The installed Geo exporter retains torch.split(...)[0] for cross SDPA.
        node = helper.make_node("Split", ("cross_attention",), ("cross_split",), axis=2, num_outputs=1)
        model.graph.node.insert(4, node)
        model.graph.node[5].input[2] = "cross_split"
        model.graph.value_info.append(helper.make_tensor_value_info("cross_split", TensorProto.FLOAT, (1, 2, 6, 4)))
    return model


def evaluate(model):
    # Integers and quarter weights isolate layout from floating reduction choices.
    values = {
        "weights": (np.arange(36, dtype=np.float32) % 5).reshape(1, 3, 2, 6),
        "raw": (np.arange(48, dtype=np.float32) % 7).reshape(1, 6, 2, 4),
        "cross_raw": (np.arange(48, dtype=np.float32) % 9 - 3).reshape(1, 6, 2, 4),
        **{value.name: numpy_helper.to_array(value) for value in model.graph.initializer},
    }
    for node in model.graph.node:
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if node.op_type == "PNMIRExactAttention":
            continue
        if node.op_type == "PNMIRExactWeightedBlend":
            left, alpha, right, beta = (values[name] for name in node.input)
            result = left * alpha + right * beta
        elif node.op_type == "PNMIRExactDesliceBmm":
            weights, mixed = (values[name] for name in node.input)
            assert mixed.shape == (1, 6, 2, 4), "plugin requires physical BSHD"
            result = np.matmul(weights.transpose(0, 2, 1, 3), mixed.transpose(0, 2, 1, 3))
        elif node.op_type == "Einsum":
            result = np.einsum(attrs["equation"].decode(), *(values[name] for name in node.input))
        elif node.op_type == "Transpose":
            result = values[node.input[0]].transpose(attrs["perm"])
        elif node.op_type == "Split":
            assert len(node.output) == attrs["num_outputs"] == 1
            result = values[node.input[0]]
        else:
            raise AssertionError(node.op_type)
        values[node.output[0]] = result
    return {value.name: values[value.name] for value in model.graph.output}


@pytest.mark.parametrize("split", (False, True))
def test_geo_deslice_after_weighted_attention_preserves_values_and_shared_blend(split):
    model = fixture(split=split)
    model.graph.output.append(helper.make_tensor_value_info("mixed", TensorProto.FLOAT, (1, 2, 6, 4)))
    expected = evaluate(model)
    assert graphs._replace_deslice_bmms(onnx, model) == 1
    assert sum(node.op_type == "PNMIRExactDesliceBmm" for node in model.graph.node) == 1
    assert not any(node.op_type == "Einsum" for node in model.graph.node)
    actual = evaluate(model)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])
    _check_model_with_tensorrt_plugins(model)


@pytest.mark.parametrize("invalid", (
    "self_producer", "cross_producer", "cross_permutation", "cross_shape",
    "blend_version", "blend_namespace", "weights_dtype", "mixed_dtype",
    "coefficient_dtype", "coefficient_shape", "coefficient_nonfinite",
    "blend_domain", "blend_inputs", "blend_outputs", "self_domain", "cross_domain",
    "attention_inputs", "attention_outputs", "transpose_domain", "transpose_inputs",
    "coefficient_graph_input",
))
def test_geo_rejects_unproven_blend_layout_without_mutation(invalid):
    model = fixture()
    if invalid == "self_producer":
        model.graph.node[0].op_type = "OtherAttention"
    elif invalid == "cross_producer":
        model.graph.node[2].op_type = "OtherAttention"
    elif invalid == "cross_permutation":
        model.graph.node[3].attribute[0].ints[:] = [0, 1, 2, 3]
    elif invalid == "cross_shape":
        model.graph.value_info[1].type.tensor_type.shape.dim[2].dim_value = 7
    elif invalid in ("blend_version", "blend_namespace"):
        name = "plugin_version" if invalid == "blend_version" else "plugin_namespace"
        next(a for a in model.graph.node[4].attribute if a.name == name).s = b"other"
    elif invalid == "weights_dtype":
        model.graph.input[0].type.tensor_type.elem_type = TensorProto.DOUBLE
    elif invalid == "mixed_dtype":
        model.graph.value_info[2].type.tensor_type.elem_type = TensorProto.DOUBLE
    elif invalid == "blend_domain":
        model.graph.node[4].domain = "other"
    elif invalid == "blend_inputs":
        model.graph.node[4].input.append("alpha")
    elif invalid == "blend_outputs":
        model.graph.node[4].output.append("mixed_other")
    elif invalid in ("self_domain", "cross_domain"):
        model.graph.node[0 if invalid == "self_domain" else 2].domain = "other"
    elif invalid == "attention_inputs":
        model.graph.node[2].input.append("cq")
    elif invalid == "attention_outputs":
        model.graph.node[2].output.append("cross_raw_other")
    elif invalid == "transpose_domain":
        model.graph.node[3].domain = "other"
    elif invalid == "transpose_inputs":
        model.graph.node[3].input.append("cross_raw")
    elif invalid == "coefficient_graph_input":
        model.graph.input.append(helper.make_tensor_value_info("alpha", TensorProto.FLOAT, ()))
    else:
        value = {"coefficient_dtype": np.array(0.25, dtype=np.float64),
                 "coefficient_shape": np.array([0.25], dtype=np.float32),
                 "coefficient_nonfinite": np.array(np.nan, dtype=np.float32)}[invalid]
        model.graph.initializer[0].CopyFrom(numpy_helper.from_array(value, "alpha"))
    original = model.SerializeToString()
    with pytest.raises(ValueError, match="unsupported.*deslice"):
        graphs._replace_deslice_bmms(onnx, model)
    assert model.SerializeToString() == original


@pytest.mark.parametrize("invalid", ("outputs", "shape", "dtype", "domain", "axis", "split_input"))
def test_geo_cross_attention_requires_proven_identity_split(invalid):
    model = fixture(split=True)
    split = model.graph.node[4]
    if invalid == "outputs":
        next(a for a in split.attribute if a.name == "num_outputs").i = 2
        split.output.append("cross_split_other")
    elif invalid == "shape":
        model.graph.value_info[-1].type.tensor_type.shape.dim[2].dim_value = 3
    elif invalid == "dtype":
        model.graph.value_info[-1].type.tensor_type.elem_type = TensorProto.DOUBLE
    elif invalid == "domain":
        split.domain = "other"
    elif invalid == "axis":
        next(a for a in split.attribute if a.name == "axis").i = 1
    elif invalid == "split_input":
        model.graph.initializer.append(numpy_helper.from_array(np.array([3, 3], dtype=np.int64), "sections"))
        split.input.append("sections")
    original = model.SerializeToString()
    with pytest.raises(ValueError, match="unsupported.*deslice"):
        graphs._replace_deslice_bmms(onnx, model)
    assert model.SerializeToString() == original


@pytest.mark.parametrize("namespace", ("input", "output", "value_info", "initializer", "node_output"))
def test_geo_blend_repack_uses_collision_free_names(namespace):
    model = fixture(split=True)
    _reserve_value_name(model, "mixed__pnmir_bshd", namespace)
    assert graphs._replace_deslice_bmms(onnx, model) == 1
    plugin = next(n for n in model.graph.node if n.op_type == "PNMIRExactDesliceBmm")
    assert plugin.input[1] != "mixed__pnmir_bshd"
    _check_model_with_tensorrt_plugins(model)


def test_geo_uses_stable_protobuf_node_identity():
    model = fixture(split=True)
    assert graphs._replace_deslice_bmms(onnx, _with_rematerialized_nodes(model)) == 1
    assert sum(n.op_type == "PNMIRExactDesliceBmm" for n in model.graph.node) == 1
    _check_model_with_tensorrt_plugins(model)


def test_geo_valid_then_invalid_deslice_rejects_without_partial_mutation():
    model = fixture(split=True)
    model.graph.node.append(helper.make_node("Einsum", ("weights", "weights"), ("bad",), equation="bths,bhsd->bthd"))
    model.graph.output.append(helper.make_tensor_value_info("bad", TensorProto.FLOAT, (1, 3, 2, 4)))
    original = model.SerializeToString()
    with pytest.raises(ValueError, match="unsupported.*deslice"):
        graphs._replace_deslice_bmms(onnx, model)
    assert model.SerializeToString() == original
