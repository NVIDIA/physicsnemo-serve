"""Bounded ONNX substitutions for the native layout-order-exact profile.

Mechanically ported from gpu_programming/python/pnmir_export/tensorrt_builder.py.
Retains the qualified Transolver substitutions and GeoTransolver weighted blend.
These matchers reject unsupported layouts; they are not general ONNX kernels.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

_EXACT_GEMM_PLUGIN_NAME = "PNMIRExactGemm"


_EXACT_GEMM_PLUGIN_VERSION = "1"


_EXACT_LINEAR_PLUGIN_NAME = "PNMIRExactLinear"


_EXACT_LINEAR_PLUGIN_VERSION = "1"


_EXACT_TOKEN_SUM_PLUGIN_NAME = "PNMIRExactTokenSum"


_EXACT_TOKEN_SUM_PLUGIN_VERSION = "1"


_EXACT_SLICE_BMM_PLUGIN_NAME = "PNMIRExactSliceBmm"


_EXACT_SLICE_BMM_PLUGIN_VERSION = "1"


_EXACT_LAYER_NORM_PLUGIN_NAME = "PNMIRExactLayerNorm"


_EXACT_LAYER_NORM_PLUGIN_VERSION = "1"


_EXACT_SOFTMAX_PLUGIN_NAME = "PNMIRExactSoftmax"


_EXACT_SOFTMAX_PLUGIN_VERSION = "1"


_EXACT_ATTENTION_PLUGIN_NAME = "PNMIRExactAttention"


_EXACT_ATTENTION_PLUGIN_VERSION = "1"


_EXACT_GELU_PLUGIN_NAME = "PNMIRExactGelu"


_EXACT_GELU_PLUGIN_VERSION = "1"


_TENSORRT_MAX_DIMS = 8


_NodeKey = tuple[str, ...]


def _static_tensor_shapes(model: Any) -> dict[str, list[int]]:
    shapes = {
        initializer.name: list(initializer.dims)
        for initializer in model.graph.initializer
    }
    for value in (*model.graph.input, *model.graph.value_info, *model.graph.output):
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape") or any(
            not dimension.HasField("dim_value")
            for dimension in tensor_type.shape.dim
        ):
            continue
        shapes[value.name] = [
            dimension.dim_value for dimension in tensor_type.shape.dim
        ]
    return shapes


def _static_tensor_dtypes(model: Any) -> dict[str, int]:
    dtypes = {
        initializer.name: initializer.data_type
        for initializer in model.graph.initializer
    }
    for value in (*model.graph.input, *model.graph.value_info, *model.graph.output):
        tensor_type = value.type.tensor_type
        if tensor_type.HasField("elem_type"):
            dtypes[value.name] = tensor_type.elem_type
    return dtypes


def _tensor_ranks(model: Any) -> dict[str, int]:
    ranks = {
        initializer.name: len(initializer.dims)
        for initializer in model.graph.initializer
    }
    for value in (*model.graph.input, *model.graph.value_info, *model.graph.output):
        tensor_type = value.type.tensor_type
        if tensor_type.HasField("shape"):
            ranks[value.name] = len(tensor_type.shape.dim)
    return ranks


def _plugin_ranks_are_compatible(
    ranks: dict[str, int],
    activation: str,
    outputs: Sequence[str],
    *,
    minimum_rank: int = 2,
) -> bool:
    if len(outputs) != 1:
        return False
    activation_rank = ranks.get(activation)
    output_rank = ranks.get(outputs[0])
    return (
        activation_rank is not None
        and minimum_rank <= activation_rank <= _TENSORRT_MAX_DIMS
        and output_rank == activation_rank
    )


@dataclass
class _OnnxValueNameAllocator:
    used: set[str]

    @classmethod
    def from_model(cls, model: Any) -> _OnnxValueNameAllocator:
        used = {
            value.name
            for values in (
                model.graph.input,
                model.graph.output,
                model.graph.value_info,
                model.graph.initializer,
            )
            for value in values
            if value.name
        }
        used.update(
            output for node in model.graph.node for output in node.output if output
        )
        return cls(used)

    def allocate(self, preferred: str) -> str:
        candidate = preferred
        suffix = 1
        while candidate in self.used:
            candidate = f"{preferred}_{suffix}"
            suffix += 1
        self.used.add(candidate)
        return candidate


@dataclass
class _TensorConsumers:
    owners: set[_NodeKey | None] = field(default_factory=set)
    occurrences: int = 0

    def add(self, owner: _NodeKey | None) -> None:
        self.owners.add(owner)
        self.occurrences += 1


def _node_key(node: Any) -> _NodeKey:
    key = tuple(output for output in node.output if output)
    if not key:
        raise ValueError("ONNX graph nodes must have at least one named output")
    return key


def _validate_node_keys(model: Any) -> None:
    owners: dict[str, _NodeKey] = {}
    for node in model.graph.node:
        key = _node_key(node)
        for output in key:
            if output in owners:
                raise ValueError(
                    f"ONNX graph tensor {output!r} has ambiguous node ownership"
                )
            owners[output] = key


def _tensor_consumers(model: Any) -> dict[str, _TensorConsumers]:
    _validate_node_keys(model)
    consumers: dict[str, _TensorConsumers] = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, _TensorConsumers()).add(_node_key(node))
    for output in model.graph.output:
        consumers.setdefault(output.name, _TensorConsumers()).add(None)
    return consumers


def _removed_tensors_are_internal(
    removed_nodes: Sequence[Any],
    fusion_nodes: Sequence[Any],
    consumers: dict[str, _TensorConsumers],
) -> bool:
    fusion_keys = {_node_key(node) for node in fusion_nodes}
    return all(
        consumers.get(output, _TensorConsumers()).owners.issubset(fusion_keys)
        for node in removed_nodes
        for output in node.output
    )


def _replace_constant_rhs_matmuls(onnx: Any, model: Any) -> int:
    """Route static linear projections through the exact cuBLAS plugin."""
    import numpy as np

    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    consumers = _tensor_consumers(model)
    ranks = _tensor_ranks(model)
    names = _OnnxValueNameAllocator.from_model(model)

    replacements = 0
    for node in model.graph.node:
        if node.op_type != "MatMul" or len(node.input) != 2:
            continue
        if not _plugin_ranks_are_compatible(ranks, node.input[0], node.output):
            continue
        initializer = initializers.get(node.input[1])
        if initializer is None:
            continue
        weight = onnx.numpy_helper.to_array(initializer)
        if weight.ndim != 2 or weight.dtype != np.float32:
            continue

        transposed = np.ascontiguousarray(weight.T)
        if consumers[initializer.name].occurrences == 1:
            initializer.CopyFrom(
                onnx.numpy_helper.from_array(transposed, name=initializer.name)
            )
        else:
            name = names.allocate(
                f"{initializer.name}__pnmir_exact_gemm_{replacements}"
            )
            replacement = onnx.numpy_helper.from_array(transposed, name=name)
            model.graph.initializer.append(replacement)
            node.input[1] = name

        node.op_type = _EXACT_GEMM_PLUGIN_NAME
        node.attribute.extend(
            (
                onnx.helper.make_attribute(
                    "plugin_version", _EXACT_GEMM_PLUGIN_VERSION
                ),
                onnx.helper.make_attribute("plugin_namespace", ""),
            )
        )
        replacements += 1
    return replacements


def _replace_linear_subgraphs(
    onnx: Any,
    model: Any,
    *,
    bias_name_prefixes: Sequence[str] = (),
) -> int:
    """Fuse selected ONNX MatMul-plus-bias expansions into exact CUDA linears.

    Dynamo exports a reused ``nn.Linear`` weight as a shared Transpose node,
    while a weight used once is commonly folded into a transposed initializer.
    Both forms describe the same PyTorch operation and must be recognized.
    ``bias_name_prefixes`` provides a stable module-qualified scope when only a
    numerically divergent part of a larger graph should use the plugin.
    """
    import numpy as np

    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    consumers = _tensor_consumers(model)
    ranks = _tensor_ranks(model)
    names = _OnnxValueNameAllocator.from_model(model)
    producer = {output: node for node in model.graph.node for output in node.output}
    replacements: dict[_NodeKey, tuple[Any, Any, str, str]] = {}
    removed_keys: set[_NodeKey] = set()
    candidate_transpose_keys: set[_NodeKey] = set()
    count = 0
    for add in model.graph.node:
        if add.op_type != "Add" or len(add.input) != 2:
            continue
        candidates = []
        for index in range(2):
            matmul = producer.get(add.input[index])
            bias = initializers.get(add.input[1 - index])
            if matmul is not None and matmul.op_type == "MatMul" and bias is not None:
                candidates.append((matmul, bias))
        if len(candidates) != 1:
            continue
        matmul, bias_initializer = candidates[0]
        if bias_name_prefixes and not bias_initializer.name.startswith(
            tuple(bias_name_prefixes)
        ):
            continue
        if len(matmul.input) != 2 or not _removed_tensors_are_internal(
            (matmul,), (matmul, add), consumers
        ):
            continue
        if not _plugin_ranks_are_compatible(
            ranks, matmul.input[0], add.output
        ):
            continue
        weight_initializer = initializers.get(matmul.input[1])
        transposed_weight = False
        weight_transpose = None
        if weight_initializer is None:
            weight_transpose = producer.get(matmul.input[1])
            if weight_transpose is None or weight_transpose.op_type != "Transpose":
                continue
            attributes = {
                attribute.name: attribute for attribute in weight_transpose.attribute
            }
            permutation = (
                list(onnx.helper.get_attribute_value(attributes["perm"]))
                if "perm" in attributes
                else None
            )
            if (
                permutation != [1, 0]
                or len(weight_transpose.input) != 1
                or len(weight_transpose.output) != 1
            ):
                continue
            weight_initializer = initializers.get(weight_transpose.input[0])
            if weight_initializer is None:
                continue
            transposed_weight = True
        weight = onnx.numpy_helper.to_array(weight_initializer)
        bias = onnx.numpy_helper.to_array(bias_initializer)
        if (
            weight.ndim != 2
            or weight.dtype != np.float32
            or bias.ndim != 1
            or bias.dtype != np.float32
            or bias.shape[0] != weight.shape[0 if transposed_weight else 1]
        ):
            continue

        if transposed_weight:
            weight_name = weight_initializer.name
            candidate_transpose_keys.add(_node_key(weight_transpose))
        else:
            transposed = np.ascontiguousarray(weight.T)
            if consumers[weight_initializer.name].occurrences == 1:
                weight_initializer.CopyFrom(
                    onnx.numpy_helper.from_array(
                        transposed, name=weight_initializer.name
                    )
                )
                weight_name = weight_initializer.name
            else:
                weight_name = names.allocate(
                    f"{weight_initializer.name}__pnmir_exact_linear_{count}"
                )
                model.graph.initializer.append(
                    onnx.numpy_helper.from_array(transposed, name=weight_name)
                )
        replacements[_node_key(add)] = (
            add,
            matmul,
            weight_name,
            bias_initializer.name,
        )
        removed_keys.add(_node_key(matmul))
        count += 1

    retained_inputs = {
        name
        for node in model.graph.node
        if _node_key(node) not in removed_keys
        for name in node.input
    }
    retained_inputs.update(output.name for output in model.graph.output)
    rewritten = []
    for node in model.graph.node:
        replacement = replacements.get(_node_key(node))
        if replacement is not None:
            add, matmul, weight_name, bias_name = replacement
            rewritten.append(
                onnx.helper.make_node(
                    _EXACT_LINEAR_PLUGIN_NAME,
                    (matmul.input[0], weight_name, bias_name),
                    tuple(add.output),
                    name=f"{add.name}__pnmir_exact_linear",
                    plugin_version=_EXACT_LINEAR_PLUGIN_VERSION,
                    plugin_namespace="",
                )
            )
        elif (
            _node_key(node) not in removed_keys
            and not (
                _node_key(node) in candidate_transpose_keys
                and all(output not in retained_inputs for output in node.output)
            )
        ):
            rewritten.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    return count


def _replace_token_sums(onnx: Any, model: Any) -> int:
    """Route supported token-axis sums through the PyTorch-order kernel."""
    import numpy as np

    dtypes = _static_tensor_dtypes(model)
    initializers = {
        initializer.name: initializer for initializer in model.graph.initializer
    }
    shapes: dict[str, list[int]] = {}
    for value in (*model.graph.input, *model.graph.value_info, *model.graph.output):
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        shape = []
        for dimension in tensor_type.shape.dim:
            if not dimension.HasField("dim_value"):
                shape = []
                break
            shape.append(dimension.dim_value)
        if shape:
            shapes[value.name] = shape

    replacements = 0
    for node in model.graph.node:
        if node.op_type != "ReduceSum" or len(node.input) != 2:
            continue
        axes_initializer = initializers.get(node.input[1])
        if axes_initializer is None:
            continue
        axes = np.asarray(onnx.numpy_helper.to_array(axes_initializer)).reshape(-1)
        attributes = {attribute.name: attribute for attribute in node.attribute}
        keepdims = (
            int(onnx.helper.get_attribute_value(attributes["keepdims"]))
            if "keepdims" in attributes
            else 1
        )
        if axes.tolist() != [1] or keepdims != 0:
            continue
        input_shape = shapes.get(node.input[0])
        if (
            any(
                dtypes.get(name) != onnx.TensorProto.FLOAT
                for name in (node.input[0], *node.output)
            )
            or input_shape is None
            or len(input_shape) != 4
            or input_shape[0] != 1
            or any(dimension <= 0 for dimension in input_shape)
            or (input_shape[2] * input_shape[3]) % 4 != 0
        ):
            continue
        del node.input[1:]
        del node.attribute[:]
        node.op_type = _EXACT_TOKEN_SUM_PLUGIN_NAME
        node.attribute.extend(
            (
                onnx.helper.make_attribute(
                    "plugin_version", _EXACT_TOKEN_SUM_PLUGIN_VERSION
                ),
                onnx.helper.make_attribute("plugin_namespace", ""),
            )
        )
        replacements += 1
    return replacements


def _replace_slice_bmms(onnx: Any, model: Any) -> int:
    """Preserve the strided Transolver slice-projection BMM layout."""
    dtypes = _static_tensor_dtypes(model)
    shapes: dict[str, list[int]] = {}
    for value in (*model.graph.input, *model.graph.value_info, *model.graph.output):
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        shape = []
        for dimension in tensor_type.shape.dim:
            if not dimension.HasField("dim_value"):
                shape = []
                break
            shape.append(dimension.dim_value)
        if shape:
            shapes[value.name] = shape
    producer = {output: node for node in model.graph.node for output in node.output}
    consumers = _tensor_consumers(model)

    def permutation(node: Any) -> list[int] | None:
        if node.op_type != "Transpose":
            return None
        attributes = {attribute.name: attribute for attribute in node.attribute}
        if "perm" not in attributes:
            return None
        return list(onnx.helper.get_attribute_value(attributes["perm"]))

    replacements: dict[_NodeKey, tuple[Any, Any, Any]] = {}
    removed_keys: set[_NodeKey] = set()
    for matmul in model.graph.node:
        if matmul.op_type != "MatMul" or len(matmul.input) != 2:
            continue
        weights_transpose = producer.get(matmul.input[0])
        features_transpose = producer.get(matmul.input[1])
        if weights_transpose is None or features_transpose is None:
            continue
        if permutation(weights_transpose) != [0, 2, 3, 1] or permutation(
            features_transpose
        ) != [0, 2, 1, 3]:
            continue
        if not _removed_tensors_are_internal(
            (weights_transpose, features_transpose),
            (weights_transpose, features_transpose, matmul),
            consumers,
        ):
            continue
        weights_shape = shapes.get(weights_transpose.input[0])
        features_shape = shapes.get(features_transpose.input[0])
        if (
            any(
                dtypes.get(name) != onnx.TensorProto.FLOAT
                for name in (
                    weights_transpose.input[0],
                    features_transpose.input[0],
                    *matmul.output,
                )
            )
            or weights_shape is None
            or features_shape is None
            or len(weights_shape) != 4
            or len(features_shape) != 4
            or weights_shape[0] != 1
            or any(dimension <= 0 for dimension in weights_shape)
            or any(dimension <= 0 for dimension in features_shape)
            or features_shape[:3] != weights_shape[:3]
        ):
            continue
        replacements[_node_key(matmul)] = (
            matmul,
            weights_transpose,
            features_transpose,
        )
        removed_keys.update(
            (_node_key(weights_transpose), _node_key(features_transpose))
        )

    rewritten = []
    for node in model.graph.node:
        replacement = replacements.get(_node_key(node))
        if replacement is not None:
            matmul, weights_transpose, features_transpose = replacement
            rewritten.append(
                onnx.helper.make_node(
                    _EXACT_SLICE_BMM_PLUGIN_NAME,
                    (weights_transpose.input[0], features_transpose.input[0]),
                    tuple(matmul.output),
                    name=f"{matmul.name}__pnmir_exact_slice_bmm",
                    plugin_version=_EXACT_SLICE_BMM_PLUGIN_VERSION,
                    plugin_namespace="",
                )
            )
        elif _node_key(node) not in removed_keys:
            rewritten.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    return len(replacements)


def _replace_layer_norms(onnx: Any, model: Any) -> int:
    """Route last-dimension FP32 LayerNorm through the exact CUDA plugin."""
    shapes = _static_tensor_shapes(model)
    dtypes = _static_tensor_dtypes(model)
    replacements = 0
    for node in model.graph.node:
        if (
            node.op_type != "LayerNormalization"
            or len(node.input) != 3
            or len(node.output) != 1
        ):
            continue
        attributes = {attribute.name: attribute for attribute in node.attribute}
        axis = onnx.helper.get_attribute_value(attributes["axis"]) if "axis" in attributes else -1
        if axis != -1:
            continue
        input_shape = shapes.get(node.input[0])
        gamma_shape = shapes.get(node.input[1])
        beta_shape = shapes.get(node.input[2])
        if (
            any(
                dtypes.get(name) != onnx.TensorProto.FLOAT
                for name in (*node.input, *node.output)
            )
            or input_shape is None
            or len(input_shape) < 2
            or len(input_shape) > _TENSORRT_MAX_DIMS
            or any(dimension <= 0 for dimension in input_shape)
            or input_shape[-1] % 4 != 0
            or gamma_shape != [input_shape[-1]]
            or beta_shape != [input_shape[-1]]
        ):
            continue
        epsilon = (
            float(onnx.helper.get_attribute_value(attributes["epsilon"]))
            if "epsilon" in attributes
            else 1.0e-5
        )
        del node.attribute[:]
        node.op_type = _EXACT_LAYER_NORM_PLUGIN_NAME
        node.attribute.extend(
            (
                onnx.helper.make_attribute("epsilon", epsilon),
                onnx.helper.make_attribute(
                    "plugin_version", _EXACT_LAYER_NORM_PLUGIN_VERSION
                ),
                onnx.helper.make_attribute("plugin_namespace", ""),
            )
        )
        replacements += 1
    return replacements


def _replace_softmaxes(onnx: Any, model: Any) -> int:
    """Route supported last-dimension Softmax through the exact CUDA plugin."""
    shapes = _static_tensor_shapes(model)
    dtypes = _static_tensor_dtypes(model)
    replacements = 0
    for node in model.graph.node:
        if node.op_type != "Softmax" or len(node.input) != 1:
            continue
        attributes = {attribute.name: attribute for attribute in node.attribute}
        axis = (
            onnx.helper.get_attribute_value(attributes["axis"])
            if "axis" in attributes
            else -1
        )
        if axis != -1:
            continue
        input_shape = shapes.get(node.input[0])
        if (
            any(
                dtypes.get(name) != onnx.TensorProto.FLOAT
                for name in (*node.input, *node.output)
            )
            or input_shape is None
            or not 1 <= len(input_shape) <= _TENSORRT_MAX_DIMS
            or any(dimension <= 0 for dimension in input_shape)
            or input_shape[-1] not in (128, 512)
        ):
            continue
        del node.attribute[:]
        node.op_type = _EXACT_SOFTMAX_PLUGIN_NAME
        node.attribute.extend(
            (
                onnx.helper.make_attribute(
                    "plugin_version", _EXACT_SOFTMAX_PLUGIN_VERSION
                ),
                onnx.helper.make_attribute("plugin_namespace", ""),
            )
        )
        replacements += 1
    return replacements


def _replace_attention_subgraphs(onnx: Any, model: Any) -> int:
    """Restore the fixed Transolver SDPA calls from their ONNX decomposition."""
    import numpy as np

    shapes: dict[str, list[int]] = {}
    for value in (*model.graph.input, *model.graph.value_info, *model.graph.output):
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        shape = []
        for dimension in tensor_type.shape.dim:
            if not dimension.HasField("dim_value"):
                shape = []
                break
            shape.append(dimension.dim_value)
        if shape:
            shapes[value.name] = shape

    initializers = {
        initializer.name: initializer for initializer in model.graph.initializer
    }
    producer = {output: node for node in model.graph.node for output in node.output}
    consumers = _tensor_consumers(model)
    names = _OnnxValueNameAllocator.from_model(model)
    def initializer_matches(name: str, expected: Any) -> bool:
        initializer = initializers.get(name)
        if initializer is None:
            return False
        value = onnx.numpy_helper.to_array(initializer)
        return (
            value.dtype == expected.dtype
            and value.shape == expected.shape
            and np.array_equal(value, expected)
        )

    def reshape_target_matches(node: Any, expected: Any) -> bool:
        initializer = initializers.get(node.input[1])
        if initializer is None:
            return False
        target = onnx.numpy_helper.to_array(initializer)
        if target.dtype != expected.dtype or target.shape != expected.shape:
            return False
        if np.array_equal(target, expected):
            return True
        target_dimensions = target.tolist()
        if target_dimensions.count(-1) != 1 or any(
            dimension <= 0 and dimension != -1
            for dimension in target_dimensions
        ):
            return False
        input_shape = shapes.get(node.input[0])
        if input_shape is None or any(dimension <= 0 for dimension in input_shape):
            return False
        input_elements = int(np.prod(input_shape))
        known_elements = int(
            np.prod([dimension for dimension in target_dimensions if dimension != -1])
        )
        if input_elements % known_elements != 0:
            return False
        target_dimensions[target_dimensions.index(-1)] = (
            input_elements // known_elements
        )
        return target_dimensions == expected.tolist() and shapes.get(
            node.output[0]
        ) == expected.tolist()

    def dynamic_mul_input(node: Any) -> tuple[str, Any] | None:
        if node.op_type != "Mul" or len(node.input) != 2:
            return None
        dynamic = [name for name in node.input if name not in initializers]
        constant = [name for name in node.input if name in initializers]
        if len(dynamic) != 1 or len(constant) != 1:
            return None
        scale = onnx.numpy_helper.to_array(initializers[constant[0]])
        if scale.dtype != np.float32 or scale.size != 1:
            return None
        return dynamic[0], np.float32(scale.item())

    replacements: list[tuple[Any, list[Any], Any, Any]] = []
    for output_matmul in model.graph.node:
        if output_matmul.op_type != "MatMul" or len(output_matmul.input) != 2:
            continue
        softmax = producer.get(output_matmul.input[0])
        if softmax is None or softmax.op_type != "Softmax":
            continue
        softmax_attributes = {
            attribute.name: attribute for attribute in softmax.attribute
        }
        axis = (
            onnx.helper.get_attribute_value(softmax_attributes["axis"])
            if "axis" in softmax_attributes
            else -1
        )
        if axis != -1 or len(softmax.input) != 1:
            continue
        scores = producer.get(softmax.input[0])
        if scores is None or scores.op_type != "MatMul" or len(scores.input) != 2:
            continue
        query_scale = producer.get(scores.input[0])
        key_scale = producer.get(scores.input[1])
        if query_scale is None or key_scale is None:
            continue
        query_match = dynamic_mul_input(query_scale)
        key_match = dynamic_mul_input(key_scale)
        if query_match is None or key_match is None:
            continue
        query, query_factor = query_match
        key_transformed, key_factor = key_match
        query_shape = shapes.get(query)
        if (
            query_shape is None
            or len(query_shape) != 4
            or any(dimension <= 0 for dimension in query_shape)
        ):
            continue
        batches, heads, sequence, head_dimension = query_shape
        expected_input_scale = np.float32(head_dimension**-0.25)
        if (
            query_factor != expected_input_scale
            or key_factor != expected_input_scale
        ):
            continue
        expected_key_shape_1 = np.array(
            [batches * heads, sequence, head_dimension], dtype=np.int64
        )
        expected_key_shape_2 = np.array(
            [batches, heads, head_dimension, sequence], dtype=np.int64
        )

        key_reshape_2 = producer.get(key_transformed)
        if (
            key_reshape_2 is None
            or key_reshape_2.op_type != "Reshape"
            or len(key_reshape_2.input) != 2
            or not initializer_matches(
                key_reshape_2.input[1], expected_key_shape_2
            )
        ):
            continue
        key_transpose = producer.get(key_reshape_2.input[0])
        if (
            key_transpose is None
            or key_transpose.op_type != "Transpose"
            or len(key_transpose.input) != 1
        ):
            continue
        transpose_attributes = {
            attribute.name: attribute for attribute in key_transpose.attribute
        }
        if "perm" not in transpose_attributes or list(
            onnx.helper.get_attribute_value(transpose_attributes["perm"])
        ) != [0, 2, 1]:
            continue
        key_reshape_1 = producer.get(key_transpose.input[0])
        if (
            key_reshape_1 is None
            or key_reshape_1.op_type != "Reshape"
            or len(key_reshape_1.input) != 2
            or not reshape_target_matches(key_reshape_1, expected_key_shape_1)
        ):
            continue
        key = key_reshape_1.input[0]
        value = output_matmul.input[1]

        if not all(
            shapes.get(name) == query_shape
            for name in (query, key, value, output_matmul.output[0])
        ):
            continue
        removed = [
            key_reshape_1,
            key_transpose,
            key_reshape_2,
            query_scale,
            key_scale,
            scores,
            softmax,
            output_matmul,
        ]
        if not _removed_tensors_are_internal(removed[:-1], removed, consumers):
            continue
        replacements.append((output_matmul, removed, query, key))

    if not replacements:
        return 0

    replacement_by_output = {_node_key(item[0]): item for item in replacements}
    removed_keys = {
        _node_key(node) for _, nodes, _, _ in replacements for node in nodes
    }
    rewritten = []
    for node in model.graph.node:
        replacement = replacement_by_output.get(_node_key(node))
        if replacement is not None:
            output_matmul, _, query, key = replacement
            value = output_matmul.input[1]
            raw_output = names.allocate(f"{output_matmul.output[0]}__pnmir_bmhd")
            rewritten.append(
                onnx.helper.make_node(
                    _EXACT_ATTENTION_PLUGIN_NAME,
                    (query, key, value),
                    (raw_output,),
                    name=f"{output_matmul.name}__pnmir_exact_attention",
                    plugin_version=_EXACT_ATTENTION_PLUGIN_VERSION,
                    plugin_namespace="",
                )
            )
            rewritten.append(
                onnx.helper.make_node(
                    "Transpose",
                    (raw_output,),
                    tuple(output_matmul.output),
                    name=f"{output_matmul.name}__pnmir_restore_bhmd",
                    perm=(0, 2, 1, 3),
                )
            )
        elif _node_key(node) not in removed_keys:
            rewritten.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    return len(replacements)


def _replace_gelu_subgraphs(
    onnx: Any,
    model: Any,
    *,
    exact_linear_sources_only: bool = False,
) -> int:
    """Restore exact PyTorch GELU calls from their five-node ONNX expansion."""
    import numpy as np

    _validate_node_keys(model)
    initializers = {
        initializer.name: initializer for initializer in model.graph.initializer
    }
    ranks = _tensor_ranks(model)
    consumers: dict[str, int] = {}
    for node in model.graph.node:
        for name in node.input:
            consumers[name] = consumers.get(name, 0) + 1
    for output in model.graph.output:
        consumers[output.name] = consumers.get(output.name, 0) + 1
    producer = {output: node for node in model.graph.node for output in node.output}

    def initializer_is(name: str, expected: np.float32) -> bool:
        value = onnx.numpy_helper.to_array(initializers[name])
        return (
            value.dtype == np.float32
            and value.size == 1
            and value.item() == expected
        )

    replacements: list[tuple[Any, list[Any], str]] = []
    for final_mul in model.graph.node:
        if final_mul.op_type != "Mul" or len(final_mul.input) != 2:
            continue
        candidates = []
        for index in range(2):
            half_mul = producer.get(final_mul.input[index])
            if half_mul is not None and half_mul.op_type == "Mul":
                candidates.append((half_mul, final_mul.input[1 - index]))
        if len(candidates) != 1:
            continue
        half_mul, source = candidates[0]
        if exact_linear_sources_only:
            source_producer = producer.get(source)
            if (
                source_producer is None
                or source_producer.op_type != _EXACT_LINEAR_PLUGIN_NAME
            ):
                continue
        if not _plugin_ranks_are_compatible(
            ranks, source, final_mul.output, minimum_rank=1
        ):
            continue
        half_dynamic = [name for name in half_mul.input if name not in initializers]
        half_constant = [name for name in half_mul.input if name in initializers]
        if (
            len(half_dynamic) != 1
            or len(half_constant) != 1
            or not initializer_is(half_constant[0], np.float32(0.5))
        ):
            continue
        add = producer.get(half_dynamic[0])
        if add is None or add.op_type != "Add":
            continue
        add_dynamic = [name for name in add.input if name not in initializers]
        add_constant = [name for name in add.input if name in initializers]
        if (
            len(add_dynamic) != 1
            or len(add_constant) != 1
            or not initializer_is(add_constant[0], np.float32(1.0))
        ):
            continue
        erf = producer.get(add_dynamic[0])
        if erf is None or erf.op_type != "Erf" or len(erf.input) != 1:
            continue
        divide = producer.get(erf.input[0])
        if (
            divide is None
            or divide.op_type != "Div"
            or len(divide.input) != 2
            or divide.input[0] != source
            or divide.input[1] not in initializers
            or not initializer_is(
                divide.input[1], np.float32(np.sqrt(2.0))
            )
        ):
            continue
        if any(
            consumers.get(name) != 1
            for name in (
                divide.output[0],
                erf.output[0],
                add.output[0],
                half_mul.output[0],
            )
        ):
            continue
        replacements.append((final_mul, [divide, erf, add, half_mul, final_mul], source))

    replacement_by_output = {_node_key(item[0]): item for item in replacements}
    removed_keys = {
        _node_key(node) for _, nodes, _ in replacements for node in nodes
    }
    rewritten = []
    for node in model.graph.node:
        replacement = replacement_by_output.get(_node_key(node))
        if replacement is not None:
            final_mul, _, source = replacement
            rewritten.append(
                onnx.helper.make_node(
                    _EXACT_GELU_PLUGIN_NAME,
                    (source,),
                    tuple(final_mul.output),
                    name=f"{final_mul.name}__pnmir_exact_gelu",
                    plugin_version=_EXACT_GELU_PLUGIN_VERSION,
                    plugin_namespace="",
                )
            )
        elif _node_key(node) not in removed_keys:
            rewritten.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    return len(replacements)


_EXACT_WEIGHTED_BLEND_PLUGIN_NAME = "PNMIRExactWeightedBlend"
_EXACT_WEIGHTED_BLEND_PLUGIN_VERSION = "1"


def _replace_weighted_blends(onnx: Any, model: Any) -> int:
    """Preserve separate FP32 multiply/multiply/add rounding for blends."""
    import numpy as np

    _validate_node_keys(model)
    initializers = {
        initializer.name: initializer for initializer in model.graph.initializer
    }
    producer = {output: node for node in model.graph.node for output in node.output}
    consumers = _tensor_consumers(model)
    shapes = _static_tensor_shapes(model)
    dtypes = _static_tensor_dtypes(model)

    def scaled_input(node: Any) -> tuple[str, str] | None:
        if node is None or node.op_type != "Mul" or len(node.input) != 2:
            return None
        dynamic = [name for name in node.input if name not in initializers]
        constant = [name for name in node.input if name in initializers]
        if len(dynamic) != 1 or len(constant) != 1:
            return None
        scalar = np.asarray(
            onnx.numpy_helper.to_array(initializers[constant[0]])
        )
        if scalar.ndim != 0 or scalar.dtype != np.float32:
            return None
        return dynamic[0], constant[0]

    replacements: dict[_NodeKey, tuple[Any, Any, Any, tuple[str, ...]]] = {}
    removed_keys: set[_NodeKey] = set()
    for add in model.graph.node:
        if add.op_type != "Add" or len(add.input) != 2 or len(add.output) != 1:
            continue
        left_mul = producer.get(add.input[0])
        right_mul = producer.get(add.input[1])
        left = scaled_input(left_mul)
        right = scaled_input(right_mul)
        if left is None or right is None:
            continue
        left_value, left_scale = left
        right_value, right_scale = right
        output_shape = shapes.get(add.output[0])
        if (
            output_shape is None
            or len(output_shape) < 1
            or shapes.get(left_value) != output_shape
            or shapes.get(right_value) != output_shape
            or any(
                dtypes.get(name) != onnx.TensorProto.FLOAT
                for name in (left_value, right_value, add.output[0])
            )
            or not _removed_tensors_are_internal(
                (left_mul, right_mul), (left_mul, right_mul, add), consumers
            )
        ):
            continue
        plugin_inputs = (left_value, left_scale, right_value, right_scale)
        replacements[_node_key(add)] = (
            add,
            left_mul,
            right_mul,
            plugin_inputs,
        )
        removed_keys.update((_node_key(left_mul), _node_key(right_mul)))

    rewritten = []
    for node in model.graph.node:
        replacement = replacements.get(_node_key(node))
        if replacement is not None:
            add, _, _, plugin_inputs = replacement
            rewritten.append(
                onnx.helper.make_node(
                    _EXACT_WEIGHTED_BLEND_PLUGIN_NAME,
                    plugin_inputs,
                    tuple(add.output),
                    name=f"{add.name}__pnmir_exact_weighted_blend",
                    plugin_version=_EXACT_WEIGHTED_BLEND_PLUGIN_VERSION,
                    plugin_namespace="",
                )
            )
        elif _node_key(node) not in removed_keys:
            rewritten.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    return len(replacements)


_EXACT_SCALAR_DIV_PLUGIN_NAME = "PNMIRExactScalarDiv"
_EXACT_SCALAR_DIV_PLUGIN_VERSION = "1"
_EXACT_INVERSE_DISTANCE_BLEND_PLUGIN_NAME = "PNMIRExactInverseDistanceBlend"
_EXACT_INVERSE_DISTANCE_BLEND_PLUGIN_VERSION = "1"


def _replace_scalar_divs(onnx: Any, model: Any) -> int:
    """Preserve the qualified FP32 tensor-by-10 division instruction."""
    import numpy as np

    initializers = {
        initializer.name: initializer for initializer in model.graph.initializer
    }
    ranks = _tensor_ranks(model)
    dtypes = _static_tensor_dtypes(model)
    replacements = 0
    for node in model.graph.node:
        if (
            node.op_type != "Div"
            or len(node.input) != 2
            or not _plugin_ranks_are_compatible(
                ranks, node.input[0], node.output, minimum_rank=1
            )
            or dtypes.get(node.input[0]) != onnx.TensorProto.FLOAT
            or dtypes.get(node.output[0]) != onnx.TensorProto.FLOAT
        ):
            continue
        scalar_initializer = initializers.get(node.input[1])
        if scalar_initializer is None:
            continue
        scalar = onnx.numpy_helper.to_array(scalar_initializer)
        if (
            scalar.ndim != 0
            or scalar.dtype != np.float32
            or scalar.item() != np.float32(10.0)
        ):
            continue
        node.op_type = _EXACT_SCALAR_DIV_PLUGIN_NAME
        node.attribute.extend(
            (
                onnx.helper.make_attribute(
                    "plugin_version", _EXACT_SCALAR_DIV_PLUGIN_VERSION
                ),
                onnx.helper.make_attribute("plugin_namespace", ""),
            )
        )
        replacements += 1
    return replacements


def _replace_inverse_distance_blends(onnx: Any, model: Any) -> int:
    """Fuse PyTorch-order inverse-distance blending into one exact kernel."""
    import numpy as np

    _validate_node_keys(model)
    initializers = {
        initializer.name: initializer for initializer in model.graph.initializer
    }
    producer = {output: node for node in model.graph.node for output in node.output}
    consumers = _tensor_consumers(model)
    shapes = _static_tensor_shapes(model)
    dtypes = _static_tensor_dtypes(model)

    def half_scaled_source(name: str) -> tuple[str, Any] | None:
        node = producer.get(name)
        if node is None or node.op_type != "Mul" or len(node.input) != 2:
            return None
        dynamic = [value for value in node.input if value not in initializers]
        constants = [value for value in node.input if value in initializers]
        if len(dynamic) != 1 or len(constants) != 1:
            return None
        value = onnx.numpy_helper.to_array(initializers[constants[0]])
        if (
            value.dtype != np.float32
            or value.size != 1
            or value.item() != np.float32(0.5)
        ):
            return None
        return dynamic[0], node

    def left_add_terms(name: str) -> tuple[list[str], list[Any]]:
        node = producer.get(name)
        if node is None or node.op_type != "Add" or len(node.input) != 2:
            return [name], []
        terms, nodes = left_add_terms(node.input[0])
        return [*terms, node.input[1]], [*nodes, node]

    candidates: list[tuple[Any, tuple[str, ...], tuple[Any, ...]]] = []
    for final_add in model.graph.node:
        if final_add.op_type != "Add" or len(final_add.input) != 2:
            continue
        parsed = None
        for center_index in range(2):
            center = half_scaled_source(final_add.input[center_index])
            normalized = producer.get(final_add.input[1 - center_index])
            if (
                center is not None
                and normalized is not None
                and normalized.op_type == "Div"
                and len(normalized.input) == 2
            ):
                parsed = (center, normalized)
                break
        if parsed is None:
            continue
        (center_prediction, center_mul), normalized = parsed
        neighbor = half_scaled_source(normalized.input[0])
        if neighbor is None:
            continue
        neighbor_sum, neighbor_half = neighbor
        weighted_terms, neighbor_adds = left_add_terms(neighbor_sum)
        inverse_terms, inverse_adds = left_add_terms(normalized.input[1])
        if len(weighted_terms) != len(inverse_terms) or not weighted_terms:
            continue

        plugin_inputs = [center_prediction]
        fused_nodes = [
            center_mul,
            neighbor_half,
            normalized,
            final_add,
            *neighbor_adds,
            *inverse_adds,
        ]
        valid = True
        for weighted_name, inverse_name in zip(
            weighted_terms, inverse_terms, strict=True
        ):
            reciprocal = producer.get(inverse_name)
            weighted = producer.get(weighted_name)
            if (
                reciprocal is None
                or reciprocal.op_type != "Reciprocal"
                or len(reciprocal.input) != 1
                or weighted is None
                or weighted.op_type != "Mul"
                or len(weighted.input) != 2
                or inverse_name not in weighted.input
            ):
                valid = False
                break
            prediction = next(
                value for value in weighted.input if value != inverse_name
            )
            plugin_inputs.extend((prediction, reciprocal.input[0]))
            fused_nodes.extend((reciprocal, weighted))
        if not valid:
            continue

        output_name = final_add.output[0]
        expected_shape = shapes.get(center_prediction)
        tensor_names = (*plugin_inputs, output_name)
        if (
            expected_shape is None
            or len(expected_shape) < 1
            or len(expected_shape) > _TENSORRT_MAX_DIMS
            or any(dimension <= 0 for dimension in expected_shape)
            or any(shapes.get(name) != expected_shape for name in tensor_names)
            or any(
                dtypes.get(name) != onnx.TensorProto.FLOAT
                for name in tensor_names
            )
        ):
            continue
        candidates.append(
            (final_add, tuple(plugin_inputs), tuple(fused_nodes))
        )

    removed_keys = {
        _node_key(node) for _, _, nodes in candidates for node in nodes
    }
    replacement_outputs = {
        output
        for final_add, _, _ in candidates
        for output in final_add.output
    }
    for _, _, nodes in candidates:
        for node in nodes:
            for output in node.output:
                if output in replacement_outputs:
                    continue
                owners = consumers.get(output, _TensorConsumers()).owners
                if any(
                    owner is not None and owner not in removed_keys
                    for owner in owners
                ) or None in owners:
                    return 0

    replacements = {
        _node_key(final_add): (final_add, plugin_inputs)
        for final_add, plugin_inputs, _ in candidates
    }
    rewritten = []
    for node in model.graph.node:
        replacement = replacements.get(_node_key(node))
        if replacement is not None:
            final_add, plugin_inputs = replacement
            rewritten.append(
                onnx.helper.make_node(
                    _EXACT_INVERSE_DISTANCE_BLEND_PLUGIN_NAME,
                    plugin_inputs,
                    tuple(final_add.output),
                    name=f"{final_add.name}__pnmir_exact_inverse_distance_blend",
                    plugin_version=(
                        _EXACT_INVERSE_DISTANCE_BLEND_PLUGIN_VERSION
                    ),
                    plugin_namespace="",
                )
            )
        elif _node_key(node) not in removed_keys:
            rewritten.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    return len(candidates)
