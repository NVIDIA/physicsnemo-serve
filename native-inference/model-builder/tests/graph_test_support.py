"""Shared ONNX graph fixtures; importing test suites does not collect other suites."""

from __future__ import annotations

import numpy as np
import onnx
from onnx import helper, numpy_helper


class _RematerializingNodes:
    def __init__(self, nodes: object) -> None:
        self._nodes = nodes

    def __iter__(self):
        return (
            onnx.NodeProto.FromString(node.SerializeToString()) for node in self._nodes
        )

    def __delitem__(self, index: object) -> None:
        del self._nodes[index]

    def extend(self, nodes: object) -> None:
        self._nodes.extend(nodes)


class _RematerializingGraph:
    def __init__(self, graph: onnx.GraphProto) -> None:
        self._graph = graph

    @property
    def node(self) -> _RematerializingNodes:
        return _RematerializingNodes(self._graph.node)

    def __getattr__(self, name: str) -> object:
        return getattr(self._graph, name)


class _RematerializingModel:
    def __init__(self, model: onnx.ModelProto) -> None:
        self.graph = _RematerializingGraph(model.graph)


def _with_rematerialized_nodes(
    model: onnx.ModelProto,
) -> _RematerializingModel:
    rematerialized = _RematerializingModel(model)
    first = list(rematerialized.graph.node)
    second = list(rematerialized.graph.node)
    assert {id(node) for node in first}.isdisjoint(id(node) for node in second)
    return rematerialized


def _reserve_value_name(model: onnx.ModelProto, name: str, namespace: str) -> None:
    source = model.graph.input[0]

    def value_info() -> onnx.ValueInfoProto:
        value = onnx.ValueInfoProto()
        value.CopyFrom(source)
        value.name = name
        return value

    if namespace == "input":
        model.graph.input.append(value_info())
    elif namespace == "output":
        model.graph.input.append(value_info())
        model.graph.output.append(value_info())
    elif namespace == "value_info":
        model.graph.node.append(helper.make_node("Identity", (source.name,), (name,)))
        model.graph.value_info.append(value_info())
    elif namespace == "initializer":
        model.graph.initializer.append(
            numpy_helper.from_array(np.array(0.0, dtype=np.float32), name=name)
        )
    elif namespace == "node_output":
        model.graph.node.append(helper.make_node("Identity", (source.name,), (name,)))
    else:
        raise AssertionError(f"unsupported namespace: {namespace}")


def _check_model_with_tensorrt_plugins(model: onnx.ModelProto) -> None:
    checkable = onnx.ModelProto.FromString(model.SerializeToString())
    plugin_domain = "pnmir.test"
    for node in checkable.graph.node:
        if node.op_type.startswith("PNMIRExact"):
            node.domain = plugin_domain
    checkable.opset_import.append(helper.make_opsetid(plugin_domain, 1))
    onnx.checker.check_model(checkable, full_check=True)
