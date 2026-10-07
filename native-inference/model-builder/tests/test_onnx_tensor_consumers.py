"""Consumer ownership checks without optional ONNX dependencies."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_builder.export.tensorrt_exact_graphs import _tensor_consumers


def _node(inputs, outputs, graphs=(), *, graph_list=False):
    attributes = [
        SimpleNamespace(
            g=None if graph_list else graph,
            graphs=graphs if graph_list else (),
            HasField=lambda name: name == "g" and not graph_list,
        )
        for graph in (graphs[:1] if graph_list else graphs)
    ]
    return SimpleNamespace(input=inputs, output=outputs, attribute=attributes)


def _graph(nodes, *, inputs=(), outputs=(), initializers=(), sparse=()):
    return SimpleNamespace(
        node=nodes,
        input=[SimpleNamespace(name=name) for name in inputs],
        output=[SimpleNamespace(name=name) for name in outputs],
        initializer=[SimpleNamespace(name=name) for name in initializers],
        sparse_initializer=[
            SimpleNamespace(values=SimpleNamespace(name=name)) for name in sparse
        ],
        value_info=[],
    )


class TensorConsumerTests(unittest.TestCase):
    def test_nested_capture_belongs_to_top_level_control_flow_node(self):
        branch = _graph([_node(("projected", "weight"), ("branch_result",))])
        body = _graph([_node(("condition",), ("nested_result",), (branch,))])
        model = SimpleNamespace(
            graph=_graph(
                [
                    _node(("input", "weight"), ("projected",)),
                    _node(("projected", "bias"), ("output",)),
                    _node(("trip_count", "condition"), ("loop_result",), (body,)),
                ]
            )
        )

        consumers = _tensor_consumers(model)

        self.assertEqual(consumers["projected"].owners, {("output",), ("loop_result",)})
        self.assertEqual(consumers["weight"].occurrences, 2)
        self.assertNotIn("branch_result", consumers)

    def test_local_bindings_shadow_outer_names_in_nested_captures(self):
        for binding in ("input", "initializer", "sparse_initializer", "node_output"):
            with self.subTest(binding=binding):
                branch = _graph([_node(("weight",), ("branch_result",))])
                body = _graph([_node(("condition",), ("nested_result",), (branch,))])
                if binding == "node_output":
                    body.node.insert(0, _node(("source",), ("weight",)))
                elif binding == "sparse_initializer":
                    body.sparse_initializer.append(
                        SimpleNamespace(values=SimpleNamespace(name="weight"))
                    )
                else:
                    getattr(body, binding).append(SimpleNamespace(name="weight"))
                model = SimpleNamespace(
                    graph=_graph(
                        [
                            _node(("weight", "weight"), ("projected",)),
                            _node(("condition",), ("control_result",), (body,)),
                        ]
                    )
                )

                consumers = _tensor_consumers(model)

                self.assertEqual(consumers["weight"].owners, {("projected",)})
                self.assertEqual(consumers["weight"].occurrences, 2)

    def test_graph_list_captures_include_outputs_but_value_info_does_not_bind(self):
        branch = _graph([], outputs=("weight",))
        branch.value_info.append(SimpleNamespace(name="weight"))
        model = SimpleNamespace(
            graph=_graph(
                [
                    _node(("weight",), ("projected",)),
                    _node((), ("control_result",), (branch,), graph_list=True),
                ]
            )
        )

        consumers = _tensor_consumers(model)

        self.assertEqual(
            consumers["weight"].owners, {("projected",), ("control_result",)}
        )
        self.assertEqual(consumers["weight"].occurrences, 2)


if __name__ == "__main__":
    unittest.main()
