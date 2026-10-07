"""Real CPU model checks for contracts inferred from customer validation cases."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

try:
    import torch
except ImportError:
    torch = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_builder.build import worker
from model_builder.build.inputs import resolve_inputs


ADAPTER = """import torch

class Projection(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor([99.0]))
        assert self.scale.device.type == 'cpu'
        self.config = config
        self.calls = 0

    def forward(self, coordinates, coefficients):
        assert not self.training
        if self.config.get('forbid_eager'):
            raise AssertionError('invalid input reached eager inference')
        self.calls += 1
        value = coordinates @ coefficients * self.scale
        mode = self.config.get('output')
        if mode == 'shape' and self.calls == 2:
            value = value.squeeze(-1)
        if mode == 'dtype':
            value = value.double()
        if mode == 'nonfinite':
            value = value * float('nan')
        if mode == 'scalar':
            value = value.sum()
        if mode == 'zero':
            value = value[:0]
        if mode == 'empty':
            return ()
        if mode == 'dict':
            return {'value': value}
        if mode == 'count' and self.calls == 2:
            return value
        return value, value.sum().reshape(1)


def create_model(config, assets):
    return Projection(config)


def create_cases(config, assets):
    coordinates = torch.tensor([[1., 2.], [3., 4.]])
    coefficients = torch.tensor([[1.], [2.]])
    mode = config.get('input')
    if mode == 'dtype':
        coordinates = coordinates.double()
    if mode == 'integer':
        coordinates = coordinates.long()
    if mode == 'scalar':
        coordinates = coordinates.sum()
    if mode == 'zero':
        coordinates = coordinates[:0]
    if mode == 'nonfinite':
        coordinates = coordinates * float('nan')
    if mode == 'nontensor':
        coordinates = [[1., 2.], [3., 4.]]
    first = (coordinates, coefficients)
    if mode == 'empty':
        return []
    if mode == 'empty-tuple':
        return [()]
    if mode == 'list':
        return [[coordinates, coefficients]]
    if mode == 'nontensor':
        return [first]
    if mode == 'count':
        return [first, (coordinates,)]
    if mode == 'shape':
        return [first, (coordinates[:1], coefficients)]
    return [first, (coordinates + 1., coefficients)]
"""


@unittest.skipIf(torch is None, "requires the producer Torch environment")
class AuthoringWorkerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.recipe_path = self.root / "recipe.json"
        self.config_path = self.root / "config.json"
        self.config_path.write_text("{}")
        self.checkpoint_path = self.root / "checkpoint.pt"
        torch.save({"scale": torch.tensor([3.0])}, self.checkpoint_path)
        (self.root / "export.py").write_text(ADAPTER)
        self.recipe = {
            "format_version": 2,
            "name": "customer-projection",
            "version": "0.1.0",
            "adapter": "export.py",
            "factory": "create_model",
            "cases": "create_cases",
            "config": {"path": "config.json"},
            "checkpoint": {"format": "torch-state-dict", "path": "checkpoint.pt"},
        }
        self.recipe_path.write_text(json.dumps(self.recipe))

    def prepare(self, **options):
        return worker._prepare_model(
            self.recipe,
            self.recipe_path,
            "cpu",
            model_inputs=resolve_inputs(self.recipe, self.recipe_path),
            infer_contract=True,
            **options,
        )

    def test_shape_free_adapter_infers_inputs_and_outputs_from_loaded_model(self):
        original = copy.deepcopy(self.recipe)
        try:
            prepared = self.prepare()
        except ValueError as error:
            self.fail(f"valid shape-free model preparation was rejected: {error}")
        self.assertEqual(
            prepared["tensor_contract"],
            {
                "inputs": [
                    {"name": "input_0", "dtype": "float32", "shape": [2, 2]},
                    {"name": "input_1", "dtype": "float32", "shape": [2, 1]},
                ],
                "outputs": [
                    {"name": "output_0", "dtype": "float32", "shape": [2, 1]},
                    {"name": "output_1", "dtype": "float32", "shape": [1]},
                ],
            },
        )
        self.assertEqual(len(prepared["cases"]), 2)
        self.assertEqual(
            struct.unpack("=2f", prepared["references"][0][0]["data"]), (15.0, 33.0)
        )
        self.assertEqual(
            struct.unpack("=2f", prepared["references"][1][0]["data"]), (24.0, 42.0)
        )
        self.assertEqual(self.recipe, original)

    def test_optional_names_propagate_to_records_and_contract(self):
        self.recipe.update(
            input_names=["coordinates", "coefficients"],
            output_names=["prediction", "total"],
        )
        prepared = self.prepare()
        for key in ("inputs", "outputs"):
            expected = self.recipe["input_names" if key == "inputs" else "output_names"]
            self.assertEqual(
                [item["name"] for item in prepared["tensor_contract"][key]], expected
            )
            records = prepared["inputs" if key == "inputs" else "references"]
            self.assertEqual([item["name"] for item in records[0]], expected)

    def test_checkpoint_is_loaded_once_with_cpu_weights_only(self):
        with mock.patch.object(torch, "load", wraps=torch.load) as load:
            prepared = self.prepare()
        self.assertEqual(load.call_count, 1)
        self.assertEqual(
            load.call_args.kwargs, {"weights_only": True, "map_location": "cpu"}
        )
        self.assertEqual(prepared["model"].scale.device.type, "cpu")

    def test_strict_checkpoint_validation_precedes_eager_contract_inference(self):
        self.config_path.write_text('{"forbid_eager":true}')
        for issue, state in (
            ("keys", {}),
            ("keys", {"scale": torch.tensor([3.0]), "extra": torch.tensor([1.0])}),
            ("shape", {"scale": torch.ones(2)}),
            ("dtype", {"scale": torch.ones(1, dtype=torch.float64)}),
            ("nonfinite", {"scale": torch.tensor([float("inf")])}),
            ("plain", {"state_dict": {"scale": torch.tensor([3.0])}}),
        ):
            with self.subTest(issue=issue):
                torch.save(state, self.checkpoint_path)
                with self.assertRaisesRegex(ValueError, issue):
                    self.prepare()

    def test_invalid_input_contracts_are_rejected_before_eager_execution(self):
        for mode, expected in (
            ("dtype", "dtype"),
            ("integer", "dtype"),
            ("scalar", "positive shape"),
            ("zero", "positive shape"),
            ("nontensor", "tensor"),
            ("empty", "at least one case"),
            ("empty-tuple", "non-empty|input count"),
            ("list", "tuples"),
        ):
            with self.subTest(mode=mode):
                self.config_path.write_text(
                    json.dumps({"input": mode, "forbid_eager": True})
                )
                with self.assertRaisesRegex(ValueError, expected):
                    self.prepare()

    def test_later_inputs_must_match_first_case_shape_and_count(self):
        for mode in ("shape", "count"):
            with self.subTest(mode=mode):
                self.config_path.write_text(json.dumps({"input": mode}))
                with self.assertRaisesRegex(ValueError, mode):
                    self.prepare()

    def test_outputs_must_be_static_nonempty_positive_fp32_tensors(self):
        for mode, expected in (
            ("shape", "shape"),
            ("count", "count"),
            ("dtype", "dtype"),
            ("scalar", "positive shape"),
            ("zero", "positive shape"),
            ("empty", "non-empty"),
            ("dict", "outputs.*tensor"),
        ):
            with self.subTest(mode=mode):
                self.config_path.write_text(json.dumps({"output": mode}))
                with self.assertRaisesRegex((ValueError, TypeError), expected):
                    self.prepare()

    def test_nonfinite_inputs_and_outputs_remain_rejected(self):
        for kind in ("input", "output"):
            with self.subTest(kind=kind):
                self.config_path.write_text(json.dumps({kind: "nonfinite"}))
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    self.prepare()

    def test_optional_names_must_be_safe_unique_and_match_actual_tensor_count(self):
        for key in ("input_names", "output_names"):
            for names in ([], ["one"], ["same", "same"], ["bad/name", "other"], "ab"):
                with self.subTest(key=key, names=names):
                    self.recipe.pop("input_names", None)
                    self.recipe.pop("output_names", None)
                    self.recipe[key] = names
                    with self.assertRaisesRegex(ValueError, "names|count"):
                        self.prepare()

    def test_explicit_recipes_still_require_declared_shapes_by_default(self):
        self.recipe.update(
            input_names=["x", "y"], output_names=["z", "total"], dtype="float32"
        )
        with self.assertRaisesRegex(ValueError, "shape"):
            worker._prepare_model(
                self.recipe,
                self.recipe_path,
                "cpu",
                model_inputs=resolve_inputs(self.recipe, self.recipe_path),
            )


if __name__ == "__main__":
    unittest.main()
