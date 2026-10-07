"""Real CPU Torch checks for independently declared static tensor contracts."""

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
        self.scale = torch.nn.Parameter(torch.tensor([1.0]))
        self.config = config
    def forward(self, coordinates, coefficients, bias):
        if self.config.get('forbid_eager'):
            raise AssertionError('invalid input reached eager inference')
        output = coordinates @ coefficients + bias * self.scale
        mode = self.config.get('output')
        if mode == 'shape':
            return output.squeeze(-1)
        if mode == 'dtype':
            return output.to(torch.float64)
        if mode == 'count':
            return output, output
        return output

def create_model(config, assets):
    return Projection(config)

def create_cases(config, assets):
    coordinates = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    coefficients = torch.tensor([[1.], [2.], [3.]])
    bias = torch.tensor([0.5])
    mode = config.get('input')
    if mode == 'shape':
        coefficients = coefficients.reshape(1, 3)
    if mode == 'dtype':
        coefficients = coefficients.to(torch.float64)
    if mode == 'count':
        return [(coordinates, coefficients)]
    return [(coordinates, coefficients, bias),
            (coordinates + 1., coefficients * 2., bias - 0.5)]
"""


@unittest.skipIf(torch is None, "requires the producer Torch environment")
class TensorWorkerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.recipe_path = self.root / "recipe.json"
        self.config_path = self.root / "config.json"
        self.config_path.write_text("{}")
        (self.root / "export.py").write_text(ADAPTER)
        torch.save({"scale": torch.tensor([2.0])}, self.root / "checkpoint.pt")
        self.recipe = {
            "format_version": 2,
            "name": "projection",
            "version": "0.1.0",
            "adapter": "export.py",
            "factory": "create_model",
            "cases": "create_cases",
            "inputs": [
                {"name": "coordinates", "dtype": "float32", "shape": [2, 3]},
                {"name": "coefficients", "dtype": "float32", "shape": [3, 1]},
                {"name": "bias", "dtype": "float32", "shape": [1]},
            ],
            "outputs": [{"name": "prediction", "dtype": "float32", "shape": [2, 1]}],
            "supported_backends": ["aoti"],
            "default_backend": "aoti",
            "config": {"path": "config.json"},
            "checkpoint": {"format": "torch-state-dict", "path": "checkpoint.pt"},
        }
        self.recipe_path.write_text(json.dumps(self.recipe))
        self.runtime = self.root / "physicsnemo-infer"
        self.runtime.write_text("#!/bin/sh\nexit 0\n")
        self.runtime.chmod(0o755)

    def prepare(self):
        return worker._prepare_model(
            self.recipe,
            self.recipe_path,
            "cpu",
            model_inputs=resolve_inputs(self.recipe, self.recipe_path),
        )

    def test_distinct_shapes_and_named_output_are_preserved(self):
        original = copy.deepcopy(self.recipe)
        try:
            prepared = self.prepare()
        except (ValueError, KeyError) as error:
            self.fail(f"valid named tensor recipe was rejected: {error}")
        self.assertEqual(self.recipe, original)
        self.assertEqual(
            [item["shape"] for item in prepared["inputs"][0]], [[2, 3], [3, 1], [1]]
        )
        self.assertEqual(
            [item["name"] for item in prepared["inputs"][0]],
            ["coordinates", "coefficients", "bias"],
        )
        reference = prepared["references"][0][0]
        self.assertEqual(reference["name"], "prediction")
        self.assertEqual(reference["shape"], [2, 1])
        self.assertEqual(struct.unpack("=2f", reference["data"]), (15.0, 33.0))
        self.assertEqual(
            struct.unpack("=2f", prepared["references"][1][0]["data"]), (40.0, 76.0)
        )

    def test_input_errors_are_rejected_before_eager(self):
        for problem in ("shape", "dtype", "count"):
            with self.subTest(problem=problem):
                self.config_path.write_text(
                    json.dumps({"input": problem, "forbid_eager": True})
                )
                with self.assertRaisesRegex(ValueError, problem):
                    self.prepare()

    def test_output_errors_prevent_backend_execution(self):
        for problem in ("shape", "dtype", "count"):
            with self.subTest(problem=problem):
                self.config_path.write_text(json.dumps({"output": problem}))
                output = self.root / f"bad-{problem}"
                with mock.patch.object(worker, "_build_backend") as backend:
                    with self.assertRaisesRegex(ValueError, problem):
                        worker.execute_build(
                            self.recipe_path, output, ["aoti"], "cpu", self.runtime
                        )
                    backend.assert_not_called()
                receipt = json.loads((output / "build.json").read_text())
                self.assertEqual(receipt["status"], "failed")
                self.assertFalse((output / "model/model-release.json").exists())

    def test_export_uses_declared_input_and_output_names(self):
        from model_builder.export import exporter

        prepared = {
            "model": torch.nn.Identity(),
            "cases": [(torch.ones(2, 3), torch.ones(3, 1), torch.ones(1))],
        }
        with mock.patch.object(exporter, "export_package") as export:
            try:
                worker._build_backend(
                    "aoti",
                    prepared,
                    self.recipe,
                    "cpu",
                    self.root / "package",
                    self.root / "exported",
                )
            except KeyError as error:
                self.fail(f"export could not derive named tensor arguments: {error}")
        self.assertEqual(
            export.call_args.kwargs["input_names"],
            ("coordinates", "coefficients", "bias"),
        )
        self.assertEqual(export.call_args.kwargs["output_names"], ("prediction",))

    def test_completed_named_build_retains_exact_recipe_and_native_contract(self):
        output = self.root / "build"
        original = self.recipe_path.read_bytes()

        def fake_export(backend, prepared, recipe, device, package, exported):
            package.mkdir(parents=True)
            exported.mkdir(parents=True)
            (package / "model.pt2").write_bytes(b"worker test compiler substitute")
            (package / "model.json").write_text("{}")
            (exported / "program.pt2").write_bytes(b"worker test graph substitute")
            return {
                "format": "torch.export.ExportedProgram",
                "entrypoint": "program.pt2",
            }

        def fake_native(
            runtime, package, backend, device, inputs, references, case_dir, log
        ):
            self.assertEqual(
                [value["shape"] for value in inputs], [[2, 3], [3, 1], [1]]
            )
            self.assertEqual(references[0]["shape"], [2, 1])
            self.assertEqual(references[0]["name"], "prediction")
            return {"passed": True, "outputs": []}

        with (
            mock.patch.object(worker, "_build_backend", side_effect=fake_export),
            mock.patch.object(
                worker, "_native_case", side_effect=fake_native
            ) as native,
        ):
            try:
                receipt = worker.execute_build(
                    self.recipe_path, output, ["aoti"], "cpu", self.runtime
                )
            except ValueError as error:
                self.fail(f"valid named build was rejected: {error}")
        self.assertEqual(receipt["status"], "complete")
        self.assertEqual(native.call_count, 2)
        self.assertEqual((output / "source/recipe.json").read_bytes(), original)
        effective = json.loads((output / "source/effective-recipe.json").read_text())
        self.assertEqual(effective["inputs"], self.recipe["inputs"])
        self.assertEqual(effective["outputs"], self.recipe["outputs"])
        self.assertNotIn("input_names", effective)
        self.assertNotIn("output_names", effective)

    def test_legacy_float64_input_is_rejected_before_eager(self):
        self.recipe.pop("inputs")
        self.recipe.pop("outputs")
        self.recipe.update(
            input_names=["coordinates", "coefficients", "bias"],
            output_names=["prediction"],
            dtype="float32",
            shape=[2, 3],
        )
        legacy_adapter = ADAPTER.replace(
            "coefficients = torch.tensor([[1.], [2.], [3.]])",
            "coefficients = torch.ones(2, 3)",
        ).replace("bias = torch.tensor([0.5])", "bias = torch.ones(2, 3)")
        (self.root / "export.py").write_text(legacy_adapter)
        self.config_path.write_text(
            json.dumps({"input": "dtype", "forbid_eager": True})
        )
        with self.assertRaisesRegex(ValueError, "dtype"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
