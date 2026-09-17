"""Named, independently shaped tensors for real model stages."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pnmir_build.tensors import tensor_contracts, tensor_names
from pnmir_build import worker
from pnmir_build import cli
import test_model_input_cli


class TensorContractTests(unittest.TestCase):
    def setUp(self):
        self.recipe = {
            "format_version": 2,
            "inputs": [
                {"name": "local", "dtype": "float32", "shape": [1, 3, 2]},
                {"name": "context", "dtype": "float32", "shape": [1, 2, 4]},
            ],
            "outputs": [{"name": "fields", "dtype": "float32", "shape": [1, 3, 1]}],
        }

    def test_independent_shapes_preserve_names_and_order(self):
        inputs, outputs = tensor_contracts(self.recipe)
        self.assertEqual(inputs, self.recipe["inputs"])
        self.assertEqual(outputs, self.recipe["outputs"])
        self.assertEqual(tensor_names(self.recipe, "inputs"), ["local", "context"])
        self.assertEqual(tensor_names(self.recipe, "outputs"), ["fields"])

    def test_legacy_inputs_keep_shared_shape_and_inferred_output_shapes(self):
        for version in (1, 2):
            with self.subTest(version=version):
                recipe = {
                    "format_version": version,
                    "input_names": ["x", "y"],
                    "output_names": ["z"],
                    "dtype": "float32",
                    "shape": [4],
                }
                inputs, outputs = tensor_contracts(recipe)
                self.assertEqual(
                    inputs,
                    [
                        {"name": name, "dtype": "float32", "shape": [4]}
                        for name in ("x", "y")
                    ],
                )
                self.assertEqual(
                    outputs, [{"name": "z", "dtype": "float32", "shape": None}]
                )

    def test_invalid_and_ambiguous_contracts_are_rejected(self):
        invalid = []
        for key, value in (
            ("format_version", 1),
            ("inputs", []),
            ("outputs", []),
            ("input_names", ["local"]),
            ("shape", [4]),
            ("dtype", "float32"),
        ):
            invalid.append(dict(self.recipe, **{key: value}))
        for field in ("inputs", "outputs"):
            for key, value in (
                ("name", "../bad"),
                ("dtype", "int32"),
                ("shape", [-1]),
                ("shape", [True]),
                ("shape", []),
                ("extra", 1),
            ):
                recipe = copy.deepcopy(self.recipe)
                recipe[field][0][key] = value
                invalid.append(recipe)
            recipe = copy.deepcopy(self.recipe)
            recipe[field].append(recipe[field][0].copy())
            invalid.append(recipe)
        recipe = copy.deepcopy(self.recipe)
        del recipe["outputs"]
        invalid.append(recipe)
        for recipe in invalid:
            with self.subTest(recipe=recipe):
                with self.assertRaises(ValueError):
                    tensor_contracts(recipe)

    def test_worker_and_framework_free_doctor_accept_named_shapes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            recipe = dict(
                self.recipe,
                name="named-shapes",
                version="1",
                adapter="export.py",
                factory="create_model",
                cases="create_cases",
                config={"path": "config.json"},
                checkpoint={"format": "torch-state-dict", "path": "weights.pt"},
                supported_backends=["aoti"],
                default_backend="aoti",
            )
            path = root / "recipe.json"
            path.write_text(json.dumps(recipe))
            (root / "export.py").write_text(
                "raise AssertionError('doctor must not import adapter')\n"
            )
            (root / "config.json").write_text("{}")
            (root / "weights.pt").write_bytes(b"doctor must not deserialize weights")
            try:
                parsed, _ = worker._read_recipe(path, ["aoti"])
            except ValueError as error:
                self.fail(f"Worker must accept distinct static shapes: {error}")
            self.assertEqual(parsed, recipe)
            command = [
                sys.executable,
                "-S",
                str(Path(__file__).resolve().parents[2] / "physicsnemo-model-builder"),
                "doctor",
                "--recipe",
                str(path),
                "--executor",
                "local",
                "--device",
                "cpu",
                "--runtime",
                sys.executable,
                "--output",
                str(root / "output"),
            ]
            result = subprocess.run(command, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((root / "output").exists())
            recipe["outputs"][0]["shape"] = [-1]
            path.write_text(json.dumps(recipe))
            invalid = subprocess.run(
                command, capture_output=True, text=True, timeout=15
            )
            self.assertEqual(invalid.returncode, 2, invalid.stderr)
            self.assertIn("shape", invalid.stderr)
            self.assertFalse((root / "output").exists())

    def test_frontend_rejects_invalid_declared_shape_without_export(self):
        fixture = test_model_input_cli.ModelInputCliTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        recipe = dict(fixture.recipe)
        recipe.pop("input_names")
        recipe.pop("output_names")
        recipe.update(inputs=self.recipe["inputs"], outputs=self.recipe["outputs"])
        recipe["outputs"][0]["shape"] = [-1]
        fixture.recipe_path.write_text(json.dumps(recipe))
        with self.assertRaisesRegex(cli.UsageError, "shape"):
            cli.read_recipe(fixture.recipe_path)

    def test_container_completion_honors_named_tensor_contract(self):
        fixture = test_model_input_cli.ModelInputCliTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        plan, *_ = fixture.completed_v2()
        recipe = plan["recipe"]
        for key in ("input_names", "output_names", "dtype", "shape"):
            recipe.pop(key)
        recipe["inputs"] = [{"name": "input", "dtype": "float32", "shape": [4]}]
        recipe["outputs"] = [{"name": "output", "dtype": "float32", "shape": [4]}]
        try:
            cli._validate_container_completion(plan)
        except RuntimeError as error:
            self.fail(f"Named contracts must accept matching native results: {error}")
        for field in ("inputs", "outputs"):
            recipe[field][0]["shape"] = [2, 2]
            with self.subTest(field=field):
                with self.assertRaisesRegex(RuntimeError, "tensor contract"):
                    cli._validate_container_completion(plan)
            recipe[field][0]["shape"] = [4]


if __name__ == "__main__":
    unittest.main()
