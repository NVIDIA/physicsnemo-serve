"""Real CPU Torch contract checks; compiler/native execution are separate lanes."""

from __future__ import annotations

import hashlib
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


ADAPTER = """import json
from pathlib import Path
import torch

class Configured(torch.nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor([99.0], dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.tensor([98.0], dtype=torch.float32))
        self.offset = offset
    def forward(self, value):
        return value * self.scale + self.bias + self.offset

def create_model(config, assets):
    assert isinstance(assets['normalization'], Path)
    extra = json.loads(assets.pop('normalization').read_text())['offset']
    result = Configured(config['offset'] + extra)
    config['nested']['value'] = 999
    return result

def create_cases(config, assets):
    assert config['nested']['value'] == 1, 'factory mutated cases configuration'
    assert assets['normalization'].is_file(), 'factory mutated cases assets'
    return [(torch.tensor([0., 1., 2., 3.]),), (torch.tensor([-1., 2., -3., 4.]),)]
"""


def descriptor(path, origin="recipe"):
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
        "origin": origin,
    }


@unittest.skipIf(torch is None, "requires the producer Torch environment")
class ModelInputWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.recipe_path = self.root / "recipe.json"
        self.config_path = self.root / "config.json"
        self.checkpoint_path = self.root / "checkpoint.pt"
        self.asset_path = self.root / "normalization.json"
        self.config_path.write_text(json.dumps({"offset": 3.0, "nested": {"value": 1}}))
        self.asset_path.write_text('{"offset":0.5}')
        self.write_checkpoint(2.0, 1.0)
        self.recipe = {
            "format_version": 2,
            "name": "configured-affine",
            "version": "0.1.0",
            "adapter": "export.py",
            "factory": "create_model",
            "cases": "create_cases",
            "input_names": ["input"],
            "output_names": ["output"],
            "dtype": "float32",
            "shape": [4],
            "supported_backends": ["aoti"],
            "default_backend": "aoti",
            "config": {"path": "config.json"},
            "checkpoint": {"format": "torch-state-dict", "path": "checkpoint.pt"},
            "assets": {"normalization": {"path": "normalization.json"}},
        }
        self.recipe_path.write_text(json.dumps(self.recipe))
        (self.root / "export.py").write_text(ADAPTER)
        self.runtime = self.root / "physicsnemo-infer"
        self.runtime.write_text("#!/bin/sh\nexit 0\n")
        self.runtime.chmod(0o755)
        self.output = self.root / "build"
        self.compiled_references = []

    def write_checkpoint(self, scale, bias):
        torch.save(
            {
                "scale": torch.tensor([scale], dtype=torch.float32),
                "bias": torch.tensor([bias], dtype=torch.float32),
            },
            self.checkpoint_path,
        )

    def resolved(self):
        return {
            "config": descriptor(self.config_path),
            "checkpoint": descriptor(self.checkpoint_path),
            "assets": {"normalization": descriptor(self.asset_path)},
            "config_data": json.loads(self.config_path.read_text()),
            "effective_config": {"sha256": "f" * 64, "size_bytes": 0},
        }

    def prepare(self):
        return worker._prepare_model(
            self.recipe, self.recipe_path, "cpu", model_inputs=self.resolved()
        )

    def fake_backend(self, backend, prepared, recipe, device, package, exported):
        self.compiled_references.append(prepared["references"])
        package.mkdir(parents=True)
        exported.mkdir(parents=True)
        (package / "model.pt2").write_bytes(
            prepared["weights"]["state_sha256"].encode()
        )
        (package / "model.json").write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "artifacts": [{"backend": backend, "path": "model.pt2"}],
                }
            )
        )
        (exported / "program.pt2").write_bytes(
            b"compiler substitute for worker-only unit checks"
        )
        return {"format": "torch.export.ExportedProgram", "entrypoint": "program.pt2"}

    def fake_native(
        self, runtime, package, backend, device, inputs, references, case_dir, log
    ):
        return {
            "passed": True,
            "outputs": [
                {"name": value["name"], "max_abs": 0.0, "relative_l2": 0.0}
                for value in references
            ],
        }

    def build(self, *, model_inputs=None):
        with (
            mock.patch.object(worker, "_build_backend", side_effect=self.fake_backend),
            mock.patch.object(worker, "_native_case", side_effect=self.fake_native),
        ):
            return worker.execute_build(
                self.recipe_path,
                self.output,
                ["aoti"],
                "cpu",
                self.runtime,
                model_inputs=model_inputs,
            )

    def test_v2_config_assets_and_checkpoint_change_eager_outputs(self):
        prepared = self.prepare()
        # The factory mutates its copy; backend plugins still need the resolved assets.
        self.assertEqual(prepared["assets"], {"normalization": self.asset_path})
        self.assertEqual(
            struct.unpack("=4f", prepared["references"][0][0]["data"]),
            (4.5, 6.5, 8.5, 10.5),
        )
        first_state = prepared["weights"]["state_sha256"]
        self.write_checkpoint(-1.0, 0.5)
        self.config_path.write_text(
            json.dumps({"offset": -2.0, "nested": {"value": 1}})
        )
        changed = self.prepare()
        self.assertEqual(
            struct.unpack("=4f", changed["references"][0][0]["data"]),
            (-1.0, -2.0, -3.0, -4.0),
        )
        self.assertNotEqual(first_state, changed["weights"]["state_sha256"])
        self.assertEqual(changed["model"].scale.device.type, "cpu")

    def test_checkpoint_loading_is_weights_only_and_cpu(self):
        with mock.patch.object(torch, "load", wraps=torch.load) as load:
            self.prepare()
        self.assertEqual(load.call_count, 1)
        self.assertIs(load.call_args.kwargs["weights_only"], True)
        self.assertEqual(load.call_args.kwargs["map_location"], "cpu")

    def test_inplace_model_preserves_inputs_for_each_validation_case(self):
        adapter = """import torch
SOURCE = torch.tensor([1., 2., 3., 4.])
class Mutating(torch.nn.Module):
    def forward(self, value):
        return value.add_(1)
def create_model():
    model = Mutating()
    model.source = SOURCE
    return model
def create_cases():
    return [(SOURCE,), (SOURCE,)]
"""
        (self.root / "export.py").write_text(adapter)
        recipe = {
            key: value
            for key, value in self.recipe.items()
            if key not in {"config", "checkpoint", "assets"}
        }
        recipe["format_version"] = 1
        for infer_contract in (False, True):
            with self.subTest(infer_contract=infer_contract):
                prepared = worker._prepare_model(
                    recipe, self.recipe_path, "cpu", infer_contract=infer_contract
                )
                self.assertEqual(len(prepared["cases"]), 2)
                for case, inputs, references in zip(
                    prepared["cases"],
                    prepared["inputs"],
                    prepared["references"],
                    strict=True,
                ):
                    self.assertEqual(
                        struct.unpack("=4f", inputs[0]["data"]), (1.0, 2.0, 3.0, 4.0)
                    )
                    self.assertEqual(case[0].tolist(), [1.0, 2.0, 3.0, 4.0])
                    expected = struct.unpack("=4f", references[0]["data"])
                    self.assertEqual(expected, (2.0, 3.0, 4.0, 5.0))
                    self.assertEqual(
                        prepared["model"](case[0].clone()).tolist(), list(expected)
                    )
                self.assertEqual(
                    prepared["model"].source.tolist(), [1.0, 2.0, 3.0, 4.0]
                )

    def test_strict_state_errors_fail_before_backend_execution(self):
        valid = {"scale": torch.tensor([2.0]), "bias": torch.tensor([1.0])}
        invalid = {
            "keys-missing": {"scale": torch.tensor([2.0])},
            "keys-extra": {**valid, "unexpected": torch.tensor([0.0])},
            "shape": {**valid, "scale": torch.tensor([2.0, 3.0])},
            "dtype": {**valid, "scale": torch.tensor([2.0], dtype=torch.float64)},
            "finite": {**valid, "scale": torch.tensor([float("nan")])},
            "plain": {"state_dict": valid},
        }
        for issue, state in invalid.items():
            with self.subTest(issue=issue):
                torch.save(state, self.checkpoint_path)
                self.output = self.root / ("build-" + issue)
                with mock.patch.object(worker, "_build_backend") as backend:
                    with self.assertRaisesRegex(
                        (ValueError, RuntimeError), issue.split("-")[0]
                    ):
                        worker.execute_build(
                            self.recipe_path, self.output, ["aoti"], "cpu", self.runtime
                        )
                    backend.assert_not_called()
                if self.output.exists():
                    self.assertEqual(
                        json.loads((self.output / "build.json").read_text())["status"],
                        "failed",
                    )

    def test_original_and_effective_source_and_input_identities_are_retained(self):
        original_recipe = self.recipe_path.read_bytes()
        expected_config = json.loads(self.config_path.read_text())
        original_checkpoint = self.checkpoint_path.read_bytes()
        report = self.build()
        self.assertEqual(report["status"], "complete")
        self.assertEqual(
            (self.output / "source/recipe.json").read_bytes(), original_recipe
        )
        effective = json.loads(
            (self.output / "source/effective-recipe.json").read_text()
        )
        for value in [
            effective["config"],
            effective["checkpoint"],
            *effective["assets"].values(),
        ]:
            relative = Path(value["path"])
            self.assertFalse(relative.is_absolute())
            self.assertTrue(value["path"].startswith("model-inputs/"))
            retained = self.output / "source" / relative
            self.assertTrue(retained.is_file())
            self.assertEqual(
                hashlib.sha256(retained.read_bytes()).hexdigest(), value["sha256"]
            )
        self.assertEqual(
            json.loads(
                (self.output / "source/model-inputs/effective-config.json").read_text()
            ),
            expected_config,
        )
        for original in [
            self.recipe_path,
            self.config_path,
            self.checkpoint_path,
            self.asset_path,
            self.root / "export.py",
        ]:
            original.unlink()
        self.assertEqual(
            (self.output / "source" / effective["checkpoint"]["path"]).read_bytes(),
            original_checkpoint,
        )
        release = json.loads((self.output / "model/model-release.json").read_text())
        self.assertEqual(release["model_inputs"], report["model_inputs"])
        self.assertEqual(
            report["model_inputs"]["checkpoint"]["sha256"],
            hashlib.sha256(original_checkpoint).hexdigest(),
        )

        def check_no_paths(value):
            if isinstance(value, dict):
                self.assertNotIn("path", value)
                self.assertNotIn("origin", value)
                for child in value.values():
                    check_no_paths(child)

        check_no_paths(release["model_inputs"])
        self.assertFalse(list((self.output / "model").rglob("*.py")))
        self.assertEqual(len(self.compiled_references), 1)

    def test_explicit_resolved_overrides_are_used_and_recorded(self):
        from model_builder.build.inputs import resolve_inputs

        replacement = self.root / "override.json"
        replacement.write_text(json.dumps({"offset": 8.0, "nested": {"value": 1}}))
        override = resolve_inputs(self.recipe, self.recipe_path, config=replacement)
        report = self.build(model_inputs=override)
        actual = struct.unpack("=4f", self.compiled_references[0][0][0]["data"])
        self.assertEqual(actual, (9.5, 11.5, 13.5, 15.5))
        self.assertEqual(
            report["model_inputs"]["config"]["sha256"],
            hashlib.sha256(replacement.read_bytes()).hexdigest(),
        )
        self.assertEqual(report["model_input_sources"]["config"]["origin"], "cli")

    def test_v1_cannot_silently_ignore_explicit_model_inputs(self):
        self.recipe["format_version"] = 1
        for key in ("config", "checkpoint", "assets"):
            self.recipe.pop(key)
        self.recipe_path.write_text(json.dumps(self.recipe))
        with self.assertRaisesRegex(ValueError, "format.1|model.inputs"):
            self.build(model_inputs=self.resolved())
        self.assertFalse(self.output.exists())

    def test_adapter_cannot_silently_modify_retained_inputs(self):
        mutating = ADAPTER.replace(
            "extra = json.loads(assets.pop('normalization').read_text())['offset']",
            "retained_asset = assets.pop('normalization')\n"
            "    extra = json.loads(retained_asset.read_text())['offset']\n"
            "    retained_asset.write_text('{\"offset\":100.0}')",
        )
        (self.root / "export.py").write_text(mutating)
        with (
            mock.patch.object(
                worker, "_build_backend", side_effect=self.fake_backend
            ) as backend,
            mock.patch.object(worker, "_native_case", side_effect=self.fake_native),
        ):
            with self.assertRaisesRegex(
                ValueError, "retained.*changed|source.*changed"
            ):
                worker.execute_build(
                    self.recipe_path, self.output, ["aoti"], "cpu", self.runtime
                )
            backend.assert_not_called()
        receipt = json.loads((self.output / "build.json").read_text())
        self.assertEqual(receipt["status"], "failed")
        self.assertFalse((self.output / "model/model-release.json").exists())

    def test_adapter_cannot_occupy_reserved_model_inputs_directory(self):
        adapter = self.root / "model-inputs" / "export.py"
        adapter.parent.mkdir()
        adapter.write_text(ADAPTER)
        self.recipe["adapter"] = "model-inputs/export.py"
        self.recipe_path.write_text(json.dumps(self.recipe))
        with self.assertRaisesRegex(ValueError, "reserved|model-inputs"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_export_cannot_publish_if_it_modifies_retained_inputs(self):
        def mutating_export(*args):
            result = self.fake_backend(*args)
            asset = (
                self.output
                / "source/model-inputs/assets/normalization/normalization.json"
            )
            asset.write_text('{"offset":100.0}')
            return result

        with (
            mock.patch.object(worker, "_build_backend", side_effect=mutating_export),
            mock.patch.object(worker, "_native_case", side_effect=self.fake_native),
        ):
            with self.assertRaisesRegex(
                ValueError, "retained.*changed|source.*changed"
            ):
                worker.execute_build(
                    self.recipe_path, self.output, ["aoti"], "cpu", self.runtime
                )
        receipt = json.loads((self.output / "build.json").read_text())
        self.assertEqual(receipt["status"], "failed")
        self.assertFalse((self.output / "model/model-release.json").exists())


if __name__ == "__main__":
    unittest.main()
