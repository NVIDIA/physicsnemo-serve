"""Structural tests with a tiny upstream API double, not CFD parity evidence."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock
import zipfile

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parent / "legacy"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = load_module("geotransolver_adapter_test", ROOT / "adapter.py")
PREPARATION = ROOT / "prepare_geotransolver.py"
preparation = load_module("geotransolver_preparation_test", PREPARATION)


if torch is not None:

    class GlobalTokenizer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(2, 2)

        def forward(self, value):
            return self.projection(value).reshape(1, 1, 1, 2).expand(1, 2, 3, 2)

    class Context(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.global_tokenizer = GlobalTokenizer()
            self.features = torch.nn.Linear(3, 2)

        def build_context(self, embeddings, positions, geometry, global_embedding):
            local = self.features(positions[0])
            static = (geometry.mean() + positions[0].mean()).expand(1, 2, 3, 4)
            context = torch.cat(
                (static, self.global_tokenizer(global_embedding)), dim=-1
            )
            return context, (local,), None

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.1))

        def forward(self, streams, context):
            return (streams[0] + self.weight * context.mean(),)

    class GeoTransolverDouble(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self._args = {
                "__name__": "GeoTransolver",
                "__module__": "physicsnemo.experimental.models.geotransolver.geotransolver",
                "__args__": copy.deepcopy(kwargs),
            }
            self.preprocess = torch.nn.ModuleList([torch.nn.Linear(6, 4)])
            self.context_builder = Context()
            self.blocks = torch.nn.ModuleList([Block()])
            self.ln_mlp_out = torch.nn.ModuleList([torch.nn.Linear(6, 4)])

        def forward(self, local_embedding, local_positions, global_embedding, geometry):
            context, features, _ = self.context_builder.build_context(
                (local_embedding,), (local_positions,), geometry, global_embedding
            )
            streams = (
                torch.cat((self.preprocess[0](local_embedding), features[0]), dim=-1),
            )
            for block in self.blocks:
                streams = block(streams, context)
            return self.ln_mlp_out[0](streams[0])


@unittest.skipIf(torch is None, "requires producer Torch for structural tests")
class GeoTransolverExampleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.checkpoint = self.root / "GeoTransolver.0.501.mdlus"
        self.checkpoint.write_bytes(
            b"explicit trusted checkpoint fixture; loader is a test double"
        )
        self.output = self.root / "prepared"
        self.model_args = {
            "functional_dim": 6,
            "out_dim": 4,
            "geometry_dim": 3,
            "global_dim": 2,
            "n_layers": 20,
            "n_hidden": 256,
            "dropout": 0.0,
            "n_head": 8,
            "act": "gelu",
            "mlp_ratio": 2,
            "slice_num": 128,
            "use_te": False,
            "plus": False,
            "include_local_features": True,
            "radii": [0.01, 0.05, 0.25, 1.0, 2.5, 5.0],
            "neighbors_in_radius": [4, 8, 16, 64, 128, 256],
            "n_hidden_local": 32,
        }
        torch.manual_seed(17)
        self.official = GeoTransolverDouble(**self.model_args).eval()
        with zipfile.ZipFile(self.checkpoint, "w") as archive:
            archive.writestr("args.json", json.dumps(self.official._args))
            archive.writestr(
                "metadata.json",
                json.dumps(
                    {"physicsnemo_version": "2.2.0a0", "mdlus_file_version": "1.0"}
                ),
            )
            archive.writestr("model.pt", b"trusted loader test double")
        package = ModuleType("physicsnemo")
        package.Module = mock.Mock()
        package.Module.from_checkpoint.return_value = self.official
        self.loader = package.Module.from_checkpoint
        experimental = ModuleType("physicsnemo.experimental")
        models = ModuleType("physicsnemo.experimental.models")
        geotransolver = ModuleType("physicsnemo.experimental.models.geotransolver")
        geotransolver.GeoTransolver = GeoTransolverDouble
        self.addCleanup(mock.patch.stopall)
        mock.patch.dict(
            sys.modules,
            {
                "physicsnemo": package,
                "physicsnemo.experimental": experimental,
                "physicsnemo.experimental.models": models,
                "physicsnemo.experimental.models.geotransolver": geotransolver,
            },
        ).start()
        mock.patch("importlib.metadata.version", return_value="2.1.1").start()

    def run_preparation(self):
        result = preparation.prepare(
            self.checkpoint, self.output, points=8, geometry_points=16, device="cpu"
        )
        self.assertEqual(
            result.get("status"), "complete", "preparation must produce verified inputs"
        )
        return result

    def prepared_inputs(self):
        self.run_preparation()
        config = json.loads((self.output / "config.json").read_text())
        assets = {"fixtures": self.output / "fixtures.pt"}
        return config, assets

    def test_generated_recipe_requires_explicit_exact_aoti_profile(self):
        self.run_preparation()
        recipe = json.loads((self.output / "recipe.json").read_text())
        self.assertEqual(recipe.get("aoti_profile"), "aten-boundary-exact-v2")
        self.assertEqual(recipe["supported_backends"], ["aoti", "tensorrt"])

    def test_state_identity_is_sensitive_to_names_shapes_dtype_and_values(self):
        value = torch.tensor([1.0, 2.0])
        expected = adapter.state_sha256({"weight": value})
        self.assertRegex(expected, r"^[0-9a-f]{64}$")
        for state in (
            {"other": value},
            {"weight": value.reshape(1, 2)},
            {"weight": value.double()},
            {"weight": value + 1},
        ):
            self.assertNotEqual(adapter.state_sha256(state), expected)
        self.assertEqual(
            adapter.state_sha256({"a": value, "b": value + 1}),
            adapter.state_sha256({"b": value + 1, "a": value}),
        )

    def test_archive_model_version_and_constructor_are_retained_separately_from_builder(
        self,
    ):
        report = self.run_preparation()
        self.assertEqual(
            report.get("checkpoint_metadata"),
            {"physicsnemo_version": "2.2.0a0", "mdlus_file_version": "1.0"},
        )
        self.assertEqual(report["environment"]["physicsnemo"], "2.1.1")
        self.assertEqual(report.get("checkpoint_constructor"), self.official._args)

    def test_preparation_retains_real_constructor_state_and_three_verified_cases(self):
        report = self.run_preparation()
        self.loader.assert_called_once_with(str(self.checkpoint), strict=True)
        config = json.loads((self.output / "config.json").read_text())
        self.assertEqual(config["model_args"], self.model_args)
        state = torch.load(
            self.output / "checkpoint.pt", weights_only=True, map_location="cpu"
        )
        self.assertEqual(set(state), set(self.official.state_dict()))
        for name, value in state.items():
            self.assertTrue(torch.equal(value, self.official.state_dict()[name]))
        fixtures = torch.load(
            self.output / "fixtures.pt", weights_only=True, map_location="cpu"
        )
        self.assertEqual(len(fixtures["cases"]), 3)
        self.assertFalse(
            torch.equal(
                fixtures["cases"][0]["expected"], fixtures["cases"][1]["expected"]
            )
        )
        self.assertFalse(
            torch.equal(
                fixtures["cases"][0]["expected"], fixtures["cases"][2]["expected"]
            )
        )
        self.assertTrue(all(case["passed"] for case in report["comparisons"]))
        self.assertEqual(
            report["checkpoint"]["sha256"],
            hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
        )
        recipe = json.loads((self.output / "recipe.json").read_text())
        self.assertEqual(recipe["format_version"], 2)
        self.assertEqual(
            [entry["name"] for entry in recipe["inputs"]],
            ["local_embedding", "local_features", "static_context", "global_embedding"],
        )
        self.assertEqual(
            [entry["shape"] for entry in recipe["inputs"]],
            [[1, 8, 6], [1, 8, 2], [1, 2, 3, 4], [1, 1, 2]],
        )
        self.assertEqual(
            recipe["outputs"],
            [
                {
                    "name": "surface_fields_standardized",
                    "dtype": "float32",
                    "shape": [1, 8, 4],
                }
            ],
        )
        for field in ("config", "checkpoint"):
            path = self.output / recipe[field]["path"]
            self.assertEqual(
                recipe[field]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
            )
        self.assertNotIn("input_names", recipe)

    def test_adapter_preserves_upstream_keys_and_replays_full_model_references(self):
        config, assets = self.prepared_inputs()
        model = adapter.create_model(config, assets)
        self.assertIsInstance(model, GeoTransolverDouble)
        self.assertEqual(set(model.state_dict()), set(self.official.state_dict()))
        model.load_state_dict(self.official.state_dict(), strict=True)
        self.assertFalse(model._load_state_dict_post_hooks)
        cases = adapter.create_cases(config, assets)
        fixtures = torch.load(assets["fixtures"], weights_only=True, map_location="cpu")
        self.assertEqual(len(cases), 3)
        for case, fixture in zip(cases, fixtures["cases"], strict=True):
            self.assertIsInstance(case, tuple)
            torch.testing.assert_close(
                model(*case), fixture["expected"], rtol=0, atol=0
            )

    def test_changed_checkpoint_is_rejected_before_export(self):
        config, assets = self.prepared_inputs()
        model = adapter.create_model(config, assets)
        self.assertIsInstance(model, torch.nn.Module)
        changed = copy.deepcopy(self.official.state_dict())
        first = next(iter(changed))
        changed[first] = changed[first] + 1
        with self.assertRaisesRegex(ValueError, "checkpoint|state.*fixture"):
            model.load_state_dict(changed, strict=True)

    def test_changed_configuration_is_rejected(self):
        config, assets = self.prepared_inputs()
        config["model_args"]["dropout"] = 0.25
        with self.assertRaisesRegex(ValueError, "config"):
            adapter.create_model(config, assets)
        with self.assertRaisesRegex(ValueError, "config"):
            adapter.create_cases(config, assets)

    def test_existing_output_and_missing_checkpoint_fail_without_loading(self):
        self.output.mkdir()
        sentinel = self.output / "keep"
        sentinel.write_text("preserve")
        with self.assertRaises((ValueError, FileExistsError)):
            preparation.prepare(self.checkpoint, self.output, device="cpu")
        self.assertEqual(sentinel.read_text(), "preserve")
        self.checkpoint.unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            preparation.prepare(self.checkpoint, self.root / "missing", device="cpu")
        self.loader.assert_not_called()

    def test_failed_full_model_comparison_does_not_emit_a_recipe(self):
        original = self.official.forward
        self.official.forward = lambda *args, **kwargs: original(*args, **kwargs) + 1
        with self.assertRaisesRegex(
            (ValueError, AssertionError), "parity|match|differ"
        ):
            preparation.prepare(
                self.checkpoint, self.output, points=8, geometry_points=16, device="cpu"
            )
        self.assertFalse((self.output / "recipe.json").exists())
        receipt = json.loads((self.output / "preparation.json").read_text())
        self.assertEqual(receipt["status"], "failed")

    def test_adapter_used_for_parity_and_emitted_recipe_comes_from_one_snapshot(self):
        source = self.root / "producer"
        source.mkdir()
        for name in ("prepare.py", "adapter.py"):
            shutil.copyfile(
                PREPARATION if name == "prepare.py" else ROOT / name, source / name
            )
        producer = load_module("geotransolver_mutable_producer", source / "prepare.py")
        adapter_path = source / "adapter.py"
        original_bytes = adapter_path.read_bytes()
        changed_bytes = original_bytes.replace(
            b"return self.ln_mlp_out[0](streams[0])",
            b"return self.ln_mlp_out[0](streams[0]) + 100",
        )
        self.assertNotEqual(original_bytes, changed_bytes)
        forward = self.official.forward

        def change_adapter(*args, **kwargs):
            adapter_path.write_bytes(changed_bytes)
            return forward(*args, **kwargs)

        self.official.forward = change_adapter
        report = producer.prepare(
            self.checkpoint,
            self.output,
            points=8,
            geometry_points=16,
            device="cpu",
            adapter_path=adapter_path,
        )
        self.assertEqual(report["status"], "complete")
        self.assertEqual(adapter_path.read_bytes(), changed_bytes)
        emitted = self.output / "adapter.py"
        self.assertEqual(emitted.read_bytes(), original_bytes)
        self.assertEqual(
            report["source"]["adapter"]["sha256"],
            hashlib.sha256(emitted.read_bytes()).hexdigest(),
        )
        emitted_adapter = load_module("geotransolver_emitted_adapter", emitted)
        config = json.loads((self.output / "config.json").read_text())
        assets = {"fixtures": self.output / "fixtures.pt"}
        model = emitted_adapter.create_model(config, assets)
        model.load_state_dict(self.official.state_dict(), strict=True)
        fixtures = torch.load(assets["fixtures"], weights_only=True, map_location="cpu")
        for case in fixtures["cases"]:
            torch.testing.assert_close(
                model(*case["inputs"]), case["expected"], rtol=0, atol=0
            )

    def test_changed_preparation_source_cannot_publish_complete_provenance(self):
        source = self.root / "producer"
        source.mkdir()
        for name in ("prepare.py", "adapter.py"):
            shutil.copyfile(
                PREPARATION if name == "prepare.py" else ROOT / name, source / name
            )
        source_path = source / "prepare.py"
        producer = load_module("geotransolver_changed_preparation", source_path)
        forward = self.official.forward

        def change_source(*args, **kwargs):
            source_path.write_text(
                source_path.read_text() + "\n# concurrently edited\n"
            )
            return forward(*args, **kwargs)

        self.official.forward = change_source
        with self.assertRaisesRegex(ValueError, "preparation source.*changed"):
            producer.prepare(
                self.checkpoint,
                self.output,
                points=8,
                geometry_points=16,
                device="cpu",
                adapter_path=source / "adapter.py",
            )
        self.assertFalse((self.output / "recipe.json").exists())
        receipt = json.loads((self.output / "preparation.json").read_text())
        self.assertEqual(receipt["status"], "failed")


if __name__ == "__main__":
    unittest.main()
