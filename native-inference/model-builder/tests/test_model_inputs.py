import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pnmir_build.inputs import (
    effective_recipe,
    input_identities,
    resolve_inputs,
    stage_inputs,
    validate_input_spec,
)


class ModelInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.recipe_path = self.root / "recipe.json"
        self.config = self.root / "config.json"
        self.config.write_text('{"offset": 3, "nested": {"enabled": true}}\n')
        self.checkpoint = self.root / "weights.pt"
        self.checkpoint.write_bytes(b"checkpoint bytes: framework-free resolver")
        self.asset = self.root / "normalization.json"
        self.asset.write_text('{"mean": 1}')
        self.recipe = {
            "format_version": 2,
            "adapter": "export.py",
            "config": {"path": "config.json"},
            "checkpoint": {"format": "torch-state-dict", "path": "weights.pt"},
            "assets": {"normalization": {"path": "normalization.json"}},
        }

    def resolve(self, **kwargs):
        return resolve_inputs(self.recipe, self.recipe_path, **kwargs)

    def test_default_files_resolve_without_ml_imports(self):
        torch_was_loaded = "torch" in sys.modules
        result = self.resolve()
        self.assertIsInstance(result, dict, "v2 must resolve actual input files")
        self.assertEqual(result["config_data"]["offset"], 3)
        self.assertEqual(result["checkpoint"]["path"], str(self.checkpoint))
        self.assertEqual(result["checkpoint"]["origin"], "recipe")
        self.assertEqual(
            result["checkpoint"]["sha256"],
            hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
        )
        self.assertEqual("torch" in sys.modules, torch_was_loaded)

    def test_version_one_has_no_inputs_and_rejects_overrides(self):
        self.recipe = {"format_version": 1}
        self.assertIsNone(self.resolve())
        for overrides in (
            {"config": self.config},
            {"checkpoint": self.checkpoint},
            {"checkpoint_sha256": "a" * 64},
            {"assets": {"n": self.asset}},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, "format.*2"):
                    self.resolve(**overrides)

    def test_required_files_and_missing_paths_are_actionable(self):
        for field in ("config", "checkpoint"):
            with self.subTest(field=field):
                recipe = copy.deepcopy(self.recipe)
                del recipe[field]["path"]
                with self.assertRaisesRegex(ValueError, "--" + field):
                    resolve_inputs(recipe, self.recipe_path)
        self.checkpoint.unlink()
        with self.assertRaisesRegex(ValueError, "checkpoint.*regular file"):
            self.resolve()

    def test_hash_pins_fail_and_explicit_override_replaces_default_pin(self):
        self.recipe["checkpoint"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "checkpoint.*SHA-256"):
            self.resolve()
        result = self.resolve(checkpoint=self.checkpoint)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["checkpoint"]["origin"], "cli")
        with self.assertRaisesRegex(ValueError, "checkpoint.*SHA-256"):
            self.resolve(checkpoint=self.checkpoint, checkpoint_sha256="1" * 64)
        with self.assertRaisesRegex(ValueError, "checkpoint.*SHA-256"):
            self.resolve(checkpoint_sha256="not-a-digest")

    def test_external_overrides_replace_config_and_assets(self):
        external = self.root / "external"
        external.mkdir()
        config = external / "new.json"
        config.write_text('{"different": 9}')
        asset = external / "new-data.txt"
        asset.write_text("replacement")
        result = self.resolve(config=config, assets={"normalization": asset})
        self.assertIsInstance(result, dict)
        self.assertEqual(result["config_data"], {"different": 9})
        self.assertEqual(result["assets"]["normalization"]["path"], str(asset))
        with self.assertRaisesRegex(ValueError, "Unknown asset"):
            self.resolve(assets={"typo": asset})

    def test_cli_digest_alone_does_not_replace_a_pinned_default_checkpoint(self):
        self.recipe["checkpoint"]["sha256"] = "0" * 64
        actual_digest = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "checkpoint.*SHA-256"):
            self.resolve(checkpoint_sha256=actual_digest)
        selected = self.resolve(
            checkpoint=self.checkpoint, checkpoint_sha256=actual_digest
        )
        self.assertEqual(selected["checkpoint"]["sha256"], actual_digest)

    def test_strict_json_object_configuration(self):
        for content in (
            "[]",
            '{"x":NaN}',
            '{"x":Infinity}',
            '{"x":1e999}',
            '{"x":1,"x":2}',
            '{"nested":{"x":1,"x":2}}',
            "{",
        ):
            with self.subTest(content=content):
                self.config.write_text(content)
                with self.assertRaisesRegex(ValueError, "config"):
                    self.resolve()

    def test_descriptors_are_explicit_and_paths_contained(self):
        for field, value in (
            ("config", None),
            ("config", {"path": "../outside.json"}),
            ("config", {"path": str(self.config)}),
            ("config", {"path": "config.json", "sh256": "a" * 64}),
            ("checkpoint", {"format": "pickle", "path": "weights.pt"}),
            ("assets", {"../escape": {"path": "normalization.json"}}),
            ("assets", {"normalization": {}}),
            ("adapter", "model-inputs/export.py"),
        ):
            with self.subTest(field=field, value=value):
                recipe = copy.deepcopy(self.recipe)
                recipe[field] = value
                with self.assertRaises(ValueError):
                    validate_input_spec(recipe)

    def test_default_and_override_symlinks_and_directories_rejected(self):
        link = self.root / "linked.pt"
        link.symlink_to(self.checkpoint)
        self.recipe["checkpoint"]["path"] = link.name
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.resolve()
        self.recipe["checkpoint"]["path"] = self.checkpoint.name
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.resolve(checkpoint=link)
        directory = self.root / "data"
        directory.mkdir()
        with self.assertRaisesRegex(ValueError, "regular file"):
            self.resolve(checkpoint=directory)
        parent_link = self.root / "parent-link"
        parent_link.symlink_to(self.root, target_is_directory=True)
        self.recipe["checkpoint"]["path"] = "parent-link/weights.pt"
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.resolve()

    def test_staging_retains_bytes_and_replayable_recipe(self):
        resolved = self.resolve()
        self.assertIsInstance(resolved, dict)
        destination = self.root / "retained"
        staged = stage_inputs(resolved, destination)
        self.assertTrue((destination / "model-inputs/checkpoint.pt").is_file())
        self.assertEqual(input_identities(staged), input_identities(resolved))
        canonical = Path(staged["effective_config"]["path"])
        self.assertEqual(json.loads(canonical.read_text()), resolved["config_data"])
        replay = effective_recipe(self.recipe, staged, destination)
        self.assertEqual(replay["checkpoint"]["path"], "model-inputs/checkpoint.pt")
        self.assertEqual(
            replay["checkpoint"]["sha256"], resolved["checkpoint"]["sha256"]
        )
        self.config.unlink()
        self.checkpoint.unlink()
        self.asset.unlink()
        rerun = resolve_inputs(replay, destination / "effective-recipe.json")
        self.assertEqual(input_identities(rerun), input_identities(resolved))
        encoded = json.dumps(input_identities(staged))
        self.assertNotIn(str(self.root), encoded)
        self.assertNotIn("origin", encoded)

    def test_explicit_path_can_use_parent_alias_like_macos_tmp(self):
        alias = self.root / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        try:
            result = self.resolve(checkpoint=alias / "weights.pt")
        except ValueError as error:
            self.fail(
                f"Explicit file selections must normalize parent aliases: {error}"
            )
        self.assertEqual(result["checkpoint"]["path"], str(self.checkpoint))

    def test_staging_detects_input_change_and_preserves_existing_directory(self):
        resolved = self.resolve()
        self.assertIsInstance(resolved, dict)
        self.checkpoint.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed|integrity"):
            stage_inputs(resolved, self.root / "changed-stage")
        destination = self.root / "existing"
        (destination / "model-inputs").mkdir(parents=True)
        sentinel = destination / "model-inputs/config.json"
        sentinel.write_text("preserve")
        with self.assertRaises((ValueError, FileExistsError)):
            stage_inputs(self.resolve(), destination)
        self.assertEqual(sentinel.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
