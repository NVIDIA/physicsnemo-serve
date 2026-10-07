"""Project authoring contracts remain independent of ML and execution."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_builder.build.projects import apply_project


class ProjectConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.path = self.root / "model-build.json"

    def write(self, **fields):
        document = {"format_version": 1, "model": "affine", **fields}
        self.path.write_text(json.dumps(document))
        return document

    def args(self, **changes):
        values = dict(
            command="build",
            model=str(self.root),
            profile=None,
            recipe=None,
            checkpoint=None,
            checkpoint_sha256=None,
            config=None,
            asset=[],
            backend=[],
            executor=None,
            device=None,
            runtime=None,
            builder_image=None,
            lock=None,
            output=None,
            required_gpu_arch=None,
        )
        return argparse.Namespace(**(values | changes))

    def test_named_model_without_project_remains_unchanged(self):
        args = self.args(model="affine")
        before = vars(args).copy()
        self.assertIsNone(apply_project(args))
        self.assertEqual(vars(args), before)

    def test_directory_resolves_project_and_records_exact_source_identity(self):
        self.write(checkpoint="weights/trained.pt", device="cpu")
        args = self.args()
        metadata = apply_project(args)
        self.assertIsNotNone(metadata)
        self.assertEqual(args.model, "affine")
        self.assertEqual(args.checkpoint, self.root / "weights/trained.pt")
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.executor, "container")
        self.assertEqual(metadata["path"], self.path)
        self.assertIsNone(metadata["profile"])
        source = self.path.read_bytes()
        self.assertEqual(
            metadata["source_identity"],
            {"sha256": hashlib.sha256(source).hexdigest(), "size_bytes": len(source)},
        )
        self.assertEqual(
            metadata["effective"]["checkpoint"],
            str(self.root / "weights/trained.pt"),
        )

    def test_explicit_project_file_supports_custom_recipe(self):
        self.path.write_text(
            json.dumps({"format_version": 1, "recipe": "custom/recipe.json"})
        )
        args = self.args(model=str(self.path))
        metadata = apply_project(args)
        self.assertIsNotNone(metadata)
        self.assertIsNone(args.model)
        self.assertEqual(args.recipe, self.root / "custom/recipe.json")

    def test_profile_and_cli_precedence_replaces_backend_list(self):
        self.write(
            backends=["aoti"],
            executor="container",
            device="cuda",
            profiles={
                "gpu": {"backends": ["aoti", "tensorrt"], "device": "cuda:1"},
                "cpu": {"executor": "local", "device": "cpu"},
            },
            default_profile="gpu",
        )
        args = self.args(backend=["tensorrt"], device="cuda:2")
        metadata = apply_project(args)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata["profile"], "gpu")
        self.assertEqual(args.backend, ["tensorrt"])
        self.assertEqual(args.device, "cuda:2")
        selected = self.args(profile="cpu")
        self.assertEqual(apply_project(selected)["profile"], "cpu")
        self.assertEqual(selected.executor, "local")
        self.assertEqual(selected.device, "cpu")
        self.assertEqual(selected.backend, ["aoti"])

    def test_paths_use_project_location_and_cli_paths_use_current_directory(self):
        self.write(
            config="configuration/model.json",
            runtime="sdk/physicsnemo-infer",
            toolchain_lock="toolchain.json",
            profiles={"local": {"runtime": "alternative/physicsnemo-infer"}},
            default_profile="local",
        )
        args = self.args(checkpoint=Path("cli.pt"), lock=Path("cli-lock.json"))
        metadata = apply_project(args)
        self.assertIsNotNone(metadata)
        self.assertEqual(args.config, self.root / "configuration/model.json")
        self.assertEqual(args.runtime, self.root / "alternative/physicsnemo-infer")
        self.assertEqual(args.checkpoint, Path("cli.pt").absolute())
        self.assertEqual(args.lock, Path("cli-lock.json").absolute())
        self.assertEqual(metadata["effective"]["toolchain_lock"], str(args.lock))

    def test_cli_asset_overrides_merge_by_name_and_retain_path_identity(self):
        self.write(assets={"normalization": "norm.json", "geometry": "geom.bin"})
        args = self.args(asset=["geometry=external.bin", "extra=extra.bin"])
        metadata = apply_project(args)
        self.assertIsNotNone(metadata)
        self.assertEqual(
            dict(item.split("=", 1) for item in args.asset),
            {
                "normalization": str(self.root / "norm.json"),
                "geometry": str(Path("external.bin").absolute()),
                "extra": str(Path("extra.bin").absolute()),
            },
        )

    def test_output_root_generates_distinct_fresh_paths_without_writes(self):
        self.write(output_root="builds")
        before = sorted(self.root.rglob("*"))
        first, second = self.args(), self.args()
        metadata = apply_project(first)
        self.assertIsNotNone(metadata)
        apply_project(second)
        self.assertNotEqual(first.output, second.output)
        self.assertEqual(first.output.parent, self.root / "builds")
        self.assertFalse(first.output.exists())
        self.assertEqual(sorted(self.root.rglob("*")), before)
        self.assertEqual(
            metadata["effective"]["output_root"], str(self.root / "builds")
        )
        self.assertNotIn("output", metadata["effective"])
        explicit = self.args(output=Path("selected-output"))
        apply_project(explicit)
        self.assertEqual(explicit.output, Path("selected-output").absolute())

    def test_leaf_symlink_is_preserved_for_downstream_input_validation(self):
        target = self.root / "target.pt"
        target.write_bytes(b"checkpoint")
        checkpoint = self.root / "weights.pt"
        checkpoint.symlink_to(target)
        self.write(checkpoint="weights.pt")
        args = self.args()
        metadata = apply_project(args)
        self.assertIsNotNone(metadata)
        self.assertEqual(args.checkpoint, checkpoint)
        self.assertTrue(args.checkpoint.is_symlink())

    def test_paths_normalize_parent_components_but_keep_symlink_leaf(self):
        (self.root / "nested").mkdir()
        target = self.root / "target.pt"
        target.write_bytes(b"checkpoint")
        checkpoint = self.root / "weights.pt"
        checkpoint.symlink_to(target)
        self.write(
            checkpoint="nested/../weights.pt",
            runtime="nested/../physicsnemo-infer",
            output_root="nested/..",
        )
        args = self.args()
        metadata = apply_project(args)
        self.assertEqual(args.checkpoint, checkpoint)
        self.assertTrue(args.checkpoint.is_symlink())
        self.assertEqual(args.runtime, self.root / "physicsnemo-infer")
        self.assertEqual(metadata["effective"]["output_root"], str(self.root))

    def test_project_configuration_symlink_is_rejected(self):
        actual = self.root / "actual.json"
        actual.write_text(json.dumps({"format_version": 1, "model": "affine"}))
        self.path.symlink_to(actual)
        with self.assertRaisesRegex(ValueError, "project.*symlink"):
            apply_project(self.args())

    def test_numeric_overflow_is_rejected_as_nonfinite_json(self):
        self.path.write_text('{"format_version":1,"model":"affine","points":1e999}')
        with self.assertRaisesRegex(ValueError, "non-finite"):
            apply_project(self.args())

    def test_model_preparation_fields_belong_to_external_examples(self):
        for field in ("points", "geometry_points"):
            with self.subTest(field=field):
                self.write(**{field: 32})
                with self.assertRaisesRegex(ValueError, "Unknown project field"):
                    apply_project(self.args())

    def test_missing_project_paths_fail_instead_of_falling_back_to_model_lookup(self):
        for value in (str(self.root / "missing"), "./missing", "missing.json"):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "project"),
            ):
                apply_project(self.args(model=value))
        with self.assertRaisesRegex(ValueError, "model-build.json"):
            apply_project(self.args())

    def test_missing_windows_project_paths_do_not_fall_back_to_model_lookup(self):
        for value in (
            rf"C:\{self.root.name}\missing",
            rf"{self.root.name}\missing",
        ):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "Cannot find project"),
            ):
                apply_project(self.args(model=value))

    def test_profiles_require_a_project_and_existing_name(self):
        with self.assertRaisesRegex(ValueError, "profile.*project"):
            apply_project(self.args(model="affine", profile="gpu"))
        self.write(profiles={"gpu": {"device": "cuda"}})
        with self.assertRaisesRegex(ValueError, "profile.*missing"):
            apply_project(self.args(profile="missing"))
        self.write(default_profile="missing")
        with self.assertRaisesRegex(ValueError, "default_profile"):
            apply_project(self.args())

    def test_project_and_explicit_recipe_are_ambiguous(self):
        self.write()
        with self.assertRaisesRegex(ValueError, "project.*--recipe"):
            apply_project(self.args(recipe=Path("other.json")))

    def test_duplicate_cli_assets_fail_instead_of_silently_choosing_one(self):
        self.write()
        with self.assertRaisesRegex(ValueError, "duplicate asset"):
            apply_project(self.args(asset=["mesh=one.bin", "mesh=two.bin"]))
        for value in ("invalid", "=value", "mesh="):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "NAME=FILE"),
            ):
                apply_project(self.args(asset=[value]))

    def test_json_duplicate_fields_nonfinite_numbers_and_nonobjects_fail(self):
        for source, message in (
            ('{"format_version":1,"model":"a","model":"b"}', "duplicate"),
            ('{"format_version":1,"model":"a","points":NaN}', "non-finite"),
            ('{"format_version":1,"model":"a","points":Infinity}', "non-finite"),
            ('["affine"]', "object"),
            ('{"format_version":', "JSON"),
        ):
            with self.subTest(source=source):
                self.path.write_text(source)
                with self.assertRaisesRegex(ValueError, message):
                    apply_project(self.args())

    def test_schema_rejects_unknown_wrong_type_and_unsupported_values(self):
        invalid = (
            ({"format_version": True}, "format_version"),
            ({"format_version": 2}, "format_version"),
            ({"typo": True}, "typo"),
            ({"model": "../model"}, "model"),
            ({"checkpoint": ""}, "checkpoint"),
            ({"checkpoint": 42}, "checkpoint"),
            ({"checkpoint_sha256": "abc"}, "checkpoint_sha256"),
            ({"assets": []}, "assets"),
            ({"assets": {"mesh": 12}}, "assets.mesh"),
            ({"assets": {"": "mesh.bin"}}, "asset name"),
            ({"backends": "aoti"}, "backends"),
            ({"backends": []}, "backends"),
            ({"backends": ["aoti", "aoti"]}, "backends"),
            ({"backends": ["unknown"]}, "backends"),
            ({"executor": "remote"}, "executor"),
            ({"device": "gpu"}, "device"),
            ({"required_gpu_arch": "H100"}, "required_gpu_arch"),
            ({"points": True}, "points"),
            ({"points": 1}, "points"),
            ({"geometry_points": 0}, "geometry_points"),
            ({"profiles": []}, "profiles"),
            ({"profiles": {"gpu": []}}, "profiles.gpu"),
            ({"profiles": {"gpu": {"checkpoint": "other.pt"}}}, "checkpoint"),
            ({"profiles": {"gpu": {"points": 64}}}, "points"),
            ({"profiles": {"gpu": {"device": "bad"}}}, "device"),
        )
        for fields, message in invalid:
            with self.subTest(fields=fields):
                self.write(**fields)
                with self.assertRaisesRegex(ValueError, message):
                    apply_project(self.args())

    def test_exactly_one_model_or_recipe_is_required(self):
        for document in (
            {"format_version": 1},
            {"format_version": 1, "model": "affine", "recipe": "recipe.json"},
        ):
            with self.subTest(document=document):
                self.path.write_text(json.dumps(document))
                with self.assertRaisesRegex(ValueError, "exactly one.*model.*recipe"):
                    apply_project(self.args())


if __name__ == "__main__":
    unittest.main()
