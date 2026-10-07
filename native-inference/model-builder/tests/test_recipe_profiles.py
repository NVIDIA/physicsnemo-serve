"""Recipe-selected AOTI precision policy must reach the recorded artifact."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_builder.build import cli, worker
import model_input_test_support
import worker_test_support


EXACT = "aten-boundary-exact-v2"


class RecipeProfileTests(unittest.TestCase):
    def fixture(self):
        fixture = model_input_test_support.ModelInputFixture()
        fixture.addCleanup = self.addCleanup
        if hasattr(fixture, "setUp"):
            fixture.setUp()
        fixture.recipe.update(dtype="float32", shape=[4])
        return fixture

    def test_invalid_profile_fails_frontend_and_worker_before_model_import(self):
        fixture = self.fixture()
        for profile in ("unknown", "", None, {}, [], 2):
            fixture.recipe["aoti_profile"] = profile
            fixture.recipe_path.write_text(json.dumps(fixture.recipe))
            for reader in (
                cli.read_recipe,
                lambda path: worker._read_recipe(path, ["aoti"]),
            ):
                with self.subTest(profile=profile, reader=reader):
                    with self.assertRaisesRegex(ValueError, "AOTI.*profile"):
                        reader(fixture.recipe_path)

    def test_exact_profile_preflight_does_not_import_torch(self):
        fixture = self.fixture()
        fixture.recipe["aoti_profile"] = EXACT
        fixture.recipe_path.write_text(json.dumps(fixture.recipe))
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                "import sys; from pathlib import Path; "
                "sys.path.insert(0, sys.argv[1]); "
                "from model_builder.build import cli, worker; "
                "p = Path(sys.argv[2]); "
                "assert cli.read_recipe(p)['aoti_profile'] == sys.argv[3]; "
                "assert worker._read_recipe(p, ['aoti'])[0]['aoti_profile'] == sys.argv[3]; "
                "assert 'torch' not in sys.modules",
                str(Path(__file__).resolve().parents[1] / "src"),
                str(fixture.recipe_path),
                EXACT,
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_worker_forwards_exact_profile_and_preserves_baseline_call(self):
        fixture = self.fixture()
        for profile in (EXACT, "baseline"):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as temp:
                modules = worker_test_support.backend_modules()
                export = modules["model_builder.export.exporter"].export_package
                with mock.patch.dict(sys.modules, modules):
                    worker._build_backend(
                        "aoti",
                        {"model": object(), "cases": [()]},
                        dict(fixture.recipe, aoti_profile=profile),
                        "cpu",
                        Path(temp) / "package",
                        Path(temp) / "exported",
                    )
                if profile == EXACT:
                    self.assertEqual(export.call_args.kwargs.get("aoti_profile"), EXACT)
                else:
                    self.assertNotIn("aoti_profile", export.call_args.kwargs)

    def test_completion_rejects_missing_or_wrong_requested_profile(self):
        fixture = self.fixture()
        plan, _, _, build, release, _ = fixture.completed_v2()
        cli._validate_container_completion(plan)
        plan["recipe"]["aoti_profile"] = EXACT
        package = plan["output"] / "model" / build["variants"]["aoti"]["package"]
        manifest_path = package / "model.json"
        manifest = json.loads(manifest_path.read_text())

        def publish(profile):
            artifact = manifest["artifacts"][0]
            if profile is None:
                artifact.pop("correctness_profile", None)
            else:
                artifact["correctness_profile"] = {"name": profile, "version": 2}
            manifest_path.write_text(json.dumps(manifest))
            worker_test_support.publish_build_receipts(
                plan["output"], build, release, "aoti"
            )

        for profile in (None, "baseline", "unknown"):
            publish(profile)
            with self.subTest(profile=profile):
                with self.assertRaisesRegex(RuntimeError, "AOTI.*profile"):
                    cli._validate_container_completion(plan)
        publish(EXACT)
        cli._validate_container_completion(plan)


if __name__ == "__main__":
    unittest.main()
