"""Model input orchestration remains framework-free and preserves provenance."""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pnmir_build import cli, inputs
import test_worker


class ModelInputCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "recipe"
        self.source.mkdir()
        self.recipe_path = self.source / "recipe.json"
        self.recipe = {
            "format_version": 2,
            "name": "configured",
            "version": "0.1.0",
            "adapter": "export.py",
            "factory": "create_model",
            "cases": "create_cases",
            "input_names": ["input"],
            "output_names": ["output"],
            "supported_backends": ["aoti"],
            "default_backend": "aoti",
            "config": {"path": "config.json"},
            "checkpoint": {"format": "torch-state-dict"},
            "assets": {"normalization": {"path": "normalization.json"}},
        }
        self.recipe_path.write_text(json.dumps(self.recipe, indent=3) + "\n")
        (self.source / "export.py").write_text("# no model imports during planning\n")
        config = self.source / "config.json"
        config.write_text('{"width": 4}\n')
        checkpoint = self.root / "customer checkpoint.pt"
        checkpoint.write_bytes(b"opaque weights; never loaded by the CLI")
        asset = self.source / "normalization.json"
        asset.write_text('{"scale": 2}\n')

        def descriptor(path, origin):
            return {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
                "origin": origin,
            }

        self.resolved = {
            "config": descriptor(config, "recipe"),
            "checkpoint": descriptor(checkpoint, "cli"),
            "assets": {"normalization": descriptor(asset, "recipe")},
            "config_data": {"width": 4},
            "effective_config": {"sha256": "e" * 64, "size_bytes": 12},
        }
        self.identities = {
            name: {key: value[key] for key in ("sha256", "size_bytes")}
            for name, value in self.resolved.items()
            if name in ("config", "checkpoint", "effective_config")
        }
        self.identities["assets"] = {
            "normalization": {
                key: self.resolved["assets"]["normalization"][key]
                for key in ("sha256", "size_bytes")
            }
        }
        self.output = self.root / "candidate"
        self.runtime = self.root / "physicsnemo-infer"
        self.runtime.write_text("#!/bin/sh\nexit 0\n")
        self.runtime.chmod(0o755)
        self.image = "example/builder@sha256:" + "a" * 64
        self.lock = self.root / "toolchain.lock.json"
        self.lock.write_text(json.dumps({"format_version": 1}))
        self.plan = {
            "recipe_path": self.recipe_path,
            "recipe": self.recipe,
            "model_inputs": self.resolved,
            "output": self.output,
            "device": "cpu",
            "image": self.image,
            "runtime": self.runtime,
            "backends": ["aoti"],
            "executor": "local",
            "toolchain_lock": {"path": str(self.lock), "sha256": "b" * 64},
            "selection_source": "argument",
        }

    def arguments(self, *extra):
        return [
            "build",
            "--recipe",
            str(self.recipe_path),
            "--output",
            str(self.output),
            "--lock",
            str(self.lock),
            "--executor",
            "local",
            "--runtime",
            str(self.runtime),
            "--device",
            "cpu",
            *extra,
        ]

    def parse(self, *extra):
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                parsed = cli.parser().parse_args(self.arguments(*extra))
            except SystemExit:
                parsed = None
        self.assertIsInstance(
            parsed, argparse.Namespace, "model input flags must parse"
        )
        return parsed

    def invoke(self, *args):
        stderr = io.StringIO()
        with (
            contextlib.redirect_stderr(stderr),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            try:
                code = cli.main(self.arguments(*args))
            except SystemExit as error:
                code = error.code
        return code, stderr.getvalue()

    def test_build_and_doctor_expose_model_input_options(self):
        args = self.parse(
            "--config",
            "customer.json",
            "--checkpoint",
            "weights.pt",
            "--checkpoint-sha256",
            "c" * 64,
            "--asset",
            "normalization=stats.json",
        )
        self.assertEqual(str(args.config), "customer.json")
        self.assertEqual(str(args.checkpoint), "weights.pt")
        self.assertEqual(args.checkpoint_sha256, "c" * 64)
        self.assertEqual(args.asset, ["normalization=stats.json"])
        with contextlib.redirect_stdout(io.StringIO()) as help_text:
            with self.assertRaises(SystemExit) as exit_info:
                cli.parser().parse_args(["doctor", "--help"])
        self.assertEqual(exit_info.exception.code, 0)
        for option in ("--config", "--checkpoint", "--checkpoint-sha256", "--asset"):
            self.assertIn(option, help_text.getvalue())

    def test_v2_recipe_is_accepted_and_input_schema_errors_are_usage_errors(self):
        with mock.patch.object(inputs, "validate_input_spec") as validate:
            try:
                actual = cli.read_recipe(self.recipe_path)
            except cli.UsageError as error:
                actual = str(error)
            self.assertEqual(actual, self.recipe)
            validate.assert_called_once_with(self.recipe)
        with mock.patch.object(
            inputs,
            "validate_input_spec",
            side_effect=ValueError("checkpoint format invalid"),
        ):
            with self.assertRaisesRegex(cli.UsageError, "checkpoint format invalid"):
                cli.read_recipe(self.recipe_path)

    def test_resolution_preserves_original_inputs_and_forwards_all_overrides(self):
        args = self.parse(
            "--config",
            "customer.json",
            "--checkpoint",
            "weights.pt",
            "--checkpoint-sha256",
            "c" * 64,
            "--asset",
            "normalization=stats=latest.json",
        )
        with (
            mock.patch.object(inputs, "validate_input_spec"),
            mock.patch.object(
                inputs, "resolve_inputs", return_value=self.resolved
            ) as resolve,
        ):
            plan = cli.resolve(args)
        self.assertEqual(plan.get("model_inputs"), self.resolved)
        resolve.assert_called_once_with(
            self.recipe,
            self.recipe_path,
            config=Path("customer.json"),
            checkpoint=Path("weights.pt"),
            checkpoint_sha256="c" * 64,
            assets={"normalization": "stats=latest.json"},
        )
        self.assertFalse(self.output.exists())

    def test_invalid_asset_options_fail_before_resolution_or_output_creation(self):
        examples = (
            (
                ["--asset", "normalization=one", "--asset", "normalization=two"],
                "duplicate",
            ),
            (["--asset", "unknown=one"], "unknown"),
            (["--asset", "normalization"], "NAME=FILE"),
            (["--asset", "normalization="], "NAME=FILE"),
            (["--asset", "=one"], "NAME=FILE"),
        )
        for flags, message in examples:
            with (
                self.subTest(flags=flags),
                mock.patch.object(inputs, "resolve_inputs") as resolve,
            ):
                code, error = self.invoke(*flags)
                self.assertEqual(code, 2)
                self.assertIn(message, error)
                resolve.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_input_validation_precedes_dependency_checks_and_ml_imports(self):
        # Intentionally invalid runtime/image must not hide a useful input error.
        with (
            mock.patch.object(inputs, "validate_input_spec"),
            mock.patch.object(
                inputs,
                "resolve_inputs",
                side_effect=ValueError("checkpoint SHA-256 mismatch"),
            ),
            mock.patch.object(cli.shutil, "which") as docker,
            mock.patch("pnmir_build.worker.execute_build") as execute,
        ):
            code, error = self.invoke("--runtime", str(self.root / "missing"))
        self.assertEqual(code, 2)
        self.assertIn("checkpoint SHA-256 mismatch", error)
        execute.assert_not_called()
        docker.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_local_v2_forwards_inputs_and_retains_host_provenance_on_failure(self):
        for failed in (False, True):
            with self.subTest(failed=failed):
                self.plan["output"] = self.root / f"local-{failed}"

                def execute(*args, **kwargs):
                    self.plan["output"].mkdir()
                    if failed:
                        raise RuntimeError("compiler rejected graph")
                    return {"status": "complete"}

                with (
                    mock.patch.object(cli, "resolve", return_value=self.plan),
                    mock.patch(
                        "pnmir_build.worker.execute_build", side_effect=execute
                    ) as worker,
                ):
                    code, _ = self.invoke()
                self.assertEqual(code, int(failed))
                self.assertEqual(
                    worker.call_args.kwargs, {"model_inputs": self.resolved}
                )
                receipt = json.loads(
                    (self.plan["output"] / "execution.json").read_text()
                )
                sources = receipt.get("model_input_sources", {})
                for field in ("config", "checkpoint", "assets"):
                    self.assertEqual(sources.get(field), self.resolved[field])
                self.assertNotIn("config_data", sources)

    def test_v1_local_worker_retains_original_call_signature(self):
        self.plan["recipe"] = dict(self.recipe, format_version=1)
        self.plan["model_inputs"] = None
        with (
            mock.patch.object(cli, "resolve", return_value=self.plan),
            mock.patch("pnmir_build.worker.execute_build", return_value={}) as execute,
        ):
            code, _ = self.invoke()
        self.assertEqual(code, 0)
        execute.assert_called_once_with(
            self.recipe_path, self.output, ["aoti"], "cpu", self.runtime
        )

    def test_v1_input_override_is_rejected_before_worker_import(self):
        self.recipe_path.write_text(json.dumps(dict(self.recipe, format_version=1)))
        with mock.patch("pnmir_build.worker.execute_build") as execute:
            code, error = self.invoke("--config", self.resolved["config"]["path"])
        self.assertEqual(code, 2)
        self.assertIn("overrides require recipe format_version 2", error)
        execute.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_doctor_resolves_real_v2_files_without_site_packages(self):
        arguments = self.arguments("--checkpoint", self.resolved["checkpoint"]["path"])
        arguments[0] = "doctor"
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                str(Path(__file__).resolve().parents[2] / "physicsnemo-model-builder"),
                *arguments,
            ],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "configuration-ok")
        self.assertFalse(self.output.exists())

    def test_container_stages_inputs_and_keeps_original_recipe_bytes(self):
        original = self.recipe_path.read_bytes()

        def stage(resolved, destination):
            staged = copy.deepcopy(resolved)
            files = (
                (staged["config"], "config.json"),
                (staged["checkpoint"], "checkpoint.pt"),
                (
                    staged["assets"]["normalization"],
                    "assets/normalization/normalization.json",
                ),
            )
            for descriptor, relative in files:
                target = destination / "model-inputs" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(descriptor["path"], target)
                descriptor["path"] = str(target.resolve())
            return staged

        def docker(command, **kwargs):
            self.assertIn("--config", command)
            mount = next(value for value in command if "dst=/inputs," in value)
            self.assertTrue(mount.endswith(",readonly"))
            staged = Path(mount.split("src=", 1)[1].split(",dst=", 1)[0])
            self.assertEqual((staged / "recipe.json").read_bytes(), original)
            for flag, descriptor in (
                ("--config", self.resolved["config"]),
                ("--checkpoint", self.resolved["checkpoint"]),
            ):
                argument = command[command.index(flag) + 1]
                relative = Path(argument).relative_to("/inputs")
                self.assertEqual(
                    (staged / relative).read_bytes(),
                    Path(descriptor["path"]).read_bytes(),
                )
                self.assertNotIn(descriptor["path"], command)
            self.assertEqual(
                command[command.index("--checkpoint-sha256") + 1],
                self.resolved["checkpoint"]["sha256"],
            )
            self.assertEqual(
                command[command.index("--asset") + 1],
                "normalization=/inputs/model-inputs/assets/normalization/normalization.json",
            )
            return subprocess.CompletedProcess(command, 0)

        with (
            mock.patch.object(inputs, "stage_inputs", side_effect=stage) as stage_call,
            mock.patch.object(cli.subprocess, "run", side_effect=docker),
            mock.patch.object(cli, "_validate_container_completion") as validate,
        ):
            self.assertEqual(cli.container_build(self.plan), 0)
        self.assertEqual(stage_call.call_count, 1)
        self.assertEqual(stage_call.call_args.args[0], self.resolved)
        validate.assert_called_once_with(self.plan)
        self.assertEqual(self.recipe_path.read_bytes(), original)

    def completed_v2(self):
        fixture = test_worker.WorkerTest(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.run_build(["aoti"])
        recipe = dict(fixture.recipe, format_version=2)
        for field in ("config", "checkpoint", "assets"):
            recipe[field] = copy.deepcopy(self.recipe[field])
        resolved = inputs.resolve_inputs(
            recipe, self.recipe_path, checkpoint=self.resolved["checkpoint"]["path"]
        )
        source = fixture.output / "source"
        staged = inputs.stage_inputs(resolved, source)
        (source / "recipe.json").write_text(json.dumps(recipe))
        (source / "effective-recipe.json").write_text(
            json.dumps(inputs.effective_recipe(recipe, staged, source))
        )
        build_path = fixture.output / "build.json"
        release_path = fixture.output / "model/model-release.json"
        build = json.loads(build_path.read_text())
        build["variants"]["aoti"]["graph"]["entrypoint"] = "program.pt2"
        build["source"]["files"] = test_worker.worker._inventory(source, fixture.output)
        release = json.loads(release_path.read_text())
        identities = inputs.input_identities(resolved)
        build["model_inputs"] = identities
        release["model_inputs"] = identities
        release_path.write_text(json.dumps(release))
        build["release"].update(
            sha256=hashlib.sha256(release_path.read_bytes()).hexdigest(),
            size_bytes=release_path.stat().st_size,
        )
        build_path.write_text(json.dumps(build))
        plan = dict(
            self.plan,
            output=fixture.output,
            recipe=recipe,
            model_inputs=resolved,
        )
        return plan, build_path, release_path, build, release, identities

    def test_container_completion_checks_both_model_input_identities(self):
        plan, build_path, release_path, original_build, original_release, identities = (
            self.completed_v2()
        )
        for field in ("build", "release"):
            for mutation in ("missing", "wrong-checkpoint", "wrong-asset"):
                with self.subTest(field=field, mutation=mutation):
                    build = copy.deepcopy(original_build)
                    release = copy.deepcopy(original_release)
                    build["model_inputs"] = copy.deepcopy(identities)
                    release["model_inputs"] = copy.deepcopy(identities)
                    target = build if field == "build" else release
                    if mutation == "missing":
                        del target["model_inputs"]
                    elif mutation == "wrong-checkpoint":
                        target["model_inputs"]["checkpoint"]["sha256"] = "0" * 64
                    else:
                        target["model_inputs"]["assets"]["normalization"][
                            "size_bytes"
                        ] += 1
                    release_path.write_text(json.dumps(release))
                    build["release"].update(
                        sha256=hashlib.sha256(release_path.read_bytes()).hexdigest(),
                        size_bytes=release_path.stat().st_size,
                    )
                    build_path.write_text(json.dumps(build))
                    with self.assertRaisesRegex(RuntimeError, "model input"):
                        cli._validate_container_completion(plan)
        build["model_inputs"] = copy.deepcopy(identities)
        release["model_inputs"] = copy.deepcopy(identities)
        release_path.write_text(json.dumps(release))
        build["release"].update(
            sha256=hashlib.sha256(release_path.read_bytes()).hexdigest(),
            size_bytes=release_path.stat().st_size,
        )
        build_path.write_text(json.dumps(build))
        cli._validate_container_completion(plan)

    def test_container_binds_requested_input_identities_to_retained_bytes_and_inventory(
        self,
    ):
        plan, build_path, _, original_build, _, _ = self.completed_v2()
        source = plan["output"] / "source"
        original_files = {
            path: path.read_bytes() for path in source.rglob("*") if path.is_file()
        }
        cli._validate_container_completion(plan)
        selected = (
            "model-inputs/checkpoint.pt",
            "model-inputs/config.json",
            "model-inputs/assets/normalization/normalization.json",
            "model-inputs/effective-config.json",
        )
        scenarios = [("changed", path) for path in selected]
        scenarios += [
            ("uninventoried", path) for path in (*selected, "effective-recipe.json")
        ]
        scenarios += [("changed-with-new-pin", "model-inputs/checkpoint.pt")]
        for mutation, relative in scenarios:
            with self.subTest(mutation=mutation, relative=relative):
                for path, value in original_files.items():
                    path.write_bytes(value)
                build = copy.deepcopy(original_build)
                target = source / relative
                if mutation.startswith("changed"):
                    target.write_bytes(b'{"different":true}\n')
                    if mutation == "changed-with-new-pin":
                        effective_path = source / "effective-recipe.json"
                        effective = json.loads(effective_path.read_text())
                        effective["checkpoint"]["sha256"] = hashlib.sha256(
                            target.read_bytes()
                        ).hexdigest()
                        effective_path.write_text(json.dumps(effective))
                    build["source"]["files"] = test_worker.worker._inventory(
                        source, plan["output"]
                    )
                else:
                    build["source"]["files"] = [
                        record
                        for record in build["source"]["files"]
                        if record["path"] != "source/" + relative
                    ]
                build_path.write_text(json.dumps(build))
                with self.assertRaisesRegex(RuntimeError, "input|config|checkpoint"):
                    cli._validate_container_completion(plan)


if __name__ == "__main__":
    unittest.main()
