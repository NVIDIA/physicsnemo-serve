"""Recipe input files, frontend plans, and completed model-input receipts."""

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile

from model_builder.build import cli, inputs, worker
import worker_test_support


class ModelInputFixture:
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
            "dtype": "float32",
            "shape": [4],
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

    def completed_v2(self):
        fixture = worker_test_support.WorkerFixture()
        fixture.addCleanup = self.addCleanup
        if hasattr(fixture, "setUp"):
            fixture.setUp()
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
        build["source"]["files"] = worker._inventory(source, fixture.output)
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


class CliRecipeFixture:
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.recipe_path = self.source / "recipe.json"
        self.recipe = {
            "format_version": 1,
            "name": "affine",
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
        }
        self.recipe_path.write_text(json.dumps(self.recipe))
        (self.source / "export.py").write_text("# recipe\n")
        self.output = self.root / "candidate"
        self.image = "example/builder@sha256:" + "a" * 64
        self.lock = self.root / "toolchain.lock.json"
        self.lock.write_text(
            json.dumps({"format_version": 1, "builder_image": self.image})
        )

    def plan(self, adapter):
        return {
            "recipe_path": self.recipe_path,
            "recipe": dict(self.recipe, adapter=adapter),
            "output": self.output,
            "device": "cpu",
            "image": self.image,
            "backends": ["aoti"],
        }

    def invoke(self, *args):
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            return cli.main(
                [
                    "build",
                    "--recipe",
                    str(self.recipe_path),
                    "--output",
                    str(self.output),
                    "--lock",
                    str(self.lock),
                    "--device",
                    "cpu",
                    *args,
                ]
            )
