"""Customer source capture survives staging and rejects ambiguous imports."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_ROOT))
from pnmir_build import authoring_sources  # noqa: E402


class AuthoringSourcesTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.project = self.root / "project"
        self.project.mkdir()
        self.write("build_adapter.py", "# adapter must be captured\n")

    def write(self, name, content):
        path = self.project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content))
        return path

    def run_python(self, script):
        environment = dict(os.environ, PYTHONPATH=str(SOURCE_ROOT))
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=self.root,
            env=environment,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def retained(self, *, adapter="build_adapter.py", config=None):
        retained = self.root / "retained"
        retained.mkdir(exist_ok=True)
        source_files = {}
        assets = {}
        for index, path in enumerate(sorted(self.project.rglob("*.py"))):
            name = f"source_{index}"
            target = retained / f"{name}.py"
            shutil.copyfile(path, target)
            source_files[path.relative_to(self.project).as_posix()] = name
            assets[name] = str(target)
        datum = retained / "data.json"
        datum.write_text('{"offset": 7}')
        assets["validation"] = str(datum)
        return {
            "_source_files": source_files,
            "_adapter": adapter,
            "_model_config": config or {"width": 3},
            "_user_assets": ["validation"],
        }, assets

    def test_capture_is_explicit_sorted_and_content_identified(self):
        self.write("customer_model/__init__.py", "")
        module = self.write("customer_model/layer.py", "WIDTH = 3\n")
        self.write("unselected.py", "raise RuntimeError('do not import')\n")
        captured = authoring_sources.capture(
            self.project,
            ["customer_model", "customer_model/layer.py"],
            "build_adapter.py",
        )
        self.assertEqual(
            list(captured),
            [
                "build_adapter.py",
                "customer_model/__init__.py",
                "customer_model/layer.py",
            ],
        )
        self.assertEqual(
            captured["customer_model/layer.py"],
            {
                "path": str(module),
                "sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
                "size_bytes": module.stat().st_size,
            },
        )

    def test_directory_capture_omits_non_sources_and_generated_directories(self):
        self.write("model.py", "")
        self.write("weights.pt", "opaque")
        for name in (
            ".git",
            ".venv",
            ".private",
            "build",
            "builds",
            "dist",
            "__pycache__",
            "node_modules",
        ):
            self.write(f"{name}/private.py", "")
        self.assertEqual(
            set(authoring_sources.capture(self.project, ["."], "build_adapter.py")),
            {"build_adapter.py", "model.py"},
        )

    def test_invalid_explicit_paths_are_rejected(self):
        for selected in (
            "../outside.py",
            "/tmp/model.py",
            "missing.py",
            "weights.pt",
            "model\\layer.py",
        ):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                authoring_sources.capture(self.project, [selected], "build_adapter.py")

    def test_custom_output_directory_is_excluded_from_recursive_source_selection(self):
        self.write("model.py", "")
        self.write(
            "results/model-source/build_adapter.py", "raise RuntimeError('old build')\n"
        )
        self.write("results_other/helper.py", "")
        for exclusion in ("results", self.project / "results"):
            with self.subTest(exclusion=exclusion):
                self.assertEqual(
                    set(
                        authoring_sources.capture(
                            self.project, ["."], "build_adapter.py", exclude=[exclusion]
                        )
                    ),
                    {"build_adapter.py", "model.py", "results_other/helper.py"},
                )

    def test_explicit_sources_and_adapter_cannot_select_excluded_output_directory(self):
        self.write("results/model.py", "")
        for selected in ("results", "results/model.py"):
            with (
                self.subTest(selected=selected),
                self.assertRaisesRegex(ValueError, "excluded"),
            ):
                authoring_sources.capture(
                    self.project, [selected], "build_adapter.py", exclude=["results"]
                )
        with self.assertRaisesRegex(ValueError, "excluded"):
            authoring_sources.capture(
                self.project, ["."], "results/model.py", exclude=["results"]
            )

    def test_symlink_file_and_directory_are_rejected(self):
        self.write("real.py", "")
        (self.project / "link.py").symlink_to(self.project / "real.py")
        (self.project / "linked").symlink_to(self.root, target_is_directory=True)
        for selected in ("link.py", "linked", "."):
            with (
                self.subTest(selected=selected),
                self.assertRaisesRegex(ValueError, "symlink"),
            ):
                authoring_sources.capture(self.project, [selected], "build_adapter.py")

    def test_adapter_must_be_python_regular_file(self):
        for adapter in ("missing.py", "../build_adapter.py", "."):
            with self.subTest(adapter=adapter), self.assertRaises(ValueError):
                authoring_sources.capture(self.project, [], adapter)

    def test_retained_sources_support_package_relative_and_lazy_imports(self):
        self.write("customer_model/__init__.py", "")
        self.write("customer_model/helper.py", "VALUE = 11\n")
        self.write(
            "customer_model/adapter.py",
            """
            import json
            from .helper import VALUE

            def create_model(config, assets):
                assert set(config) == {'width'}
                assert set(assets) == {'validation'}
                config['width'] = 99
                class Model:
                    def run(self):
                        from customer_model.later import EXTRA
                        return VALUE + EXTRA
                return Model()

            def create_cases(config, assets):
                assert config == {'width': 3}
                return [(config['width'], json.loads(assets['validation'].read_text())['offset'])]
        """,
        )
        self.write("customer_model/later.py", "EXTRA = 2\n")
        config, assets = self.retained(adapter="customer_model/adapter.py")
        shutil.rmtree(self.project)
        self.run_python(f"""
            from pathlib import Path
            from pnmir_build.authoring_sources import create_model, create_cases
            config = {config!r}
            assets = {assets!r}
            model = create_model(config, assets)
            assert model is not None, 'captured adapter must construct model'
            assert model.run() == 13
            assert create_cases(config, assets) == [(3, 7)]
            assert config['_model_config'] == {{'width': 3}}
            assert not list(Path({str(self.root / "retained")!r}).rglob('__pycache__'))
        """)

    def test_optional_export_hook_uses_retained_helper_source(self):
        self.write("export_fixes.py", "def prepare_graph(module): return 0\n")
        self.write(
            "build_adapter.py",
            """
            from pnmir_export import ExportOptions
            from export_fixes import prepare_graph
            def create_model(config, assets): return config['width']
            def export_options(context):
                return ExportOptions(onnx_passes=(prepare_graph,)) if context.backend == 'tensorrt' else ExportOptions()
        """,
        )
        config, assets = self.retained()
        shutil.rmtree(self.project)
        self.run_python(f"""
            from pnmir_build.authoring_sources import create_model, export_options
            from pnmir_export import ExportContext
            assert create_model({config!r}, {assets!r}) == 3
            options = export_options({config!r}, {assets!r}, ExportContext('tensorrt', 'cuda'))
            assert len(options.onnx_passes) == 1
            assert options.onnx_passes[0](None) == 0
            assert export_options({config!r}, {assets!r}, ExportContext('aoti', 'cuda')).onnx_passes == ()
        """)

    def test_missing_export_hook_preserves_existing_adapter_behavior(self):
        self.write(
            "build_adapter.py",
            "def create_model(config, assets): return config['width']\n",
        )
        config, assets = self.retained()
        self.run_python(f"""
            from pnmir_build.authoring_sources import export_options
            from pnmir_export import ExportContext
            assert export_options({config!r}, {assets!r}, ExportContext('tensorrt', 'cuda')).onnx_passes == ()
        """)

    def test_preloaded_external_module_cannot_be_silently_reused(self):
        self.write("model.py", "VALUE = 17\n")
        self.write(
            "build_adapter.py",
            "from model import VALUE\ndef create_model(config, assets): return VALUE\n",
        )
        config, assets = self.retained()
        self.run_python(f"""
            import sys, types
            from pnmir_build.authoring_sources import create_model
            previous = types.ModuleType('model')
            previous.VALUE = -1
            sys.modules['model'] = previous
            try:
                create_model({config!r}, {assets!r})
            except ValueError as exc:
                assert 'model' in str(exc) and 'conflict' in str(exc).lower()
            else:
                raise AssertionError('ambiguous preloaded model module must be rejected')
            assert sys.modules['model'] is previous
        """)

    def test_standard_library_shadowing_is_rejected(self):
        self.write("fractions.py", "")
        config, assets = self.retained()
        self.run_python(f"""
            from pnmir_build.authoring_sources import create_model
            try:
                create_model({config!r}, {assets!r})
            except ValueError as exc:
                assert 'fractions' in str(exc) and 'conflict' in str(exc).lower()
            else:
                raise AssertionError('captured standard library shadow must be rejected')
        """)

    def test_identical_sources_reuse_imports_across_two_staging_directories(self):
        self.write("model.py", "class Model: pass\n")
        self.write(
            "build_adapter.py",
            """
            from model import Model
            def create_model(config, assets):
                return Model()
            def create_cases(config, assets):
                return [(config['width'], assets['validation'].parent.name)]
        """,
        )
        config, assets = self.retained()
        second = self.root / "second"
        shutil.copytree(self.root / "retained", second)
        second_assets = {
            key: str(second / Path(path).name) for key, path in assets.items()
        }
        second_config = {**config, "_model_config": {"width": 5}}
        self.run_python(f"""
            from pnmir_build.authoring_sources import create_model, create_cases
            first = create_model({config!r}, {assets!r})
            second = create_model({second_config!r}, {second_assets!r})
            assert type(first) is type(second)
            assert create_cases({second_config!r}, {second_assets!r}) == [(5, 'second')]
        """)

    def test_changed_source_is_rejected_instead_of_reusing_old_imports(self):
        self.write("model.py", "VALUE = 1\n")
        self.write(
            "build_adapter.py",
            "from model import VALUE\ndef create_model(config, assets): return VALUE\n",
        )
        config, assets = self.retained()
        model_asset = assets[config["_source_files"]["model.py"]]
        self.run_python(f"""
            from pathlib import Path
            from pnmir_build.authoring_sources import create_model
            assert create_model({config!r}, {assets!r}) == 1
            Path({model_asset!r}).write_text('VALUE = 2\\n')
            try:
                create_model({config!r}, {assets!r})
            except ValueError as exc:
                assert 'conflict' in str(exc).lower()
            else:
                raise AssertionError('changed source must not silently reuse stale imports')
        """)

    def test_ambiguous_module_and_package_are_rejected(self):
        self.write("model.py", "VALUE = 1\n")
        self.write("model/__init__.py", "VALUE = 2\n")
        self.write(
            "build_adapter.py",
            "from model import VALUE\ndef create_model(config, assets): return VALUE\n",
        )
        config, assets = self.retained()
        self.run_python(f"""
            from pnmir_build.authoring_sources import create_model
            try:
                create_model({config!r}, {assets!r})
            except ValueError as exc:
                assert 'model' in str(exc) and 'conflict' in str(exc).lower()
            else:
                raise AssertionError('module/package import conflict must be rejected')
        """)

    def test_second_tree_cannot_replace_a_not_yet_imported_lazy_dependency(self):
        self.write("lazy_model.py", "VALUE = 1\n")
        self.write(
            "build_adapter.py",
            """
            def create_model(config, assets):
                def model():
                    from lazy_model import VALUE
                    return VALUE
                return model
        """,
        )
        config, assets = self.retained()
        second = self.root / "second"
        shutil.copytree(self.root / "retained", second)
        second_assets = {
            key: str(second / Path(path).name) for key, path in assets.items()
        }
        second_files = dict(config["_source_files"])
        second_files["other_adapter.py"] = second_files.pop("build_adapter.py")
        second_config = {
            **config,
            "_source_files": second_files,
            "_adapter": "other_adapter.py",
        }
        lazy_asset = second_assets[second_files["lazy_model.py"]]
        Path(lazy_asset).write_text("VALUE = 2\n")
        self.run_python(f"""
            from pnmir_build.authoring_sources import create_model
            first = create_model({config!r}, {assets!r})
            try:
                create_model({second_config!r}, {second_assets!r})
            except ValueError as exc:
                assert 'lazy_model' in str(exc) and 'conflict' in str(exc).lower()
            else:
                raise AssertionError('source roots must remain reserved for lazy imports')
            assert first() == 1
        """)

    def test_reserved_mapping_rejects_traversal_aliases_and_user_asset_leaks(self):
        config, assets = self.retained()
        mutations = [
            {"_source_files": {"../outside.py": "source_0"}},
            {"_source_files": {"build_adapter.py": "source_0", "other.py": "source_0"}},
            {"_adapter": "missing.py"},
            {"_user_assets": ["source_0"]},
            {"_user_assets": ["missing"]},
            {"_model_config": []},
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                authoring_sources.create_model({**config, **mutation}, assets)

    def test_callback_cannot_mutate_its_adapter_or_lazy_source_copy(self):
        for callback in ("create_model", "create_cases"):
            for target in ("build_adapter.py", "lazy_model.py"):
                with self.subTest(callback=callback, target=target):
                    self.write("lazy_model.py", "VALUE = 1\n")
                    self.write(
                        "build_adapter.py",
                        f"""
                        from pathlib import Path
                        def {callback}(config, assets):
                            target = Path(__file__).with_name({target!r})
                            target.write_text(target.read_text() + '\\n# changed\\n')
                            return 1
                    """,
                    )
                    config, assets = self.retained()
                    source_asset = assets[config["_source_files"][target]]
                    self.run_python(f"""
                        from pathlib import Path
                        from pnmir_build import authoring_sources
                        before = Path({source_asset!r}).read_bytes()
                        try:
                            authoring_sources.{callback}({config!r}, {assets!r})
                        except ValueError as exc:
                            assert 'source' in str(exc).lower() and 'changed' in str(exc).lower()
                        else:
                            raise AssertionError('mutated imported source copy must fail')
                        assert Path({source_asset!r}).read_bytes() == before
                    """)

    def test_mutated_source_is_rejected_before_next_callback_runs(self):
        self.write("lazy_model.py", "VALUE = 1\n")
        self.write(
            "build_adapter.py",
            """
            from pathlib import Path
            def create_model(config, assets):
                return Path(__file__).with_name('lazy_model.py')
            def create_cases(config, assets):
                assets['validation'].write_text('callback ran')
                return []
        """,
        )
        config, assets = self.retained()
        self.run_python(f"""
            from pathlib import Path
            from pnmir_build import authoring_sources
            model_source = authoring_sources.create_model({config!r}, {assets!r})
            model_source.write_text('VALUE = 2\\n')
            try:
                authoring_sources.create_cases({config!r}, {assets!r})
            except ValueError as exc:
                assert 'source' in str(exc).lower() and 'changed' in str(exc).lower()
            else:
                raise AssertionError('must check live imports before invoking callback')
            assert Path({assets["validation"]!r}).read_text() != 'callback ran'
        """)

    def test_public_import_verification_detects_added_removed_and_symlinked_sources(
        self,
    ):
        self.write("lazy_model.py", "VALUE = 1\n")
        self.write(
            "build_adapter.py",
            """
            from pathlib import Path
            def create_model(config, assets):
                return Path(__file__).parent
        """,
        )
        config, assets = self.retained()
        for mutation in (
            "(directory / 'added.py').write_text('VALUE = 2\\n')",
            "(directory / 'lazy_model.py').unlink()",
            "(directory / 'lazy_model.py').unlink(); (directory / 'lazy_model.py').symlink_to(directory / 'build_adapter.py')",
        ):
            with self.subTest(mutation=mutation):
                self.run_python(f"""
                    from pnmir_build import authoring_sources
                    directory = authoring_sources.create_model({config!r}, {assets!r})
                    authoring_sources.verify_imports()
                    {mutation}
                    try:
                        authoring_sources.verify_imports()
                    except ValueError as exc:
                        assert 'source' in str(exc).lower() and 'changed' in str(exc).lower()
                    else:
                        raise AssertionError('verification must catch changed live source tree')
                """)


if __name__ == "__main__":
    unittest.main()
