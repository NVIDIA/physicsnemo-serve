"""Behavioral guards for the image's explicitly pinned producer environment."""

import copy
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[2] / "images" / "verify_builder_environment.py"
SPEC = importlib.util.spec_from_file_location("builder_image_environment", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BuilderImageEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.before = {
            "python": "3.12",
            "system": "Linux",
            "machine": "x86_64",
            "protected": {"torch": {"version": "2.10.0a0+nv26.1", "sha256": "abc"}},
            "torch_cuda": "13.1",
        }
        self.after = copy.deepcopy(self.before) | {
            "packages": {
                "nvidia-physicsnemo": {
                    "version": "2.1.1",
                    "requires": ["torch>=2.10.0", "warp-lang>=1.13.0"],
                },
                "warp-lang": {"version": "1.15.0", "requires": ["numpy"]},
                "torch": {"version": "2.10.0a0+nv26.1", "requires": []},
                "numpy": {"version": "2.1.0", "requires": []},
            },
        }
        self.contract = {
            "python": "3.12",
            "system": "Linux",
            "machine": "x86_64",
            "torch_cuda": "13.1",
            "versions": {
                name: record["version"]
                for name, record in self.after["packages"].items()
            },
            "metadata_exceptions": [
                {
                    "consumer": "nvidia-physicsnemo",
                    "requirement": "torch>=2.10.0",
                    "actual": "2.10.0a0+nv26.1",
                }
            ],
        }

    def test_accepts_exact_qualified_prerelease_exception(self):
        self.assertEqual(
            MODULE.verify_environment(self.before, self.after, self.contract)["status"],
            "passed",
        )

    def test_rejects_pinned_producer_version_drift(self):
        for name in ("nvidia-physicsnemo", "warp-lang"):
            with self.subTest(name=name):
                after = copy.deepcopy(self.after)
                after["packages"][name]["version"] = "99.0.0"
                with self.assertRaisesRegex(ValueError, "version"):
                    MODULE.verify_environment(self.before, after, self.contract)

    def test_rejects_same_version_native_stack_replacement(self):
        after = copy.deepcopy(self.after)
        after["protected"]["torch"]["sha256"] = "different native library"
        with self.assertRaisesRegex(ValueError, "protected"):
            MODULE.verify_environment(self.before, after, self.contract)

    def test_rejects_missing_protected_stack_member(self):
        after = copy.deepcopy(self.after)
        after["protected"] = {}
        with self.assertRaisesRegex(ValueError, "protected"):
            MODULE.verify_environment(self.before, after, self.contract)

    def test_rejects_wrong_python_architecture_or_cuda(self):
        for key, value in (
            ("python", "3.13"),
            ("machine", "aarch64"),
            ("system", "Darwin"),
            ("torch_cuda", "12.8"),
        ):
            with self.subTest(key=key):
                after = copy.deepcopy(self.after)
                after[key] = value
                with self.assertRaisesRegex(ValueError, key):
                    MODULE.verify_environment(self.before, after, self.contract)

    def test_rejects_unpinned_active_transitive_dependency(self):
        after = copy.deepcopy(self.after)
        after["packages"]["warp-lang"]["requires"].append("new-dependency>=1")
        after["packages"]["new-dependency"] = {"version": "1.0", "requires": []}
        with self.assertRaisesRegex(ValueError, "unpinned"):
            MODULE.verify_environment(self.before, after, self.contract)

    def test_rejects_unsatisfied_dependency_outside_exact_exception(self):
        after = copy.deepcopy(self.after)
        after["packages"]["warp-lang"]["requires"] = ["numpy>=3"]
        with self.assertRaisesRegex(ValueError, "dependency"):
            MODULE.verify_environment(self.before, after, self.contract)

    def test_does_not_generalize_prerelease_exception(self):
        after = copy.deepcopy(self.after)
        after["packages"]["nvidia-physicsnemo"]["requires"][0] = "torch>=2.11.0"
        with self.assertRaisesRegex(ValueError, "dependency"):
            MODULE.verify_environment(self.before, after, self.contract)

    def test_checks_dependency_extras_when_requested(self):
        after = copy.deepcopy(self.after)
        after["packages"]["warp-lang"]["requires"] = ["numpy[extended]"]
        after["packages"]["numpy"]["requires"] = ["new-dependency; extra == 'extended'"]
        with self.assertRaisesRegex(ValueError, "unpinned"):
            MODULE.verify_environment(self.before, after, self.contract)

    def test_ignores_unrequested_optional_dependencies(self):
        after = copy.deepcopy(self.after)
        after["packages"]["numpy"]["requires"] = ["unneeded; extra == 'dev'"]
        self.assertEqual(
            MODULE.verify_environment(self.before, after, self.contract)["status"],
            "passed",
        )


if __name__ == "__main__":
    unittest.main()
