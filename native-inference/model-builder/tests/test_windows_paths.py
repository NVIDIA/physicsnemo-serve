"""Serialized relative paths retain package syntax on native Windows."""

from pathlib import Path, PureWindowsPath
import unittest
from unittest import mock

import test_worker
from pnmir_build import worker


class WindowsPackagePathTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_worker.WorkerTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_windows_inventory_uses_forward_slashes(self):
        path = mock.Mock(wraps=self.fixture.recipe_path)
        path.relative_to.return_value = PureWindowsPath("source/recipe.json")
        record = worker._file_identity(path, self.fixture.root)
        self.assertEqual(record["path"], "source/recipe.json")
        self.assertNotIn("\\", record["path"])

    def test_windows_backend_package_path_matches_inventory_convention(self):
        original_relative_to = Path.relative_to

        def windows_package_relative_to(path, *args, **kwargs):
            relative = original_relative_to(path, *args, **kwargs)
            if path.parent.name == "backends":
                return PureWindowsPath(*relative.parts)
            return relative

        with (
            mock.patch.object(Path, "relative_to", windows_package_relative_to),
            mock.patch.object(worker, "_native_case", return_value={"outputs": []}),
        ):
            receipt = self.fixture.run_build()
        for backend, variant in receipt["variants"].items():
            self.assertEqual(variant["package"], f"backends/{backend}")
            self.assertIn(
                variant["package"] + "/model.json",
                [item["path"] for item in variant["files"]],
            )


if __name__ == "__main__":
    unittest.main()
