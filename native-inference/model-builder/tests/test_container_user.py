import getpass
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pnmir_build import cli


class ContainerUserTests(unittest.TestCase):
    def test_unlisted_host_uid_has_username_and_writable_compiler_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = {"adapter": "export.py"}
            (root / "recipe.json").write_text(json.dumps(recipe))
            (root / "export.py").write_text("# adapter\n")
            plan = {
                "recipe_path": root / "recipe.json",
                "recipe": recipe,
                "output": root / "candidate",
                "device": "cuda",
                "image": "sha256:" + "a" * 64,
                "backends": ["aoti"],
            }
            with mock.patch.object(
                cli.subprocess, "run", return_value=subprocess.CompletedProcess([], 17)
            ) as run:
                self.assertEqual(cli.container_build(plan), 17)
            command = run.call_args.args[0]
            environment = dict(
                command[index + 1].split("=", 1)
                for index, value in enumerate(command)
                if value == "-e"
            )
            error = None
            username = None
            with (
                mock.patch.dict(os.environ, environment, clear=True),
                mock.patch("pwd.getpwuid", side_effect=KeyError("uid not found")),
            ):
                try:
                    username = getpass.getuser()
                except (KeyError, OSError) as caught:
                    error = str(caught)
            self.assertIsNone(
                error, "compiler username discovery must work without a passwd entry"
            )
            self.assertTrue(username)
            self.assertTrue(
                environment.get("TRITON_CACHE_DIR", "").startswith("/tmp/"),
                "Triton cache must not default to an inaccessible image user's home",
            )
            self.assertEqual(
                command[command.index("--user") + 1], f"{os.getuid()}:{os.getgid()}"
            )


if __name__ == "__main__":
    unittest.main()
