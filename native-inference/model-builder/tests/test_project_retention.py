"""Retained provenance preserves published bytes across platform text conventions."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pnmir_build import authoring, cli, project_lock, project_run


class ProjectRetentionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_recipe_project_retains_original_source_and_published_lock_bytes(self):
        source = '{\r\n  "description": "训练模型"\r\n}\r\n'.encode("utf-8")
        identity = {"description": "训练模型"}
        for newline in ("\n", "\r\n"):
            with self.subTest(newline=repr(newline)):
                lock = self.root / "model-build.lock.json"
                lock_bytes = (
                    (
                        json.dumps(
                            {"format_version": 1, "entries": {"build": identity}},
                            ensure_ascii=False,
                            indent=2,
                        )
                        + "\n"
                    )
                    .replace("\n", newline)
                    .encode("utf-8")
                )
                lock.write_bytes(lock_bytes)
                published = project_lock.publish_lock(
                    lock, project_lock.inspect_lock(lock, "build", identity)
                )
                output = self.root / ("crlf" if newline == "\r\n" else "lf")
                output.mkdir()
                report = project_run.retain(
                    {
                        "output": output,
                        "project": {
                            "source_text": source.decode("utf-8"),
                            "published_lock": published,
                            "effective": {},
                            "profile": None,
                        },
                    }
                )
                retained = output / "project"
                self.assertEqual((retained / "model-build.json").read_bytes(), source)
                self.assertEqual((retained / lock.name).read_bytes(), lock_bytes)
                self.assertEqual(report["lock"]["sha256"], published["sha256"])
                self.assertEqual(
                    report["source"]["sha256"], hashlib.sha256(source).hexdigest()
                )

    def test_failed_authoring_build_retains_published_utf8_lock_bytes(self):
        project = self.root / "customer-model"
        project.mkdir()
        (project / "build_adapter.py").write_text(
            "def create_model(config, assets): return object()\n"
            "def create_cases(config, assets): return [(1,)]\n"
        )
        (project / "weights.pt").write_bytes(b"opaque checkpoint")
        (project / "model-build.json").write_text(
            json.dumps(
                {
                    "format_version": 2,
                    "name": "customer-model",
                    "version": "0.1.0",
                    "adapter": "build_adapter.py",
                    "source": [],
                    "checkpoint": "weights.pt",
                    "executor": "local",
                    "runtime": sys.executable,
                    "device": "cpu",
                    "backends": ["aoti"],
                }
            )
        )

        def failed_worker(plan, snapshot, operation):
            plan["output"].mkdir()
            return 7

        def build(output):
            stdout = io.StringIO()
            with (
                mock.patch.object(authoring, "_run", side_effect=failed_worker),
                contextlib.redirect_stdout(stdout),
            ):
                code = cli.main(
                    ["build", str(project), "--output", str(output), "--json"]
                )
            self.assertEqual(code, 1, stdout.getvalue())
            self.assertEqual(json.loads(stdout.getvalue())["status"], "failed")

        build(self.root / "initial")
        lock = project / "model-build.lock.json"
        document = json.loads(lock.read_bytes())
        document["entries"]["other-project"] = {"description": "训练模型"}
        for newline in ("\n", "\r\n"):
            with self.subTest(newline=repr(newline)):
                expected = (
                    (json.dumps(document, ensure_ascii=False, indent=2) + "\n")
                    .replace("\n", newline)
                    .encode("utf-8")
                )
                lock.write_bytes(expected)
                output = self.root / ("crlf" if newline == "\r\n" else "lf")
                build(output)
                self.assertEqual(lock.read_bytes(), expected)
                self.assertEqual(
                    (output / "project" / lock.name).read_bytes(), expected
                )


if __name__ == "__main__":
    unittest.main()
