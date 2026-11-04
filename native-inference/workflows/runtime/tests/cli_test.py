"""Exercise the standalone runner against an installed CPU mock fixture."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


RUNNER = None
CONFIG = None


class StandaloneRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if RUNNER is None or CONFIG is None:
            raise unittest.SkipTest("run this script with --runner and --config")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.configuration = json.loads(CONFIG.read_text())
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps(self.configuration))
        self.request = self.root / "request.json"
        self.values = [0, 1, -1, 4]
        self.request.write_text(
            json.dumps(
                {
                    "request_id": "standalone-cpu-case",
                    "parameters": {"values": self.values},
                    "output_dir": str(self.root / "attempt"),
                    "inputs": {},
                }
            )
        )

    def invoke(self, *arguments, expected=0):
        result = subprocess.run(
            [str(RUNNER), *map(str, arguments)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def test_help(self):
        result = self.invoke("--help")
        self.assertIn("pnms-native-run", result.stdout)
        self.assertEqual(result.stderr, "")

    def test_self_test(self):
        result = self.invoke("--config", self.config, "--self-test")
        self.assertEqual(json.loads(result.stdout)["status"], "passed")

    def test_request_to_stdout(self):
        result = self.invoke("--config", self.config, "--request", self.request)
        document = json.loads(result.stdout)
        self.assertEqual(document["payload"], {"values": self.values})
        self.assertEqual(document["metadata"]["session_load_count"], 1)

    def test_request_and_startup_reports(self):
        output = self.root / "result.json"
        startup = self.root / "startup.json"
        result = self.invoke(
            "--config",
            self.config,
            "--request",
            self.request,
            "--output",
            output,
            "--startup-report",
            startup,
        )
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            json.loads(output.read_text())["payload"], {"values": self.values}
        )
        self.assertEqual(json.loads(startup.read_text())["status"], "passed")

    def test_request_and_self_test_modes_are_exclusive(self):
        result = self.invoke(
            "--config",
            self.config,
            "--request",
            self.request,
            "--self-test",
            expected=2,
        )
        self.assertEqual(result.stdout, "")
        self.assertIn("usage:", result.stderr)

    def test_missing_config(self):
        result = self.invoke(
            "--config", self.root / "missing.json", "--self-test", expected=1
        )
        self.assertEqual(result.stdout, "")
        self.assertIn("cannot read JSON file", result.stderr)

    def test_startup_mismatch_prevents_request_and_report_publication(self):
        self.configuration["models"]["primary"]["self_test"]["expected_outputs"][
            "output"
        ][0] = 99
        self.config.write_text(json.dumps(self.configuration))
        output = self.root / "result.json"
        startup = self.root / "startup.json"
        result = self.invoke(
            "--config",
            self.config,
            "--request",
            self.request,
            "--output",
            output,
            "--startup-report",
            startup,
            expected=1,
        )
        self.assertEqual(result.stdout, "")
        self.assertIn("self_test output mismatch", result.stderr)
        self.assertFalse(output.exists())
        self.assertFalse(startup.exists())
        self.assertFalse((self.root / "attempt").exists())

    def test_existing_result_is_preserved(self):
        output = self.root / "result.json"
        output.write_text("previous result\n")
        result = self.invoke(
            "--config",
            self.config,
            "--request",
            self.request,
            "--output",
            output,
            expected=1,
        )
        self.assertEqual(result.stdout, "")
        self.assertIn("already exists", result.stderr)
        self.assertEqual(output.read_text(), "previous result\n")

    def test_existing_startup_report_is_preserved(self):
        startup = self.root / "startup.json"
        startup.write_text("previous startup report\n")
        result = self.invoke(
            "--config",
            self.config,
            "--self-test",
            "--startup-report",
            startup,
            expected=1,
        )
        self.assertEqual(result.stdout, "")
        self.assertIn("already exists", result.stderr)
        self.assertEqual(startup.read_text(), "previous startup report\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    selected, remaining = parser.parse_known_args()
    RUNNER = selected.runner.resolve()
    CONFIG = selected.config.resolve()
    unittest.main(argv=[sys.argv[0], *remaining])
