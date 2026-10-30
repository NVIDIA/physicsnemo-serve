"""Exercise the production CLI startup policy without a native engine or CUDA."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]


class CliStartupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.name != "posix":
            raise unittest.SkipTest("the production CLI requires POSIX setenv")
        compiler = shutil.which("c++")
        if compiler is None:
            raise unittest.SkipTest("a C++ compiler is required for CLI checks")
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        build_root = Path(temporary.name)
        stub = build_root / "runner.cpp"
        # Replace only the GPU-backed Runner boundary. Compile the production
        # main.cpp unchanged so argument parsing, startup ordering, and report
        # publication all execute through the real CLI control flow.
        stub.write_text(
            r"""
#include "transolver_workflow/runner.hpp"
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace transolver_workflow {
namespace {
void record(const std::string& event) {
  std::ofstream stream(std::getenv("PNMS_CLI_TEST_EVENTS"), std::ios::app);
  stream << event << '\n';
}
}

class Runner::Impl {
 public:
  explicit Impl(RunnerConfig value) : config(std::move(value)) {}
  RunnerConfig config;
};

Runner::Runner(RunnerConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}
Runner::~Runner() = default;

std::string Runner::self_test(const CancellationToken&) {
  const auto& fixture = impl_->config.startup_fixture_path;
  record("self_test:" + fixture);
  if (!fixture.empty()) {
    std::ifstream stream(fixture);
    std::string value;
    stream >> value;
    if (!stream || value != "match") {
      throw std::runtime_error("startup fixture mismatch");
    }
  }
  return R"({"status":"passed"})";
}

RunnerResult Runner::run(const RunnerRequest& request, const CancellationToken&) {
  record("run");
  std::ofstream(request.standardized_output) << "published fields\n";
  std::ofstream(request.metadata_path) << "published metadata\n";
  return {};
}
}
"""
        )
        cls.executable = build_root / "physicsnemo-transolver"
        subprocess.run(
            [
                compiler,
                "-std=c++20",
                "-I",
                str(WORKFLOW_ROOT / "include"),
                str(WORKFLOW_ROOT / "src" / "main.cpp"),
                str(stub),
                "-o",
                str(cls.executable),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.events = self.root / "events.txt"
        self.output = self.root / "fields.f32"
        self.metadata = self.root / "fields.f32.json"
        self.report = self.root / "startup.json"
        self.fixture = self.root / "fixture.json"
        self.fixture.write_text("match\n")

    def run_cli(self, *options, startup_only=False):
        command = [
            str(self.executable),
            "--package",
            "model.pt2",
            "--domain",
            "surface",
        ]
        if startup_only:
            command.extend(["--startup-only", "true"])
        else:
            command.extend(
                [
                    "--mesh",
                    "mesh.vtp",
                    "--stl",
                    "body.stl",
                    "--standardized-output",
                    str(self.output),
                ]
            )
        return subprocess.run(
            [*command, *map(str, options)],
            cwd=self.root,
            env={**os.environ, "PNMS_CLI_TEST_EVENTS": str(self.events)},
            capture_output=True,
            text=True,
        )

    def recorded_events(self):
        return self.events.read_text().splitlines() if self.events.exists() else []

    def test_bad_fixture_without_report_fails_before_running(self):
        self.fixture.write_text("mismatch\n")
        result = self.run_cli("--startup-fixture", self.fixture)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("startup fixture mismatch", result.stderr)
        self.assertEqual(self.recorded_events(), [f"self_test:{self.fixture}"])
        self.assertFalse(self.output.exists())
        self.assertFalse(self.metadata.exists())
        self.assertFalse(self.report.exists())

    def test_valid_fixture_without_report_checks_before_running(self):
        result = self.run_cli("--startup-fixture", self.fixture)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.recorded_events(), [f"self_test:{self.fixture}", "run"])
        self.assertEqual(self.output.read_text(), "published fields\n")
        self.assertTrue(self.metadata.is_file())
        self.assertFalse(self.report.exists())

    def test_legacy_run_without_fixture_or_report(self):
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.recorded_events(), ["run"])
        self.assertTrue(self.output.is_file())

    def test_report_option_checks_and_writes_report_before_running(self):
        result = self.run_cli("--startup-report", self.report)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.recorded_events(), ["self_test:", "run"])
        self.assertEqual(json.loads(self.report.read_text()), {"status": "passed"})
        self.assertTrue(self.output.is_file())

    def test_startup_only_with_report_checks_without_running(self):
        result = self.run_cli(
            "--startup-fixture",
            self.fixture,
            "--startup-report",
            self.report,
            startup_only=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.recorded_events(), [f"self_test:{self.fixture}"])
        self.assertEqual(json.loads(self.report.read_text()), {"status": "passed"})
        self.assertFalse(self.output.exists())
        self.assertFalse(self.metadata.exists())

    def test_startup_only_still_requires_report_with_fixture(self):
        result = self.run_cli("--startup-fixture", self.fixture, startup_only=True)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("--startup-only requires --startup-report", result.stderr)
        self.assertEqual(self.recorded_events(), [])


if __name__ == "__main__":
    unittest.main()
