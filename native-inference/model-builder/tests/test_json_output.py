"""JSON command results survive Python, subprocess and buffered native logs."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

BUILDER = Path(__file__).resolve().parents[1] / "src"


class JsonOutputTests(unittest.TestCase):
    def run_capture(self, body):
        script = (
            "import ctypes, json, os, sys\n"
            f"sys.path.insert(0, {str(BUILDER)!r})\n"
            "from pnmir_build.results import capture_stdout\n"
        ) + textwrap.dedent(body)
        run = subprocess.run(
            [sys.executable, "-S", "-c", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        return run

    def one_result(self, run):
        try:
            return json.loads(run.stdout)
        except ValueError:
            self.fail(f"Expected exactly one JSON result, received {run.stdout!r}")

    def test_buffered_c_stdio_is_flushed_before_stdout_is_restored(self):
        run = self.run_capture("""
            with capture_stdout(True, {}):
                ctypes.CDLL(None).printf(b'buffered C diagnostic\\n')
            print(json.dumps({'status': 'complete'}))
        """)
        self.assertEqual(self.one_result(run)["status"], "complete")
        self.assertIn("buffered C diagnostic", run.stderr)

    def test_preexisting_c_buffers_are_flushed_into_capture(self):
        run = self.run_capture("""
            ctypes.CDLL(None).printf(b'previous native diagnostic\\n')
            with capture_stdout(True, {}):
                pass
            print(json.dumps({'status': 'complete'}))
        """)
        self.assertEqual(self.one_result(run)["status"], "complete")
        self.assertIn("previous native diagnostic", run.stderr)

    def test_python_direct_fd_and_child_logs_are_retained_with_invalid_utf8(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.run_capture(f"""
                import subprocess
                result = {{'output': {directory!r}}}
                with capture_stdout(True, result):
                    print('Python diagnostic')
                    os.write(1, b'fd diagnostic \\xff\\n')
                    subprocess.run([sys.executable, '-S', '-c', "print('child diagnostic')"], check=True)
                    ctypes.CDLL(None).printf(b'buffered diagnostic\\n')
                print(json.dumps({{'status': 'complete'}}))
            """)
            self.assertEqual(self.one_result(run)["status"], "complete")
            retained = (Path(directory) / "frontend.log").read_text()
            for message in (
                "Python diagnostic",
                "fd diagnostic \ufffd",
                "child diagnostic",
                "buffered diagnostic",
            ):
                self.assertIn(message, run.stderr)
                self.assertIn(message, retained)

    def test_body_exception_still_drains_native_logs_and_restores_stdout(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.run_capture(f"""
                try:
                    with capture_stdout(True, {{'output': {directory!r}}}):
                        print('Python failure diagnostic')
                        ctypes.CDLL(None).printf(b'buffered failure diagnostic\\n')
                        raise ValueError('model failed')
                except ValueError as error:
                    print(json.dumps({{'status': 'failed', 'message': str(error)}}))
            """)
            self.assertEqual(self.one_result(run)["message"], "model failed")
            self.assertIn("buffered failure diagnostic", run.stderr)
            self.assertIn(
                "buffered failure diagnostic",
                (Path(directory) / "frontend.log").read_text(),
            )

    def test_unavailable_native_flush_fails_before_running_the_command(self):
        run = self.run_capture("""
            from unittest import mock
            body_ran = False
            error = None
            try:
                with mock.patch('ctypes.CDLL', side_effect=OSError('no process C runtime')):
                    with capture_stdout(True, {}):
                        body_ran = True
            except RuntimeError as failure:
                error = str(failure)
            print(json.dumps({'body_ran': body_ran, 'error': error}))
        """)
        result = self.one_result(run)
        self.assertFalse(result["body_ran"])
        self.assertIn("native stdout", result["error"])

    def test_disabled_capture_preserves_normal_stdout(self):
        run = self.run_capture("""
            with capture_stdout(False, {}):
                print('regular stdout')
                ctypes.CDLL(None).printf(b'regular native stdout\\n')
        """)
        self.assertIn("regular stdout", run.stdout)
        self.assertIn("regular native stdout", run.stdout)
        self.assertEqual(run.stderr, "")

    def test_existing_frontend_log_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            retained = Path(directory) / "frontend.log"
            retained.write_text("preserve previous log")
            run = self.run_capture(f"""
                try:
                    with capture_stdout(True, {{'output': {directory!r}}}):
                        print('new diagnostic')
                except FileExistsError:
                    print(json.dumps({{'status': 'failed'}}))
            """)
            self.assertEqual(self.one_result(run)["status"], "failed")
            self.assertEqual(retained.read_text(), "preserve previous log")


if __name__ == "__main__":
    unittest.main()
