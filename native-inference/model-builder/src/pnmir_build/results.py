"""Keep command results separate from Python and native process diagnostics."""

from contextlib import contextmanager, redirect_stdout
import ctypes
import os
import shutil
import sys
import tempfile


def _native_flusher():
    """Return a flusher for the process C runtime, or reject unsupported capture.

    POSIX libraries share this stdio runtime. Separately buffered C++ streams
    and background writers still need to drain before their operation returns.
    Windows extensions may use distinct CRTs, so flushing one cannot establish
    the same contract there.
    """
    if os.name != "posix":
        raise RuntimeError("Capturing native stdout requires a POSIX C runtime.")
    try:
        flush = ctypes.CDLL(None, use_errno=True).fflush
    except (OSError, AttributeError) as exc:
        raise RuntimeError(
            "Cannot capture native stdout: the process C runtime must expose fflush."
        ) from exc
    flush.argtypes = (ctypes.c_void_p,)
    flush.restype = ctypes.c_int

    def flush_streams():
        if flush(None) != 0:
            raise OSError(ctypes.get_errno(), "Cannot flush native stdout diagnostics.")

    return flush_streams


@contextmanager
def capture_stdout(enabled, result):
    if not enabled:
        yield
        return
    flush_native = _native_flusher()
    # Redirect the descriptor as well as Python stdout: framework extensions and
    # Docker children may write directly to fd 1.
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as log:
        sys.stdout.flush()
        original = os.dup(1)
        try:
            os.dup2(log.fileno(), 1)
            with redirect_stdout(log):
                try:
                    yield
                finally:
                    try:
                        log.flush()
                    finally:
                        # Flush C buffers only while redirected, including any
                        # inherited buffered logs. Restoring fd 1 first would
                        # leak them after the JSON result at process shutdown.
                        flush_native()
        finally:
            os.dup2(original, 1)
            os.close(original)
            log.seek(0)
            shutil.copyfileobj(log, sys.stderr)
            output = result.get("output")
            if output:
                from pathlib import Path

                root = Path(output)
                if root.is_dir() and not root.is_symlink():
                    log.seek(0)
                    with (root / "frontend.log").open(
                        "x", encoding="utf-8"
                    ) as retained:
                        shutil.copyfileobj(log, retained)
