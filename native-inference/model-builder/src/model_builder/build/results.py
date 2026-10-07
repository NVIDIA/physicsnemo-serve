"""Keep command results separate from Python and native process diagnostics."""

from contextlib import contextmanager, redirect_stdout
import ctypes
import os
import shutil
import sys
import tempfile


def _native_flusher():
    """Return a flusher for the process C runtime, or reject unsupported capture.

    POSIX libraries share the process runtime. Supported Windows builds use
    the shared UCRT (Python and MSVC /MD). Separately buffered C++ streams,
    privately linked CRTs and background writers must drain before returning.
    """
    if os.name not in ("posix", "nt"):
        raise RuntimeError("Capturing native stdout requires POSIX or Windows UCRT.")
    try:
        flush = ctypes.CDLL(
            "ucrtbase" if os.name == "nt" else None, use_errno=True
        ).fflush
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


def _windows_stdout_setter():
    if os.name != "nt":
        return None
    import msvcrt
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    set_handle = kernel.SetStdHandle
    set_handle.argtypes = (wintypes.DWORD, wintypes.HANDLE)
    set_handle.restype = wintypes.BOOL

    def redirect(fd):
        # dup2 updates the CRT descriptor table. Win32 writers and subprocess
        # inheritance also use the separate process standard-handle table.
        if not set_handle(-11, msvcrt.get_osfhandle(fd)):
            raise ctypes.WinError(ctypes.get_last_error())

    return redirect


@contextmanager
def capture_stdout(enabled, result):
    if not enabled:
        yield
        return
    flush_native = _native_flusher()
    set_windows_stdout = _windows_stdout_setter()
    # Redirect the descriptor as well as Python stdout: framework extensions and
    # Docker children may write directly to fd 1.
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as log:
        sys.stdout.flush()
        original = os.dup(1)
        try:
            os.dup2(log.fileno(), 1)
            if set_windows_stdout:
                set_windows_stdout(1)
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
            if set_windows_stdout:
                set_windows_stdout(1)
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
