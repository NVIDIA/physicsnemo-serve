"""Check the native workflow's discoverable CLI without loading CUDA or assets."""

import subprocess
import sys

result = subprocess.run([sys.argv[1], "--help"], capture_output=True, text=True)
assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
for option in ("--mesh", "--stl", "--package", "--physical-output", "--metadata"):
    assert option in result.stdout, f"missing documented option: {option}"
