"""Start the native service stack from the compiled release already in an image."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import sys
import uuid

from . import bundle, service
from .common import read_json, require


def launch(
    *, bundle_root=Path("/opt/physicsnemo-release"), artifact_root=None, output=None
):
    bundle_root = Path(bundle_root).resolve()
    seal = bundle.verify(bundle_root, expected=os.environ.get("PNMS_BUNDLE_DIGEST"))
    descriptor = read_json(bundle_root / "bundle.json")
    source = descriptor["source_sha"]
    for key in ("SOURCE_SHA", "PNMS_SOURCE_SHA"):
        require(
            not os.environ.get(key) or os.environ[key] == source,
            f"{key} differs from the sealed release source",
        )
    artifact_root = Path(
        artifact_root or os.environ.get("PNMS_OUTPUT_ROOT", "/outputs")
    ).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    output = Path(output or artifact_root / ".pnms" / ("service-" + uuid.uuid4().hex))
    project = {
        "root": bundle_root,
        "name": descriptor["name"],
        "runtime": "cpp",
        "serve": {"artifact_root": str(artifact_root)},
    }
    built = {
        "root": str(bundle_root.parent),
        "bundle": bundle_root.name,
        "bundle_digest": seal["bundle_digest"],
        "source_sha": source,
        "environment": {**descriptor["environment"], "python": sys.executable},
    }
    return service.serve(
        project,
        built,
        repository=Path("/opt/pnms/source"),
        output=output,
        port=int(os.environ.get("PORT", os.environ.get("SERVER_PORT", "8080"))),
        dev=True,
        skip_build=True,
    )


def _terminate(_signal, _frame):
    raise KeyboardInterrupt()


def main():
    signal.signal(signal.SIGTERM, _terminate)
    try:
        launch()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        print(f"Native image startup failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
