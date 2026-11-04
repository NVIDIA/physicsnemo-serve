"""Claim fresh mounted storage before starting a managed native container.

This startup-only standard-library helper deliberately does not resume an old
claim, including a claim by the same operation. Whole-container replacement is
outside the fresh deployment contract; worker-process restarts stay supervised.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def claim(root, *, operation_id, deployment_id, image):
    _require(
        isinstance(operation_id, str) and re.fullmatch(r"[0-9a-f]{32}", operation_id),
        "invalid deployment operation",
    )
    _require(
        isinstance(deployment_id, str)
        and re.fullmatch(r"[a-z][a-z0-9-]{0,48}", deployment_id),
        "invalid deployment identity",
    )
    _require(
        isinstance(image, str) and re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image),
        "immutable deployment image required",
    )
    root = Path(root)
    _require(
        root.is_absolute() and not root.is_symlink() and root.is_dir(),
        "fresh mounted artifact directory required",
    )
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        _require(
            not os.listdir(root_fd),
            "artifact scope is nonempty or already used; fresh deployment required",
        )
        try:
            os.mkdir(".native-deployment", mode=0o700, dir_fd=root_fd)
        except FileExistsError:
            raise ValueError(
                "artifact scope is already claimed; fresh deployment required"
            ) from None
        _require(
            os.listdir(root_fd) == [".native-deployment"],
            "artifact scope changed while claiming fresh storage",
        )
        claim_fd = os.open(
            ".native-deployment",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=root_fd,
        )
        try:
            record = {
                "schema_version": 1,
                "status": "claimed",
                "operation_id": operation_id,
                "deployment_id": deployment_id,
                "image": image,
            }
            fd = os.open(
                "owner.json",
                os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                0o600,
                dir_fd=claim_fd,
            )
            with os.fdopen(fd, "w") as stream:
                json.dump(record, stream, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.fsync(claim_fd)
        finally:
            os.close(claim_fd)
        os.fsync(root_fd)
        return record
    finally:
        os.close(root_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", required=True)
    parser.add_argument("--deployment-id", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    _require(
        os.environ.get("RELEASE_IMAGE_DIGEST") == args.image
        and os.environ.get("NATIVE_DEPLOYMENT_ID") == args.deployment_id
        and os.environ.get("NATIVE_DEPLOYMENT_OPERATION_ID") == args.operation
        and os.environ.get("NATIVE_WORKER_SLOT") == "0",
        "fresh deployment environment identity differs",
    )
    record = claim(
        "/outputs",
        operation_id=args.operation,
        deployment_id=args.deployment_id,
        image=args.image,
    )
    print(json.dumps(record, sort_keys=True), flush=True)
    os.execv("/app/scripts/entrypoint.sh", ["/app/scripts/entrypoint.sh"])


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        print("native fresh deployment refused: " + str(error), file=sys.stderr)
        raise SystemExit(1) from None
