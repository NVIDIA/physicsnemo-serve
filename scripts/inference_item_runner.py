#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one plugin request in a fresh Python interpreter."""

from __future__ import annotations

import contextlib
import json
import os
import sys
import traceback
from typing import Any

from inference_worker import WorkflowExecutor


def _redis_client() -> Any | None:
    redis_url = os.environ.get("REDIS_URL", "").strip()
    if not redis_url:
        return None
    import redis

    return redis.from_url(redis_url)


def main() -> int:
    redis_client = None
    executor = None
    try:
        request = json.load(sys.stdin)
        if not isinstance(request, dict):
            raise TypeError("Plugin item request must be a JSON object")

        workflow_name = str(request.get("workflow_name") or "").strip()
        run_id = str(request.get("run_id") or "").strip()
        parameters = request.get("parameters")
        payload = request.get("payload")
        if not workflow_name or not run_id:
            raise ValueError("Plugin item request requires workflow_name and run_id")
        if not isinstance(parameters, dict) or not isinstance(payload, dict):
            raise TypeError("Plugin item parameters and payload must be JSON objects")

        redis_client = _redis_client()
        executor = WorkflowExecutor(redis_client)
        # Keep stdout machine-readable even when plugin code prints diagnostics.
        with contextlib.redirect_stdout(sys.stderr):
            result = executor.execute(
                workflow_name,
                run_id,
                parameters,
                payload=payload,
            )
        json.dump(result, sys.stdout)
        sys.stdout.write("\n")
        return 0
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        if executor is not None:
            with contextlib.redirect_stdout(sys.stderr):
                executor.close()
        if redis_client is not None:
            close = getattr(redis_client, "close", None)
            if callable(close):
                close()


if __name__ == "__main__":
    raise SystemExit(main())
