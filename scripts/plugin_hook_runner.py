#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PYTHON_DIR = REPO_ROOT / "python"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from plugin_runtime import (  # noqa: E402
    build_context,
    build_postprocess_context,
    build_prepare_context,
    build_prior_result,
    build_raw_request,
    load_plugin_module,
    merge_registered_outputs_into_result,
    resolve_phase_hook,
    resolve_plugin_manifest,
    serialize_postprocess_result,
    serialize_prepare_result,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Invoke a plugin hook with a JSON payload from stdin."
    )
    parser.add_argument(
        "--phase", required=True, choices=["prepare", "execute", "postprocess"]
    )
    args = parser.parse_args()

    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            raise ValueError("Hook runner payload must be a JSON object")

        workflow_id = str(payload.get("workflow_id") or "").strip()
        if not workflow_id:
            raise ValueError("Hook runner payload must include workflow_id")

        plugin_root, manifest = resolve_plugin_manifest(workflow_id)
        runtime = manifest.get("runtime", {})
        entrypoint_name = runtime.get("entrypoint") or payload.get("runtime", {}).get(
            "entrypoint"
        )
        if not entrypoint_name:
            raise ValueError(
                f"Plugin workflow '{workflow_id}' is missing runtime.entrypoint"
            )

        module = load_plugin_module(
            workflow_id,
            plugin_root / entrypoint_name,
            module_prefix="physicsnemo_serve_hook_runner",
        )
        hook = resolve_phase_hook(module, workflow_id, args.phase)

        raw_context = build_context(payload)
        postprocess_ctx = (
            build_postprocess_context(payload) if args.phase == "postprocess" else None
        )
        if args.phase == "prepare" and _supports_explicit_contract(hook):
            result = serialize_prepare_result(
                hook(build_raw_request(payload), build_prepare_context(payload))
            )
        elif args.phase == "postprocess" and _supports_explicit_contract(hook):
            result = serialize_postprocess_result(
                hook(build_prior_result(payload), postprocess_ctx)
            )
        else:
            result = hook(raw_context)
            if result is None:
                result = {}
            if args.phase == "prepare":
                result = serialize_prepare_result(result)
            elif args.phase == "postprocess":
                result = serialize_postprocess_result(result)
            elif not isinstance(result, dict):
                raise TypeError(
                    f"Plugin workflow '{workflow_id}' hook '{args.phase}' returned "
                    f"{type(result).__name__}, expected dict"
                )

        if args.phase == "postprocess" and postprocess_ctx is not None:
            result = merge_registered_outputs_into_result(
                result, postprocess_ctx.outputs
            )

        json.dump(result, sys.stdout)
        sys.stdout.write("\n")
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _supports_explicit_contract(hook) -> bool:
    try:
        signature = inspect.signature(hook)
    except (TypeError, ValueError):
        return False

    positional_params = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    ):
        return True
    return len(positional_params) >= 2


if __name__ == "__main__":
    raise SystemExit(main())
