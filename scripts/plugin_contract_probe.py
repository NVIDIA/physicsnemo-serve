#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plugin_runtime import (  # noqa: E402
    get_workflow_schema_source,
    load_plugin_manifest,
    load_plugin_module,
)
from plugin_sdk import (  # noqa: E402
    workflow_form_schema,
    workflow_request_schema,
    workflow_result_schema,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=("Probe request and result schemas from a plugin workflow object.")
    )
    parser.add_argument("--plugin-root", required=True)
    args = parser.parse_args()

    try:
        plugin_root = Path(args.plugin_root).expanduser().resolve()
        manifest = load_plugin_manifest(plugin_root / "plugin.yaml")
        workflow_id = str(manifest.get("metadata", {}).get("id") or "").strip()
        if not workflow_id:
            raise ValueError("Plugin manifest is missing metadata.id")

        runtime = manifest.get("runtime", {})
        entrypoint_name = runtime.get("entrypoint")
        if not entrypoint_name:
            raise ValueError(
                f"Plugin workflow '{workflow_id}' is missing runtime.entrypoint"
            )

        module = load_plugin_module(
            workflow_id,
            plugin_root / str(entrypoint_name),
            module_prefix="physicsnemo_serve_contract_probe",
        )
        workflow = get_workflow_schema_source(module, workflow_id)

        payload = {
            "request_schema": workflow_request_schema(workflow),
            "form_schema": workflow_form_schema(workflow),
            "result_schema": workflow_result_schema(workflow),
        }
        json.dump(payload, sys.stdout)
        sys.stdout.write("\n")
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
