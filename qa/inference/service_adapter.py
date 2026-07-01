# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service adapter facade for Python and Rust inference services.

Abstracts path, input JSON, and output JSON differences so that the same
test code works against either backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ServiceAdapter(ABC):
    """Unified interface for interacting with an inference service."""

    # ---- Paths ----

    @abstractmethod
    def health_url(self) -> str: ...

    @abstractmethod
    def list_workflows_url(self) -> str: ...

    @abstractmethod
    def workflow_schema_url(self, workflow_name: str) -> str: ...

    @abstractmethod
    def submit_url(self, workflow_name: str) -> str: ...

    @abstractmethod
    def status_url(self, workflow_name: str, execution_id: str) -> str: ...

    @abstractmethod
    def results_url(self, workflow_name: str, execution_id: str) -> str: ...

    @abstractmethod
    def result_file_url(
        self, workflow_name: str, execution_id: str, file_path: str
    ) -> str: ...

    # ---- Input formatting ----

    @abstractmethod
    def format_submit_body(self, params: dict) -> dict: ...

    # ---- Output parsing ----

    @abstractmethod
    def parse_health_response(self, response) -> dict:
        """Returns normalized: {"status": str, "timestamp": str|None}"""
        ...

    @abstractmethod
    def parse_list_workflows_response(self, data: dict) -> list[str]:
        """Returns list of workflow name strings."""
        ...

    @abstractmethod
    def parse_submit_response(self, data: dict) -> dict:
        """Returns normalized: {"execution_id": str, "workflow_name": str, "status": str}"""
        ...

    @abstractmethod
    def parse_status_response(self, data: dict) -> dict:
        """Returns normalized: {"execution_id": str, "status": str, "progress": dict|None,
        "position": int|None, "execution_time_seconds": float|None}"""
        ...

    @abstractmethod
    def parse_results_response(self, data: dict) -> dict:
        """Returns normalized: {"output_files": [{"path": str, ...}]}"""
        ...


class PythonAdapter(ServiceAdapter):
    """Adapter for the Python earth2studio serve API."""

    def health_url(self) -> str:
        return "/health"

    def list_workflows_url(self) -> str:
        return "/v1/infer/workflows"

    def workflow_schema_url(self, workflow_name: str) -> str:
        return f"/v1/infer/workflows/{workflow_name}/schema"

    def submit_url(self, workflow_name: str) -> str:
        return f"/v1/infer/{workflow_name}"

    def status_url(self, workflow_name: str, execution_id: str) -> str:
        return f"/v1/infer/{workflow_name}/{execution_id}/status"

    def results_url(self, workflow_name: str, execution_id: str) -> str:
        return f"/v1/infer/{workflow_name}/{execution_id}/results"

    def result_file_url(
        self, workflow_name: str, execution_id: str, file_path: str
    ) -> str:
        return f"/v1/infer/{workflow_name}/{execution_id}/results/{file_path}"

    def format_submit_body(self, params: dict) -> dict:
        return {"parameters": params}

    def parse_health_response(self, response) -> dict:
        data = response.json()
        return {"status": data.get("status"), "timestamp": data.get("timestamp")}

    def parse_list_workflows_response(self, data: dict) -> list[str]:
        workflows = data.get("workflows", {})
        # Python service returns {"workflows": {"name": "description", ...}}
        if isinstance(workflows, dict):
            return list(workflows.keys())
        return workflows

    def parse_submit_response(self, data: dict) -> dict:
        return {
            "execution_id": data["execution_id"],
            "workflow_name": data.get("workflow_name"),
            "status": data.get("status"),
        }

    def parse_status_response(self, data: dict) -> dict:
        return {
            "execution_id": data.get("execution_id"),
            "status": data["status"],
            "progress": data.get("progress"),
            "position": data.get("position"),
            "execution_time_seconds": data.get("execution_time_seconds"),
        }

    def parse_results_response(self, data: dict) -> dict:
        return {"output_files": data.get("output_files", [])}


class RustAdapter(ServiceAdapter):
    """Adapter for the Rust PhysicsNeMo Serve inference API."""

    # Python workflow name → PhysicsNeMo Serve plugin name
    _WORKFLOW_MAP = {
        "deterministic_workflow": "e2s-deterministic",
        "deterministic_fcn_workflow": "e2s-deterministic-fcn",
        "diagnostic_workflow": "e2s-diagnostic",
        "ensemble_workflow": "e2s-ensemble",
        "deterministic_earth2_workflow": "e2s-deterministic-earth2",
        "stormcast_fcn3_workflow": "e2s-stormcast-fcn3",
        "example_user_workflow": "e2s-example-user",
    }
    # Reverse mapping: PhysicsNeMo Serve plugin name → Python workflow name
    _WORKFLOW_MAP_REVERSE = {v: k for k, v in _WORKFLOW_MAP.items()}

    def _resolve_workflow(self, workflow_name: str) -> str:
        """Translate Python workflow names to PhysicsNeMo Serve plugin names."""
        return self._WORKFLOW_MAP.get(workflow_name, workflow_name)

    def _reverse_workflow(self, plugin_name: str) -> str:
        """Translate PhysicsNeMo Serve plugin names back to Python workflow names."""
        return self._WORKFLOW_MAP_REVERSE.get(plugin_name, plugin_name)

    def health_url(self) -> str:
        return "/health"

    def list_workflows_url(self) -> str:
        return "/v1/infer/workflows"

    def workflow_schema_url(self, workflow_name: str) -> str:
        name = self._resolve_workflow(workflow_name)
        return f"/v1/infer/{name}/schema"

    def submit_url(self, workflow_name: str) -> str:
        name = self._resolve_workflow(workflow_name)
        return f"/v1/infer/{name}/run"

    def status_url(self, workflow_name: str, execution_id: str) -> str:
        name = self._resolve_workflow(workflow_name)
        return f"/v1/infer/{name}/{execution_id}/status"

    def results_url(self, workflow_name: str, execution_id: str) -> str:
        name = self._resolve_workflow(workflow_name)
        return f"/v1/infer/{name}/{execution_id}/results"

    def result_file_url(
        self, workflow_name: str, execution_id: str, file_path: str
    ) -> str:
        name = self._resolve_workflow(workflow_name)
        return f"/v1/infer/{name}/{execution_id}/results?artifact={file_path}"

    def format_submit_body(self, params: dict) -> dict:
        return {"parameters": params}

    def parse_health_response(self, response) -> dict:
        # Rust /health returns plain text "ok", not JSON
        text = response.text.strip()
        if text == "ok":
            return {"status": "healthy", "timestamp": None}
        # Fall back to trying JSON in case future versions change
        try:
            data = response.json()
            return {"status": data.get("status"), "timestamp": data.get("timestamp")}
        except Exception:
            return {"status": text, "timestamp": None}

    def parse_list_workflows_response(self, data: dict) -> list[str]:
        workflows = data.get("workflows", [])
        # Rust returns objects with a "name" field; extract names
        if workflows and isinstance(workflows[0], dict):
            return [wf["name"] for wf in workflows]
        return workflows

    def parse_submit_response(self, data: dict) -> dict:
        rust_workflow = data.get("workflow", "")
        return {
            "execution_id": data["run_id"],
            "workflow_name": self._reverse_workflow(rust_workflow),
            "status": data.get("status"),
        }

    def parse_status_response(self, data: dict) -> dict:
        progress = data.get("progress")
        if progress is None:
            # Rust stores stage info; synthesize progress from stage/total
            stage = data.get("stage")
            pipeline = data.get("pipeline")
            if stage and pipeline and isinstance(pipeline, list):
                try:
                    current = pipeline.index(stage) + 1
                except ValueError:
                    current = None
                if current is not None:
                    progress = {"current_step": current, "total_steps": len(pipeline)}

        # Normalize Rust status names to match Python conventions
        status = data["status"]
        if status == "succeeded":
            status = "completed"

        return {
            "execution_id": data.get("run_id"),
            "status": status,
            "progress": progress,
            "position": data.get("position"),
            "execution_time_seconds": data.get("execution_time_seconds"),
        }

    def parse_results_response(self, data: dict) -> dict:
        # Rust returns structured envelope: {request, execution, payload}
        # Extract outputs from execution.outputs and map to output_files format
        output_files = []
        execution = data.get("execution", {})
        outputs = execution.get("outputs", [])
        for output in outputs:
            # Use the "name" field as the path identifier (used for artifact download)
            # and include the filesystem path for pattern matching
            name = output.get("name", "")
            storage_path = output.get("storage_path") or output.get("path", "")
            output_files.append(
                {
                    "path": name,
                    "storage_path": storage_path,
                    "media_type": output.get("media_type"),
                    "primary": output.get("primary", False),
                }
            )
        return {"output_files": output_files}


def get_adapter(service_type: str) -> ServiceAdapter:
    """Factory function to get the appropriate adapter."""
    adapters = {
        "python": PythonAdapter,
        "rust": RustAdapter,
    }
    if service_type not in adapters:
        raise ValueError(
            f"Unknown service type '{service_type}'. Must be one of: {list(adapters.keys())}"
        )
    return adapters[service_type]()
