# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend runner abstraction for parity workflows."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .canonical_reader import build_manifest_from_dataset
from .workflow_catalog import WorkflowCatalog, create_default_workflow_catalog

BackendRunner = Callable[[dict[str, Any], Path], None]

SUPPORTED_BACKENDS = {"py_sync", "py_async", "rust"}


class BackendRunnerRegistry:
    """Registry for backend run callables."""

    def __init__(self) -> None:
        self._runners: dict[str, BackendRunner] = {}

    def register(self, backend_kind: str, runner: BackendRunner) -> None:
        """Register callable for backend kind."""
        if backend_kind not in SUPPORTED_BACKENDS:
            raise ValueError(f"unsupported backend_kind: {backend_kind}")
        self._runners[backend_kind] = runner

    def has_runner(self, backend_kind: str) -> bool:
        """Return True when backend has registered runner."""
        return backend_kind in self._runners

    def run(
        self, backend_kind: str, case_spec: dict[str, Any], dataset_path: Path
    ) -> None:
        """Execute runner for backend kind."""
        if backend_kind not in self._runners:
            raise RuntimeError(f"no runner registered for backend: {backend_kind}")
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        self._runners[backend_kind](case_spec, dataset_path)


def create_default_backend_runner_registry(
    workflow_catalog: WorkflowCatalog | None = None,
) -> BackendRunnerRegistry:
    """Create default backend registry wired to workflow catalog adapters."""
    catalog = (
        workflow_catalog
        if workflow_catalog is not None
        else create_default_workflow_catalog()
    )
    registry = BackendRunnerRegistry()
    for backend_kind in sorted(SUPPORTED_BACKENDS):
        registry.register(
            backend_kind,
            lambda case_spec, dataset_path, bk=backend_kind: catalog.run_with_backend(
                bk, case_spec, dataset_path
            ),
        )
    return registry


def run_backend_and_collect_manifest(
    *,
    registry: BackendRunnerRegistry,
    backend_kind: str,
    case_spec: dict[str, Any],
    dataset_path: str | Path,
    generated_by_backend: str,
) -> dict[str, Any]:
    """Run backend and return canonical manifest from produced dataset."""
    path = Path(dataset_path)
    registry.run(backend_kind, case_spec, path)
    return build_manifest_from_dataset(
        dataset_path=path,
        case_spec=case_spec,
        generated_by_backend=generated_by_backend,
    )


def run_backend_with_default_registry_and_collect_manifest(
    *,
    backend_kind: str,
    case_spec: dict[str, Any],
    dataset_path: str | Path,
    generated_by_backend: str,
    workflow_catalog: WorkflowCatalog | None = None,
) -> dict[str, Any]:
    """Run backend with default registry wiring and collect canonical manifest."""
    registry = create_default_backend_runner_registry(workflow_catalog=workflow_catalog)
    return run_backend_and_collect_manifest(
        registry=registry,
        backend_kind=backend_kind,
        case_spec=case_spec,
        dataset_path=dataset_path,
        generated_by_backend=generated_by_backend,
    )
