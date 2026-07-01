# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
CLI Interface for Rust Inference Server interaction.

This script is a STANDALONE entry point. It does NOT depend on earth2studio
framework code (e2workflow.py, workflow.py) to function, allowing for a "headless"
python worker that interacts with the Rust server.

It supports:
1. Workflow subclasses (with run(parameters, execution_id) method)
2. Earth2Workflow subclasses (with __call__(io, **params) method)
"""

import argparse
import importlib.util
import inspect
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, get_type_hints

from pydantic import BaseModel, Field, create_model

# Import static model metadata from separate file for easier maintenance
from e2s_tools.model_metadata import (
    DYNAMIC_METADATA_MODELS,
    get_static_metadata,
)

# ============================================================================
# Workflow Type Detection Constants
# ============================================================================


class WorkflowType:
    """Enum-like class for workflow types."""

    WORKFLOW = "workflow"  # Workflow subclass with run() method
    EARTH2_WORKFLOW = (
        "earth2_workflow"  # Earth2Workflow subclass with __call__() method
    )
    CALLABLE_CLASS = "callable_class"  # Generic callable class with __call__() method


# ============================================================================
# Helper Functions from metadata_extractor.py
# ============================================================================


def _split_function_args(args_str: str) -> List[str]:
    """
    Split function arguments by comma, respecting nested parentheses and brackets.
    """
    args = []
    current_arg = []
    depth = 0

    for char in args_str:
        if char in "([{":
            depth += 1
            current_arg.append(char)
        elif char in ")]}":
            depth -= 1
            current_arg.append(char)
        elif char == "," and depth == 0:
            arg = "".join(current_arg).strip()
            if arg:
                args.append(arg)
            current_arg = []
        else:
            current_arg.append(char)

    arg = "".join(current_arg).strip()
    if arg:
        args.append(arg)

    return args


# Known earth2studio run function signatures
RUN_FUNCTION_SIGNATURES = {
    "deterministic": ["time", "nsteps", "model", "data", "io"],
    "ensemble": ["time", "nsteps", "nmembers", "model", "data", "io"],
    "diagnostic": ["time", "nsteps", "model", "data", "io"],
}


def _detect_models_from_source(source: str) -> list[Dict[str, Any]]:
    """Detect ALL model types and modules from workflow source code."""
    models = []
    patterns = [
        (r"from earth2studio\.models\.px import ([^;\n]+)", "px"),
        (r"from earth2studio\.models\.dx import ([^;\n]+)", "dx"),
        (r"from earth2studio\.models\.rd import ([^;\n]+)", "rd"),
    ]

    for pattern, category in patterns:
        matches = re.findall(pattern, source)
        for match in matches:
            imports = [name.strip() for name in match.split(",")]
            for model_name in imports:
                model_name = model_name.split("#")[0].strip()
                if model_name and model_name[0].isupper():
                    models.append(
                        {
                            "model_name": model_name,
                            "category": category,
                            "module": f"earth2studio.models.{category}",
                        }
                    )
    return models


def _trace_workflow_to_model_mappings(wf_mod, func_name: str) -> Dict[str, str]:
    """Trace how workflow inputs map to earth2studio run function parameters."""
    try:
        func = getattr(wf_mod, func_name, None)
        if not func:
            return {}

        source = inspect.getsource(func)
        mappings = {}

        for run_type, param_names in RUN_FUNCTION_SIGNATURES.items():
            pattern = rf"\brun\.{run_type}\s*\((.*?)\)"
            matches = re.findall(pattern, source, re.DOTALL)
            if not matches:
                continue

            matches = [m for m in matches if m.strip()]
            if not matches:
                continue

            args_str = matches[0]
            args = _split_function_args(args_str)
            data_params = param_names[:3]

            for i, param_name in enumerate(data_params):
                if i >= len(args):
                    break
                arg = args[i].strip()

                if "inputs." in arg:
                    field_match = re.search(r"inputs\.(\w+)", arg)
                    if field_match:
                        mappings[param_name] = field_match.group(1)
                        continue

                var_name = arg.split("(")[0].split("[")[0].strip()
                if var_name and var_name not in ["model", "data", "io"]:
                    assign_pattern = rf"{re.escape(var_name)}\s*=\s*inputs\.(\w+)"
                    assign_match = re.search(assign_pattern, source)
                    if assign_match:
                        mappings[param_name] = assign_match.group(1)

            if mappings:
                break
        return mappings
    except Exception:
        return {}


def _trace_class_workflow_to_model_mappings(workflow_class) -> Dict[str, str]:
    """Trace how class workflow parameters map to earth2studio run function parameters."""
    try:
        # Determine which method to analyze:
        # - If class has its own run() method (in __dict__), use that
        # - Otherwise (Earth2Workflow pattern), use __call__ where the actual
        #   earth2studio run.* call lives
        if "run" in workflow_class.__dict__:
            target_method = workflow_class.run
        elif "__call__" in workflow_class.__dict__:
            target_method = workflow_class.__call__
        else:
            # Fallback to inherited run or __call__
            target_method = getattr(workflow_class, "run", None)
            if target_method is None:
                target_method = inspect.getattr_static(workflow_class, "__call__", None)

        if target_method is None:
            return {}

        source = inspect.getsource(target_method)
        mappings = {}

        extraction_patterns = [
            r'(\w+)\s*=\s*parameters\.get\(["\'](\w+)["\']\)',
            r'(\w+)\s*=\s*parameters\.get\(["\'](\w+)["\']\s*,',
            r'(\w+)\s*=\s*parameters\[["\'](\w+)["\']\]',
            r'(\w+)\s*=\s*kwargs\.get\(["\'](\w+)["\']\)',
            r'(\w+)\s*=\s*kwargs\[["\'](\w+)["\']\]',
        ]

        var_to_param = {}
        for pattern in extraction_patterns:
            matches = re.findall(pattern, source)
            for var_name, param_name in matches:
                var_to_param[var_name] = param_name

        sig = inspect.signature(target_method)
        for name in sig.parameters:
            if name not in ["self", "io", "args", "kwargs"]:
                var_to_param[name] = name

        for run_type, param_names in RUN_FUNCTION_SIGNATURES.items():
            pattern = rf"\brun\.{run_type}\s*\((.*?)\)"
            matches = re.findall(pattern, source, re.DOTALL)
            if not matches:
                continue

            matches = [m for m in matches if m.strip()]
            if not matches:
                continue

            args_str = matches[0]
            args = _split_function_args(args_str)
            data_params = param_names[:3]

            for i, param_name in enumerate(data_params):
                if i >= len(args):
                    break
                arg = args[i].strip()
                var_name = arg.split("(")[0].split("[")[0].strip()

                if var_name in var_to_param:
                    mappings[param_name] = var_to_param[var_name]
                elif var_name in sig.parameters:
                    mappings[param_name] = var_name

            if mappings:
                break
        return mappings
    except Exception:
        return {}


def _build_gfs_parameter_list(input_mappings: Dict[str, str]) -> Dict[str, Any]:
    """Build GFS parameter requirements from workflow input mappings."""
    gfs_params = {}
    if "time" in input_mappings:
        gfs_params["time"] = {
            "workflow_field": input_mappings["time"],
            "purpose": "forecast_initialization",
            "data_source_param": "time",
            "type": "List[str]",
            "description": "Forecast initialization times for GFS data retrieval",
        }
    if "nsteps" in input_mappings:
        gfs_params["nsteps"] = {
            "workflow_field": input_mappings["nsteps"],
            "purpose": "forecast_horizon",
            "affects": "lead_times_to_download",
            "type": "int",
            "description": "Number of forecast steps",
        }
    if "nmembers" in input_mappings:
        gfs_params["nmembers"] = {
            "workflow_field": input_mappings["nmembers"],
            "purpose": "ensemble_size",
            "data_source_param": "ensemble_members",
            "type": "int",
            "description": "Number of ensemble members",
        }
    return gfs_params


def _get_workflow_class_from_obj(workflow_obj) -> Optional[type]:
    """Resolve the workflow class from a workflow object or wrapper."""
    if hasattr(workflow_obj, "_class"):
        return workflow_obj._class
    if hasattr(workflow_obj, "_instance"):
        return workflow_obj._instance.__class__
    return workflow_obj.__class__


def _apply_time_field_fallback(
    metadata: Dict[str, Any],
    input_mappings: Dict[str, str],
    params_cls: Optional[type],
) -> Dict[str, str]:
    """Ensure time mapping exists when workflows define start_time."""
    mappings = input_mappings or {}
    has_time = "time" in mappings
    if not has_time and params_cls and hasattr(params_cls, "model_fields"):
        if "start_time" in params_cls.model_fields:
            mappings = dict(mappings)
            mappings["time"] = "start_time"
            has_time = True

    if mappings:
        metadata["input_mappings"] = mappings
        metadata["gfs_parameters"] = _build_gfs_parameter_list(mappings)
        if has_time:
            metadata["time_field"] = mappings["time"]

    return mappings


def _load_model_instance(model_class, model_name: str):
    """Load a model instance, trying package-based loading first, then direct instantiation."""
    try:
        package = model_class.load_default_package()
        return model_class.load_model(package)
    except Exception:
        pass

    try:
        return model_class()
    except Exception as ex:
        print(f"  Warning: Failed to load {model_name}: {ex}", file=sys.stderr)
        return None


def _extract_variables_from_coords(input_coords: dict) -> set:
    """Extract variable names from model input coordinates."""
    if "variable" not in input_coords:
        return set()

    var_array = input_coords["variable"]
    if hasattr(var_array, "tolist"):
        return set(var_array.tolist())
    return set(var_array)


def _extract_lead_times_from_coords(input_coords: dict) -> set:
    """Extract lead times (in hours) from model input coordinates."""
    import numpy as np

    if "lead_time" not in input_coords:
        return set()

    lead_times = set()
    for lt in input_coords["lead_time"]:
        hours = _parse_lead_time_to_hours(lt, np)
        if hours is not None:
            lead_times.add(hours)
    return lead_times


def _parse_lead_time_to_hours(lt, np) -> Optional[int]:
    """Parse a lead time value to hours."""
    try:
        return int(lt / np.timedelta64(1, "h"))
    except Exception:
        pass
    try:
        return int(lt.total_seconds() // 3600)
    except Exception:
        return None


def _process_single_model(model_info: Dict[str, Any]) -> tuple[set, set]:
    """Process a single model and return its variables and lead times.

    Uses static metadata registry when available to avoid expensive model loading.
    Falls back to dynamic loading for models not in the registry.
    """
    model_name = model_info["model_name"]
    model_module = model_info["module"]

    # First, check if we have static metadata for this model (fast path)
    static_meta = get_static_metadata(model_name)
    if static_meta is not None:
        print(
            f"  Using static metadata for {model_name} (no model loading required)",
            file=sys.stderr,
        )
        variables = set(static_meta.get("variables", []))
        lead_times = set(static_meta.get("lead_times", []))
        return variables, lead_times

    # Check if this model is known to require dynamic loading
    if model_name in DYNAMIC_METADATA_MODELS:
        print(
            f"  {model_name} requires dynamic loading - SKIPPING (use --full to load)",
            file=sys.stderr,
        )
        return set(), set()  # Skip dynamic models in metadata-only mode

    # Fall back to dynamic loading for unknown models
    try:
        mod = importlib.import_module(model_module)
        model_class = getattr(mod, model_name)
    except Exception as e:
        print(f"  Warning: Failed to import {model_name}: {e}", file=sys.stderr)
        return set(), set()

    print(f"  Loading {model_name} to extract metadata...", file=sys.stderr)
    model = _load_model_instance(model_class, model_name)
    if not model:
        return set(), set()

    try:
        input_coords = model.input_coords()
    except Exception as e:
        print(
            f"  Warning: Failed to get input_coords from {model_name}: {e}",
            file=sys.stderr,
        )
        return set(), set()

    variables = _extract_variables_from_coords(input_coords)
    lead_times = _extract_lead_times_from_coords(input_coords)
    return variables, lead_times


def _extract_from_models(model_infos: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Load multiple models and merge their metadata requirements."""
    all_variables: set = set()
    all_lead_times: set = set()
    model_names = []
    model_categories = []

    print(f"Detected {len(model_infos)} model(s) in workflow", file=sys.stderr)

    for model_info in model_infos:
        model_names.append(model_info["model_name"])
        model_categories.append(model_info["category"])

        variables, lead_times = _process_single_model(model_info)
        all_variables.update(variables)
        all_lead_times.update(lead_times)

    if not model_names:
        return {}

    return {
        "model_name": "+".join(model_names),
        "model_category": "+".join(set(model_categories)),
        "data_source": "GFS",
        "uri_prefix": "noaa-gfs-bdp-pds",
        "variables": sorted(all_variables),
        "lead_times": sorted(all_lead_times),
        "interp_method": "nearest",
        "num_models": len(model_names),
        "models": model_names,
    }


def _extract_model_input_requirements(model_class) -> Dict[str, Any]:
    """Extract input requirements from earth2studio model class."""
    try:
        if callable(model_class):
            method = model_class.__call__
        elif hasattr(model_class, "forward"):
            method = model_class.forward
        else:
            return {}

        sig = inspect.signature(method)
        requirements = {}

        for param_name, param in sig.parameters.items():
            if param_name in ["self", "cls"]:
                continue

            param_type = "unknown"
            if param.annotation != inspect.Parameter.empty:
                param_type = str(param.annotation)
                param_type = (
                    param_type.replace("typing.", "")
                    .replace("<class ", "")
                    .replace(">", "")
                    .replace("'", "")
                )

            is_required = param.default == inspect.Parameter.empty

            requirements[param_name] = {"type": param_type, "required": is_required}
            if not is_required:
                requirements[param_name]["default"] = str(param.default)
        return requirements
    except Exception:
        return {}


def _load_model_class(model_info: Dict[str, Any]):
    try:
        model_name = model_info["model_name"]
        model_module = model_info["module"]
        mod = importlib.import_module(model_module)
        return getattr(mod, model_name)
    except Exception:
        return None


# --- Existing Helper Functions ---


def _convert_param(param: inspect.Parameter, typeinfo: type) -> tuple[type, Field]:
    """Convert parameter to Pydantic Field declaration"""
    field_kwargs = {}
    if param.default is not param.empty:
        field_kwargs["default"] = param.default
    return (typeinfo, Field(**field_kwargs))


def func_to_model(
    function: Callable,
    model_name: str = "Parameters",
    exclude_params: Optional[Set[str]] = None,
) -> type[BaseModel]:
    """Create a Pydantic model from function call signature"""
    if exclude_params is None:
        exclude_params = set()
    exclude_params = exclude_params | {"self", "args", "kwargs"}

    params = dict(inspect.signature(function).parameters)
    try:
        type_hints = get_type_hints(function)
    except Exception:
        type_hints = {}

    converted_params = {
        name: _convert_param(param, type_hints.get(name, Any))
        for (name, param) in params.items()
        if name not in exclude_params
    }
    return create_model(model_name, **converted_params)


def ensure_parameters(obj) -> bool:
    """
    Inspects the object (class or function) and attaches a 'Parameters' Pydantic model
    if one does not exist. Returns True if successful.
    """
    if hasattr(obj, "Parameters"):
        return True

    # Identify target function to inspect
    if inspect.isfunction(obj):
        target = obj
    elif inspect.ismethod(obj) or inspect.isclass(obj) or callable(obj):
        # For classes and callable objects, get the __call__ method
        try:
            target = obj.__call__
        except AttributeError:
            target = obj
    else:
        target = None

    if not target:
        return False

    # Strategy 1: Check for Pydantic Injection (Single Argument)
    try:
        sig = inspect.signature(target)
        type_hints = get_type_hints(target)

        for name, _param in sig.parameters.items():
            if name in ("self", "io", "ctx", "args", "kwargs"):
                continue

            type_hint = type_hints.get(name)
            if (
                type_hint
                and isinstance(type_hint, type)
                and issubclass(type_hint, BaseModel)
            ):
                # Found a Pydantic model! Use it.
                obj.Parameters = type_hint
                return True
    except Exception:
        pass

    # Strategy 2: Auto-generate from signature
    try:
        obj.Parameters = func_to_model(target, exclude_params={"self", "io", "ctx"})
        return True
    except Exception:
        return False


def _trace_workflow_input_mappings(workflow_obj) -> dict:
    """
    Trace input mappings for a workflow object via class introspection.
    """
    return _trace_class_workflow_to_model_mappings(workflow_obj.__class__)


def _extract_model_requirements_from_infos(model_infos: list) -> dict:
    """Extract model-specific input requirements from a list of model infos."""
    model_requirements = {}
    for model_info in model_infos:
        model_class = _load_model_class(model_info)
        if model_class:
            reqs = _extract_model_input_requirements(model_class)
            if reqs:
                model_requirements[model_info["model_name"]] = reqs
    return model_requirements


def _get_workflow_module(workflow_obj):
    """Get the module for a workflow object.

    For wrapped workflows (WorkflowWrapper, Earth2WorkflowWrapper),
    we need to get the module from the underlying workflow class,
    not from the wrapper itself.
    """
    # Check if this is a wrapper with an underlying _class attribute
    if hasattr(workflow_obj, "_class"):
        workflow_class = workflow_obj._class
        module = inspect.getmodule(workflow_class)
        if module:
            return module

    # Check if this is a wrapper with an underlying _instance attribute
    if hasattr(workflow_obj, "_instance"):
        instance = workflow_obj._instance
        module = inspect.getmodule(instance)
        if not module:
            module = inspect.getmodule(instance.__class__)
        if module:
            return module

    # Fall back to direct module lookup
    module = inspect.getmodule(workflow_obj)
    if not module and hasattr(workflow_obj, "__class__"):
        module = inspect.getmodule(workflow_obj.__class__)
    return module


def extract_model_metadata(workflow_obj) -> dict:
    """
    Extract model metadata (variables, lead_times) using heuristic analysis
    or introspection of the workflow object.
    """
    # Check if workflow exposes metadata directly (explicit override)
    if hasattr(workflow_obj, "model_metadata"):
        return workflow_obj.model_metadata

    try:
        module = _get_workflow_module(workflow_obj)
        if not module:
            return {}

        # Get source code and detect models
        try:
            source = inspect.getsource(module)
        except Exception:
            return {}

        model_infos = _detect_models_from_source(source)
        if not model_infos:
            return {}

        # Build metadata from models and input mappings
        metadata = _extract_from_models(model_infos)
        input_mappings = _trace_workflow_input_mappings(workflow_obj)
        workflow_class = _get_workflow_class_from_obj(workflow_obj)
        params_cls = _get_parameters_class(workflow_class) if workflow_class else None
        _apply_time_field_fallback(metadata, input_mappings, params_cls)

        # Extract model-specific requirements
        try:
            model_requirements = _extract_model_requirements_from_infos(model_infos)
            if model_requirements:
                metadata["model_requirements"] = model_requirements
        except Exception as e:
            print(
                f"Warning: Could not extract model requirements: {e}", file=sys.stderr
            )

        return metadata

    except Exception as e:
        print(f"Warning: Advanced metadata extraction failed: {e}", file=sys.stderr)
        return {}


def resolve_workflow_path(file_path: str) -> Path:
    path = Path(file_path)
    if path.exists():
        return path

    workflow_dir = os.environ.get("WORKFLOW_DIR")
    if workflow_dir:
        potential_path = Path(workflow_dir) / file_path
        if potential_path.exists():
            return potential_path

    raise FileNotFoundError(f"Workflow file not found: {file_path}")


def _has_own_method(cls, method_name: str) -> bool:
    """Check if a class defines its own method (not just inherited)."""
    # Simply check if the method is in the class's own __dict__
    # This correctly excludes inherited methods
    return method_name in cls.__dict__


def _is_earth2_workflow_subclass(cls) -> bool:
    """Check if class is an Earth2Workflow subclass.

    Earth2Workflow pattern:
    - Defines its own __call__(io, **params) method
    - Does NOT define its own run() method (uses inherited one)
    """
    if not inspect.isclass(cls):
        return False

    # Check for __call__ method with io parameter defined in the class itself
    if not _has_own_method(cls, "__call__"):
        return False

    call_method = inspect.getattr_static(cls, "__call__", None)
    if call_method is None:
        return False

    try:
        sig = inspect.signature(call_method)
        params = list(sig.parameters.keys())
        # Earth2Workflow.__call__ signature: __call__(self, io, **kwargs)
        if "io" not in params:
            return False
    except (ValueError, TypeError):
        return False

    # Should NOT have its own run() method (uses inherited one from Earth2Workflow base)
    if _has_own_method(cls, "run"):
        return False

    return True


def _is_workflow_subclass(cls) -> bool:
    """Check if class is a Workflow subclass.

    Workflow pattern:
    - Defines its own run(parameters, execution_id) method
    - May or may not have __call__ method
    """
    if not inspect.isclass(cls):
        return False

    # If it's an Earth2Workflow, it's not a plain Workflow
    if _is_earth2_workflow_subclass(cls):
        return False

    # Check for run method with specific signature
    run_method = getattr(cls, "run", None)
    if run_method is None:
        return False

    try:
        sig = inspect.signature(run_method)
        params = list(sig.parameters.keys())
        # Workflow.run signature: run(self, parameters, execution_id)
        return "parameters" in params and "execution_id" in params
    except (ValueError, TypeError):
        return False


def _detect_workflow_type(workflow_class) -> str:
    """Detect the type of workflow class.

    Priority:
    1. Earth2Workflow - has __call__(io, ...) without own run() method
    2. Workflow - has run(parameters, execution_id) method
    3. Generic callable
    """
    if _is_earth2_workflow_subclass(workflow_class):
        return WorkflowType.EARTH2_WORKFLOW
    elif _is_workflow_subclass(workflow_class):
        return WorkflowType.WORKFLOW
    elif callable(workflow_class):
        return WorkflowType.CALLABLE_CLASS
    return WorkflowType.CALLABLE_CLASS


def _get_parameters_class(workflow_class):
    """Get the Parameters class from a workflow class.

    Handles both assignment (Parameters = X) and type annotation (Parameters: X) syntax.
    """
    # First check if it's a direct attribute (assignment syntax: Parameters = X)
    params_cls = getattr(workflow_class, "Parameters", None)

    # Check if it's not inherited from a base class (should be defined on this class)
    if params_cls is not None:
        # Verify it's a proper model class and not inherited default
        # Check if it has fields (not empty)
        if hasattr(params_cls, "model_fields") and len(params_cls.model_fields) > 0:
            return params_cls
        # Also check if it's in the class's own __dict__ (not inherited)
        if "Parameters" in workflow_class.__dict__:
            return params_cls

    # Check type annotations (annotation syntax: Parameters: X)
    annotations = getattr(workflow_class, "__annotations__", {})
    if "Parameters" in annotations:
        params_type = annotations["Parameters"]
        # The annotation should be a class, not a type hint
        if isinstance(params_type, type) and hasattr(params_type, "model_fields"):
            return params_type

    # Fallback: return whatever was found or None
    return params_cls


class WorkflowWrapper:
    """Wrapper that provides a unified interface for Workflow subclasses."""

    def __init__(self, workflow_instance, workflow_class):
        self._instance = workflow_instance
        self._class = workflow_class
        self.name = getattr(workflow_class, "name", workflow_class.__name__)
        self.description = getattr(workflow_class, "description", "")
        self.workflow_type = WorkflowType.WORKFLOW

        # Get the Parameters class from the workflow
        params_cls = _get_parameters_class(workflow_class)
        if (
            params_cls is not None
            and hasattr(params_cls, "model_fields")
            and len(params_cls.model_fields) > 0
        ):
            self.Parameters = params_cls
        else:
            # Auto-generate from run method
            run_method = getattr(workflow_class, "run", None)
            if run_method:
                self.Parameters = func_to_model(
                    run_method, exclude_params={"self", "execution_id", "io"}
                )
            else:
                self.Parameters = create_model("Parameters")

    def run(self, parameters: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
        """Execute the workflow's run method."""
        # Validate parameters
        validated = self.Parameters(**parameters)
        return self._instance.run(validated.model_dump(), execution_id)

    def __call__(self, io=None, **kwargs):
        """Call the workflow's run method with auto-generated execution_id."""
        execution_id = kwargs.pop("execution_id", str(uuid.uuid4()))
        return self.run(kwargs, execution_id)


class Earth2WorkflowWrapper:
    """Wrapper that provides a unified interface for Earth2Workflow subclasses."""

    def __init__(self, workflow_instance, workflow_class):
        self._instance = workflow_instance
        self._class = workflow_class
        self.name = getattr(workflow_class, "name", workflow_class.__name__)
        self.description = getattr(workflow_class, "description", "")
        self.workflow_type = WorkflowType.EARTH2_WORKFLOW

        # Get the Parameters class from the workflow (auto-generated by AutoParameters metaclass)
        params_cls = getattr(workflow_class, "Parameters", None)
        if params_cls is not None:
            self.Parameters = params_cls
        else:
            # Auto-generate from __call__ method
            call_method = inspect.getattr_static(workflow_class, "__call__", None)
            if call_method:
                self.Parameters = func_to_model(
                    call_method, exclude_params={"self", "io"}
                )
            else:
                self.Parameters = create_model("Parameters")

    def __call__(self, io=None, **kwargs):
        """Execute the workflow's __call__ method."""
        # Validate parameters
        validated = self.Parameters(**kwargs)
        return self._instance(io=io, **validated.model_dump())


def _find_registered_classes(source: str) -> set:
    """
    Find class names that are decorated with @workflow_registry.register.

    Parses the source code to detect the decorator pattern.
    """
    registered = set()

    # Pattern to match @workflow_registry.register decorator followed by class definition
    # Handles both @workflow_registry.register and @workflow_registry.register()
    pattern = (
        r"@workflow_registry\.register(?:\(\))?\s*\n"
        r"(?:@\w+(?:\.\w+)*(?:\([^)]*\))?\s*\n)*class\s+(\w+)"
    )

    matches = re.findall(pattern, source)
    registered.update(matches)

    return registered


def _find_workflow_class(module, module_name: str, source: Optional[str] = None):
    """Find the best workflow class candidate in a module.

    Priority:
    1. Classes decorated with @workflow_registry.register (if source provided)
    2. Workflow subclass (has run method with parameters + execution_id)
    3. Earth2Workflow subclass (has __call__ with io parameter)
    4. Any callable class with "Workflow" in name or matching module name

    Args:
        module: The loaded Python module
        module_name: Name of the module
        source: Optional source code string for detecting decorated classes
    """
    # First, try to find registered classes from source
    registered_classes = set()
    if source:
        registered_classes = _find_registered_classes(source)

    workflow_candidates = []
    earth2_candidates = []
    other_candidates = []

    for name, obj in inspect.getmembers(module):
        if not inspect.isclass(obj) or obj.__module__ != module.__name__:
            continue

        # Check if this class is registered (highest priority)
        is_registered = name in registered_classes

        if _is_workflow_subclass(obj):
            workflow_candidates.append((name, obj, is_registered))
        elif _is_earth2_workflow_subclass(obj):
            earth2_candidates.append((name, obj, is_registered))
        elif callable(obj):
            other_candidates.append((name, obj, is_registered))

    # Helper to find best match - prioritize registered classes
    def find_best_match(candidates):
        if not candidates:
            return None

        # First, look for registered classes
        for _name, obj, is_registered in candidates:
            if is_registered:
                return obj

        # Fall back to name matching
        for name, obj, _ in candidates:
            if "Workflow" in name or name.lower() == module_name.replace("_", ""):
                return obj

        # Return first candidate if no match
        return candidates[0][1]

    # Priority: Workflow > Earth2Workflow > Other callable
    return (
        find_best_match(workflow_candidates)
        or find_best_match(earth2_candidates)
        or find_best_match(other_candidates)
    )


def _load_module_from_path(path: Path):
    """Load a Python module from a file path.

    Returns:
        Tuple of (module, module_name, source_code)
    """
    module_name = path.stem

    # Read source code for decorator detection
    source = path.read_text(encoding="utf-8")

    spec = importlib.util.spec_from_file_location(module_name, path)
    if not spec or not spec.loader:
        raise ImportError(f"Could not load module spec from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, module_name, source


def _load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML or JSON file.

    Args:
        config_path: Path to the config file (.yaml, .yml, or .json)

    Returns:
        Dictionary with configuration values
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    content = path.read_text()

    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml

            return yaml.safe_load(content) or {}
        except ImportError as err:
            raise ImportError(
                "PyYAML is required to load YAML config files. Install with: pip install pyyaml"
            ) from err
    elif path.suffix == ".json":
        return json.loads(content)
    else:
        # Try JSON first, then YAML
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            try:
                import yaml

                return yaml.safe_load(content) or {}
            except ImportError as err:
                raise ValueError(
                    f"Could not parse config file: {config_path}. Unknown format."
                ) from err


def _get_workflow_config(
    workflow_name: str,
    config_file: Optional[str] = None,
    config_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Get workflow configuration from multiple sources.

    Priority (highest to lowest):
    1. config_override (direct dict passed in)
    2. config_file (YAML/JSON file path)
    3. WORKFLOW_CONFIG environment variable (JSON string with workflow_name key)
    4. Empty dict (default)

    Args:
        workflow_name: Name of the workflow (used for env var lookup)
        config_file: Optional path to a config file
        config_override: Optional dict to override/merge with file config

    Returns:
        Dictionary with merged configuration values
    """
    config = {}

    # 1. Try environment variable first (lowest priority base)
    config_str = os.environ.get("WORKFLOW_CONFIG", "{}")
    try:
        all_configs = json.loads(config_str)
        config = all_configs.get(workflow_name, {})
    except json.JSONDecodeError:
        pass

    # 2. Load from config file (overrides env var)
    if config_file:
        file_config = _load_config_from_file(config_file)
        # If file has nested workflow config, extract it
        if workflow_name in file_config:
            config.update(file_config[workflow_name])
        else:
            config.update(file_config)

    # 3. Apply overrides (highest priority)
    if config_override:
        config.update(config_override)

    return config


def _instantiate_workflow(
    workflow_class,
    module_name: str,
    config_file: Optional[str] = None,
    config_override: Optional[Dict[str, Any]] = None,
):
    """Instantiate a workflow class with appropriate wrapper based on type.

    Args:
        workflow_class: The workflow class to instantiate
        module_name: Name of the module (used as fallback for workflow name)
        config_file: Optional path to a config file (YAML/JSON)
        config_override: Optional dict to override config values
    """
    workflow_type = _detect_workflow_type(workflow_class)

    # Get config for the workflow
    workflow_name = getattr(workflow_class, "name", module_name)
    config = _get_workflow_config(workflow_name, config_file, config_override)

    try:
        # Try to get Config class and validate
        config_cls = getattr(workflow_class, "Config", None)
        if config_cls is not None and config:
            try:
                validated_config = config_cls(**config)
                if hasattr(validated_config, "model_dump"):
                    config = validated_config.model_dump()
                else:
                    config = dict(validated_config)
            except Exception:
                pass  # Use raw config if validation fails

        # Instantiate with config
        if config:
            instance = workflow_class(**config)
        else:
            instance = workflow_class()

    except Exception as e:
        print(
            f"Warning: Failed to instantiate {workflow_class.__name__} with config: {e}",
            file=sys.stderr,
        )
        try:
            instance = workflow_class()
        except Exception as e2:
            raise ValueError(
                f"Failed to instantiate workflow {workflow_class.__name__}: {e2}"
            ) from e2

    # Wrap based on type
    if workflow_type == WorkflowType.WORKFLOW:
        return WorkflowWrapper(instance, workflow_class)
    elif workflow_type == WorkflowType.EARTH2_WORKFLOW:
        return Earth2WorkflowWrapper(instance, workflow_class)
    else:
        # Generic callable - ensure it has Parameters
        if not hasattr(instance, "Parameters"):
            ensure_parameters(instance)
        instance.workflow_type = WorkflowType.CALLABLE_CLASS
        return instance


def load_workflow(
    file_path: str,
    config_file: Optional[str] = None,
    config_override: Optional[Dict[str, Any]] = None,
):
    """
    Load workflow from file.
    Returns an instantiated workflow object with a .Parameters attribute.

    Only loads classes decorated with @workflow_registry.register.

    Supports:
    1. Workflow subclasses (with run(parameters, execution_id) method)
    2. Earth2Workflow subclasses (with __call__(io, **params) method)
    3. Generic callable classes (with __call__ method)

    Args:
        file_path: Path to the workflow Python file
        config_file: Optional path to a config file (YAML/JSON) for workflow initialization
        config_override: Optional dict to override config values
    """
    path = resolve_workflow_path(file_path)
    module, module_name, source = _load_module_from_path(path)

    # Find and instantiate workflow class (prioritizes registered classes)
    candidate = _find_workflow_class(module, module_name, source)
    if candidate:
        return _instantiate_workflow(
            candidate, module_name, config_file, config_override
        )

    raise ValueError(
        f"No valid workflow found in {path}. Expected:\n"
        f"  - Workflow subclass with run(parameters, execution_id) method\n"
        f"  - Earth2Workflow subclass with __call__(io, **params) method\n"
        f"  - Callable class with __call__() method"
    )


def _inspect_workflow_class_only(file_path: str) -> dict:
    """Inspect workflow without instantiation - no model loading.

    This extracts metadata directly from the class definition and source code,
    avoiding the need to instantiate the workflow (which often loads models).

    Returns:
        Dictionary with workflow metadata (name, description, schemas, etc.)
    """
    path = resolve_workflow_path(file_path)
    module, module_name, source = _load_module_from_path(path)

    # Find workflow class without instantiation
    workflow_class = _find_workflow_class(module, module_name, source)
    if not workflow_class:
        raise ValueError(f"No valid workflow class found in {path}")

    # Get workflow type
    workflow_type = _detect_workflow_type(workflow_class)

    # Get Parameters schema without instantiation
    params_cls = _get_parameters_class(workflow_class)
    if params_cls is None or not hasattr(params_cls, "model_fields"):
        # Auto-generate from method signature
        if workflow_type == WorkflowType.EARTH2_WORKFLOW:
            call_method = inspect.getattr_static(workflow_class, "__call__", None)
            if call_method:
                params_cls = func_to_model(call_method, exclude_params={"self", "io"})
        elif workflow_type == WorkflowType.WORKFLOW:
            run_method = getattr(workflow_class, "run", None)
            if run_method:
                params_cls = func_to_model(
                    run_method, exclude_params={"self", "execution_id", "io"}
                )
        else:
            params_cls = create_model("Parameters")

    params_schema = params_cls.model_json_schema() if params_cls else {}

    # Get Config schema
    config_schema = None
    config_cls = getattr(workflow_class, "Config", None)
    if config_cls and hasattr(config_cls, "model_json_schema"):
        config_schema = config_cls.model_json_schema()

    # Extract model metadata from SOURCE CODE using static registry
    # This avoids instantiation entirely
    model_infos = _detect_models_from_source(source)
    if model_infos:
        model_metadata = _extract_from_models(model_infos)
        # Also add input mappings if we can trace them
        input_mappings = _trace_class_workflow_to_model_mappings(workflow_class)
        _apply_time_field_fallback(model_metadata, input_mappings, params_cls)
    else:
        model_metadata = {}

    return {
        "name": getattr(workflow_class, "name", module_name),
        "description": getattr(workflow_class, "description", ""),
        "workflow_type": workflow_type,
        "parameters_schema": params_schema,
        "config_schema": config_schema,
        "model_metadata": model_metadata,
        "schema": params_schema,  # Legacy field
    }


def inspect_command(args):
    try:
        # Check if we should use metadata-only mode (no instantiation)
        # Default is metadata-only (no model loading), use --full to instantiate
        full_inspection = getattr(args, "full", False)
        metadata_only = not full_inspection

        if metadata_only:
            # Fast path: inspect class without instantiation (no model loading)
            output = _inspect_workflow_class_only(args.file)
        else:
            # Full inspection with instantiation (may load models)
            config_file = getattr(args, "config", None)
            workflow = load_workflow(args.file, config_file=config_file)

            # Get Parameters schema
            params_schema = workflow.Parameters.model_json_schema()

            # Extract optional model metadata
            model_metadata = extract_model_metadata(workflow)

            # Get workflow type
            workflow_type = getattr(
                workflow, "workflow_type", WorkflowType.CALLABLE_CLASS
            )

            # Get Config schema if available
            config_schema = None
            workflow_class = getattr(workflow, "_class", None)
            if workflow_class:
                config_cls = getattr(workflow_class, "Config", None)
                if config_cls and hasattr(config_cls, "model_json_schema"):
                    config_schema = config_cls.model_json_schema()

            output = {
                "name": getattr(workflow, "name", Path(args.file).stem),
                "description": getattr(workflow, "description", ""),
                "workflow_type": workflow_type,
                "parameters_schema": params_schema,
                "config_schema": config_schema,
                "model_metadata": model_metadata,
                # Legacy field for backwards compatibility
                "schema": params_schema,
            }

        print(json.dumps(output, indent=2))
    except Exception as e:
        import traceback

        print(
            json.dumps({"error": str(e), "traceback": traceback.format_exc()}),
            file=sys.stderr,
        )
        sys.exit(1)


def _get_io_backend(output_path: Optional[str] = None):
    """Get an IO backend for workflow execution.

    This creates a simple IO backend if possible, or returns None.
    """
    if output_path is None:
        output_path = os.environ.get("WORKFLOW_OUTPUT_PATH")

    if output_path is None:
        return None

    try:
        # Try to import earth2studio IO backends
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        if output_path.endswith(".zarr"):
            from earth2studio.io import ZarrBackend

            return ZarrBackend(output_path)
        elif output_path.endswith(".nc"):
            from earth2studio.io import NetCDF4Backend

            return NetCDF4Backend(output_path)
        else:
            # Default to zarr
            from earth2studio.io import ZarrBackend

            return ZarrBackend(output_path + ".zarr")
    except ImportError:
        print(
            "Warning: earth2studio.io not available, using None for IO backend",
            file=sys.stderr,
        )
        return None
    except Exception as e:
        print(f"Warning: Failed to create IO backend: {e}", file=sys.stderr)
        return None


def run_command(args):
    try:
        # Parse config override from CLI if provided
        config_override = None
        if hasattr(args, "config_json") and args.config_json:
            config_override = json.loads(args.config_json)

        # Load workflow with config
        config_file = getattr(args, "config", None)
        workflow = load_workflow(
            args.file,
            config_file=config_file,
            config_override=config_override,
        )

        # Parse and Validate Inputs (runtime parameters)
        inputs = json.loads(args.inputs) if args.inputs else {}

        try:
            validated_params = workflow.Parameters(**inputs)
        except Exception as e:
            raise ValueError(f"Input validation failed: {e}") from e

        # Get workflow type
        workflow_type = getattr(workflow, "workflow_type", WorkflowType.CALLABLE_CLASS)

        # Get IO backend (from args or environment)
        output_path = getattr(args, "output", None) or os.environ.get(
            "WORKFLOW_OUTPUT_PATH"
        )
        io_backend = _get_io_backend(output_path)

        # Generate execution ID
        execution_id = (
            getattr(args, "execution_id", None)
            or os.environ.get("EXECUTION_ID")
            or str(uuid.uuid4())
        )

        # Execute based on workflow type
        if workflow_type == WorkflowType.WORKFLOW:
            # Workflow subclass - call run(parameters, execution_id)
            result = workflow.run(validated_params.model_dump(), execution_id)
        elif workflow_type == WorkflowType.EARTH2_WORKFLOW:
            # Earth2Workflow subclass - call __call__(io, **params)
            result = workflow(io=io_backend, **validated_params.model_dump())
        else:
            # Generic callable - determine best calling convention
            sig = inspect.signature(workflow.__call__)
            param_names = list(sig.parameters.keys())

            if "io" in param_names:
                result = workflow(io=io_backend, **validated_params.model_dump())
            else:
                result = workflow(**validated_params.model_dump())

        print(
            json.dumps({"status": "success", "result": str(result) if result else None})
        )

    except Exception as e:
        import traceback

        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            ),
            file=sys.stderr,
        )
        sys.exit(1)


def list_command(args):
    """List all workflows in a directory."""
    try:
        workflow_dir = Path(args.directory)
        if not workflow_dir.exists():
            raise ValueError(f"Directory does not exist: {args.directory}")

        # Use metadata-only mode by default to avoid model loading
        metadata_only = not getattr(args, "full", False)

        workflows = []
        for py_file in workflow_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            try:
                if metadata_only:
                    # Fast path: inspect class without instantiation
                    info = _inspect_workflow_class_only(str(py_file))
                    workflows.append(
                        {
                            "file": str(py_file),
                            "name": info.get("name", py_file.stem),
                            "description": info.get("description", ""),
                            "workflow_type": info.get(
                                "workflow_type", WorkflowType.CALLABLE_CLASS
                            ),
                        }
                    )
                else:
                    # Full inspection with instantiation
                    workflow = load_workflow(str(py_file))
                    workflows.append(
                        {
                            "file": str(py_file),
                            "name": getattr(workflow, "name", py_file.stem),
                            "description": getattr(workflow, "description", ""),
                            "workflow_type": getattr(
                                workflow,
                                "workflow_type",
                                WorkflowType.CALLABLE_CLASS,
                            ),
                        }
                    )
            except Exception as e:
                workflows.append(
                    {
                        "file": str(py_file),
                        "error": str(e),
                    }
                )

        print(json.dumps({"workflows": workflows}, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Earth2Studio Workflow Interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Inspect command
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect a workflow file and output its schema",
    )
    inspect_parser.add_argument("file", help="Path to the workflow file")
    inspect_parser.add_argument(
        "--config",
        default=None,
        help="Path to config file (YAML/JSON) for workflow initialization",
    )
    inspect_parser.add_argument(
        "--metadata-only",
        action="store_true",
        default=True,
        help=(
            "Extract metadata without instantiating workflow (avoids model loading). Default: True"
        ),
    )
    inspect_parser.add_argument(
        "--full",
        action="store_true",
        default=False,
        help="Full inspection with workflow instantiation (may load models)",
    )

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a workflow with given inputs")
    run_parser.add_argument("file", help="Path to the workflow file")
    run_parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to config file (YAML/JSON) for workflow initialization (__init__ args)"
        ),
    )
    run_parser.add_argument(
        "--config-json",
        default=None,
        help="JSON string of config values (overrides --config file)",
    )
    run_parser.add_argument(
        "--inputs",
        default="{}",
        help="JSON string of runtime parameters (__call__ args)",
    )
    run_parser.add_argument(
        "--output",
        default=None,
        help="Output path for results (zarr or netcdf)",
    )
    run_parser.add_argument(
        "--execution-id", default=None, help="Execution ID for tracking"
    )

    # List command
    list_parser = subparsers.add_parser(
        "list", help="List all workflows in a directory"
    )
    list_parser.add_argument("directory", help="Directory containing workflow files")
    list_parser.add_argument(
        "--full",
        action="store_true",
        default=False,
        help="Full inspection with workflow instantiation (may load models)",
    )

    args = parser.parse_args()

    if args.command == "inspect":
        inspect_command(args)
    elif args.command == "run":
        run_command(args)
    elif args.command == "list":
        list_command(args)


if __name__ == "__main__":
    main()
