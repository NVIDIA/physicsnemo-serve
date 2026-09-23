# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from plugin_runtime import serialize_prepare_result
from plugin_sdk import (
    PluginWorkflow,
    ScatterChild,
    ScatterResult,
    serialize_scatter_result,
)

CASES = json.loads(
    (Path(__file__).parent / "fixtures/scatter_contract.json").read_text()
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_shared_scatter_contract(case):
    if case["valid"]:
        assert (
            serialize_scatter_result(case["result"], case["context"]) == case["result"]
        )
    else:
        with pytest.raises((TypeError, ValueError)):
            serialize_scatter_result(case["result"], case["context"])


def instruction():
    return ScatterResult(
        children=[ScatterChild(operation="run", parameters={"value": 2})],
        child_stage_id="forecast",
        continuation_stage_id="finish",
    )


def test_prepare_serializes_control_without_prediction_fields():
    result = serialize_prepare_result(instruction())
    assert result["kind"] == "scatter"
    assert result["children"][0]["parameters"] == {"value": 2}
    assert not {"status", "output_path", "artifacts"} & result.keys()


def test_execute_scatter_bypasses_prediction_output_model(tmp_path):
    @dataclass
    class Prediction:
        forecast: str

    class Workflow(PluginWorkflow):
        output_model = Prediction

        def run(self, inputs, ctx):
            return instruction()

    result = Workflow().execute(
        {**CASES[0]["context"], "run_id": "parent", "run_dir": str(tmp_path)}
    )
    assert result["kind"] == "scatter"
    assert "forecast" not in result


def test_ordinary_output_validation_is_preserved():
    @dataclass
    class Prediction:
        forecast: str

    workflow = PluginWorkflow()
    workflow.output_model = Prediction
    with pytest.raises((TypeError, ValueError)):
        workflow._normalize_run_result({"wrong_field": 1}, {})


def test_execute_scatter_routes_to_scheduler():
    from inference_worker import (
        _build_primary_completion,
        _normalize_legacy_execute_result,
    )

    scatter = _normalize_legacy_execute_result(
        serialize_prepare_result(instruction()), CASES[0]["context"], 1.0
    )
    payload = {**CASES[0]["context"], "stage_invocation_id": "invocation-1"}
    payload["stage_context"] = {
        **payload["stage_context"],
        "pipeline": [
            *payload["stage_context"]["pipeline"],
            {"id": "scheduler", "phase": "schedule", "queue": "custom-schedule"},
        ],
    }
    completion = {**scatter, "run_id": "parent", "status": "succeeded"}
    stream, forwarded, stage = _build_primary_completion("execute", payload, completion)

    assert (stream, stage) == ("custom-schedule", "schedule")
    assert forwarded["scatter"] == scatter
    assert forwarded["stage_invocation_id"] == "invocation-1"


def test_source_identity_is_replay_stable_and_distinguishes_stage_revisits():
    from inference_worker import _assign_stage_invocation_ids

    payload = {"stage_context": {"current_stage_id": "prepare"}}
    _assign_stage_invocation_ids(payload, "parent", "execute", "1-0")
    first = payload["stage_invocation_id"]
    _assign_stage_invocation_ids(payload, "parent", "execute", "1-0")
    assert payload["stage_invocation_id"] == first
    _assign_stage_invocation_ids(payload, "parent", "execute", "2-0")
    assert payload["stage_invocation_id"] != first
    _assign_stage_invocation_ids(payload, "parent", "other-stream", "1-0")
    assert payload["stage_invocation_id"] != first


def test_batch_items_have_distinct_source_identities():
    from inference_worker import _assign_stage_invocation_ids

    payload = {
        "items": [
            {
                "run_id": run,
                "payload": {"stage_context": {"current_stage_id": "execute"}},
            }
            for run in ["a", "b"]
        ]
    }
    _assign_stage_invocation_ids(payload, "batch", "execute", "1-0")
    assert (
        payload["items"][0]["payload"]["stage_invocation_id"]
        != payload["items"][1]["payload"]["stage_invocation_id"]
    )


def test_invocation_identity_reaches_both_hook_contexts(tmp_path):
    from plugin_runtime import build_context, build_prepare_context

    payload = {
        "run_id": "parent",
        "run_dir": str(tmp_path),
        "stage_invocation_id": "invocation-1",
    }
    assert build_context(payload)["stage_invocation_id"] == "invocation-1"
    assert build_prepare_context(payload).stage_invocation_id == "invocation-1"


@pytest.mark.parametrize("phase", ["prepare", "execute"])
def test_direct_runner_cannot_publish_scatter(monkeypatch, phase):
    import plugin_direct_runner as runner

    monkeypatch.setattr(
        runner, "resolve_phase_hook", lambda *args: lambda ctx: instruction()
    )
    monkeypatch.setattr(runner, "build_context", lambda payload: payload)
    with pytest.raises(ValueError, match="Scatter requires the scheduler"):
        runner._invoke_phase(None, "generic", phase, CASES[0]["context"])


def test_nonfinite_parameters_are_rejected():
    result = ScatterResult(
        children=[ScatterChild(operation="run", parameters={"value": float("nan")})],
        child_stage_id="forecast",
        continuation_stage_id="finish",
    )
    with pytest.raises(ValueError):
        serialize_scatter_result(result)
