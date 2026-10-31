"""Use the pinned benchmark's unchanged metric path for verified physical fields.

This is an external QA evaluator. It runs no inference and does not claim that
the native HTTP response produced the provider's report artifacts.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

from qa.native_inference.cfd_contract import FieldContractError


@contextmanager
def physical_field_predictions(fields):
    from physicsnemo.cfd.evaluation.datasets.schema import build_predictions_dict
    from physicsnemo.cfd.evaluation.models.wrappers.transolver.wrapper import (
        TransolverWrapper,
    )

    seen = set()

    def load(wrapper, *args, **kwargs):
        return wrapper

    def prepare(wrapper, case):
        if case.case_id not in fields or case.case_id in seen:
            raise FieldContractError(
                f"unexpected or repeated evaluator case {case.case_id}"
            )
        values = fields[case.case_id]["physical"]
        if (
            case.reference_geometry is None
            or values.shape[0] != case.reference_geometry.n_cells
        ):
            raise FieldContractError("physical fields do not cover all source cells")
        seen.add(case.case_id)
        return {"case_id": case.case_id, "physical": values}

    def predict(wrapper, inputs):
        return inputs["physical"]

    def decode(wrapper, values, case, model_input=None):
        return build_predictions_dict(
            pressure=values[:, 0], shear_stress=values[:, 1:4]
        )

    with (
        patch.object(TransolverWrapper, "load", load),
        patch.object(TransolverWrapper, "prepare_inputs", prepare),
        patch.object(TransolverWrapper, "predict", predict),
        patch.object(TransolverWrapper, "decode_outputs", decode),
    ):
        yield
    if seen != set(fields):
        raise FieldContractError("evaluator did not execute every field case")
