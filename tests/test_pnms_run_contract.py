"""Direct native execution uses the same public schemas as HTTP serving."""

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "application-cli/src"))
from pnms_app import native  # noqa: E402
from pnms_app.common import write_json  # noqa: E402


@pytest.mark.parametrize("stage", ["request", "result", "nonfinite"])
def test_direct_run_rejects_public_contract_violation(tmp_path, monkeypatch, stage):
    root = tmp_path / "bundle"
    write_json(
        root / "bundle.json",
        {"workflow": {"runner": "runner", "config": "configuration/workflow.json"}},
    )
    write_json(root / "configuration/workflow.json", {"models": {}})
    (root / "runner").write_text("runner")
    schema = {
        "type": "object",
        "required": ["score"],
        "properties": {"score": {"type": "number", "minimum": 0}},
        "additionalProperties": False,
    }
    write_json(root / "configuration/request.schema.json", schema)
    write_json(root / "configuration/result.schema.json", schema)
    request = tmp_path / "request.json"
    request.write_text(
        '{"score": NaN}'
        if stage == "nonfinite"
        else '{"score": -1}'
        if stage == "request"
        else '{"score": 1}'
    )
    executed = []

    def execute(command, **kwargs):
        executed.append(command)
        write_json(
            Path(command[command.index("--output") + 1]), {"payload": {"score": -1}}
        )

    monkeypatch.setattr(native, "_execute", execute)
    with pytest.raises(ValueError, match="schema|finite|JSON"):
        native.run_bundle(root, request, tmp_path / "attempt")
    assert bool(executed) == (stage == "result")
