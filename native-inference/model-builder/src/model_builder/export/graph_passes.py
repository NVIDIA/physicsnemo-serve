"""Apply recorded ONNX preparation passes to an isolated exported program."""

import copy
import hashlib
import inspect
import json
import marshal
from pathlib import Path


def prepare_onnx_program(model, inputs, options, report_path):
    import torch

    # A private model also protects the reference instance from capture-time
    # changes to buffers and parameters. No customization sees the live model.
    exported = torch.export.export(copy.deepcopy(model), inputs, strict=False)
    signature = copy.deepcopy(exported.graph_signature)
    # Lifted parameter placeholders carry fake tensor metadata. Make the private
    # captured values available to constant-folding passes on their real device.
    parameters = {
        spec.arg.name: exported.state_dict[spec.target]
        for spec in exported.graph_signature.input_specs
        if spec.kind == torch.export.graph_signature.InputKind.PARAMETER
    }
    for node in exported.graph.nodes:
        if node.op == "placeholder" and node.name in parameters:
            node.meta["pnmir_parameter_value"] = parameters[node.name]
    report = {
        "format_version": 1,
        "stage": "before_onnx_decomposition",
        "torch_version": str(torch.__version__),
        "status": "running",
        "passes": [],
    }
    destination = Path(report_path)

    def record():
        destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    record()
    try:
        for transform in options.onnx_passes:
            implementation = (
                transform if inspect.isfunction(transform) else type(transform).__call__
            )
            code = getattr(implementation, "__code__", None)
            if code is None:
                raise ValueError(
                    "ONNX graph passes must be Python functions or callable classes"
                )
            item = {
                "name": f"{implementation.__module__}.{implementation.__qualname__}",
                "code_sha256": hashlib.sha256(marshal.dumps(code)).hexdigest(),
                "identity_format": "python-code-marshal-v1",
                "status": "running",
            }
            report["passes"].append(item)
            record()
            count = transform(exported.graph_module)
            if type(count) is not int or count < 0:
                raise ValueError(
                    "ONNX graph passes must return a non-negative rewrite count"
                )
            exported.graph_module.graph.lint()
            exported.graph_module.recompile()
            if exported.graph_signature != signature:
                raise ValueError(
                    "ONNX graph pass changed the exported input/output signature"
                )
            exported.validate()
            item.update(status="complete", rewritten_nodes=count)
            record()
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        for node in exported.graph.nodes:
            node.meta.pop("pnmir_parameter_value", None)
        record()
    return exported
