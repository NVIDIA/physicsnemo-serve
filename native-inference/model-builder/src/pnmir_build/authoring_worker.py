"""Isolated eager preparation followed by the existing native build harness."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tempfile

from . import authoring_sources, inputs, targets, worker


def verify_snapshot(root, manifest):
    for record in manifest["files"]:
        relative = inputs._relative(record["path"], "captured input")
        path = inputs._regular_file(root / relative, "captured input", base=root)
        if inputs._identity(path) != {
            key: record[key] for key in ("sha256", "size_bytes")
        }:
            raise ValueError(f"Captured input changed: {relative}")


def execute(snapshot, output, operation, device, runtime=None, required_gpu_arch=None):
    snapshot = Path(snapshot).resolve(strict=True)
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("Output already exists; choose a fresh directory.")
    report = {
        "format_version": 1,
        "status": "failed",
        "operation": operation,
        "device": device,
        "target_check": None,
    }
    try:
        manifest, _ = inputs._configuration(snapshot / "snapshot.json")
        report["input_identity"] = manifest["input_identity"]
        verify_snapshot(snapshot, manifest)
        report["target_check"] = targets.check_target(device, required_gpu_arch)
        # Inference derives a recipe in scratch space. User inputs stay frozen.
        with tempfile.TemporaryDirectory(prefix="physicsnemo-authoring-") as temporary:
            retained = Path(temporary) / "inputs"
            shutil.copytree(snapshot, retained)
            verify_snapshot(retained, manifest)
            recipe_path = retained / "recipe.json"
            recipe, _ = inputs._configuration(recipe_path)
            selected = inputs.resolve_inputs(recipe, recipe_path)
            if inputs.input_identities(selected) != manifest["model_inputs"]:
                raise ValueError(
                    "Captured model inputs differ from the selected project."
                )
            prepared = worker._prepare_model(
                recipe,
                recipe_path,
                device,
                model_inputs=selected,
                infer_contract=True,
            )
            authoring_sources.verify_imports()
            verify_snapshot(retained, manifest)
            contract = prepared["tensor_contract"]
            report.update(
                tensor_contract=contract,
                case_count=len(prepared["cases"]),
                environment=prepared["environment"],
                weights=prepared["weights"],
            )
            worker._verify_target_environment(report["environment"], required_gpu_arch)
            recipe.pop("input_names", None)
            recipe.pop("output_names", None)
            recipe.update(contract)
            worker._write_json(recipe_path, recipe)
            if operation == "build":
                runtime = (
                    Path(runtime)
                    if runtime
                    else Path(shutil.which("physicsnemo-infer") or "")
                )
                expected = {
                    "recipe": inputs._identity(recipe_path),
                    "adapter": inputs._identity(retained / recipe["adapter"]),
                    "model_inputs": manifest["model_inputs"],
                }
                if manifest.get("runtime"):
                    expected["runtime"] = manifest["runtime"]
                # Release the first model before loading the retained build inputs.
                del prepared

                def verify_sources(build_output, receipt):
                    authoring_sources.verify_imports()
                    model_root = build_output / "model"
                    path = model_root / "source-check.json"
                    worker._write_json(
                        path,
                        {
                            "format_version": 1,
                            "passed": True,
                            "scope": "Captured Python source integrity through compilation",
                            "source": manifest["input_identity"]["source"],
                        },
                    )
                    return {
                        "passed": True,
                        "report": worker._file_identity(path, model_root),
                    }

                built = worker.execute_build(
                    recipe_path,
                    output,
                    recipe["supported_backends"],
                    device,
                    runtime,
                    model_inputs=selected,
                    expected_identity=expected,
                    pre_release_check=verify_sources,
                    required_gpu_arch=required_gpu_arch,
                )
                report.update(
                    case_count=built["case_count"],
                    weights=built["weights"],
                    environment=built["environment"],
                )
            else:
                output.mkdir(parents=True, exist_ok=False)
                shutil.copytree(snapshot, output / "source")
                worker._write_json(output / "source" / "effective-recipe.json", recipe)
            report["status"] = "checked" if operation == "check" else "complete"
            report["source_files"] = worker._inventory(output / "source", output)
    except BaseException as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        output.mkdir(parents=True, exist_ok=True)
        worker._write_json(output / "check.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--operation", choices=("check", "build"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--required-gpu-arch")
    args = parser.parse_args()
    try:
        execute(
            args.input,
            args.output,
            args.operation,
            args.device,
            args.runtime,
            args.required_gpu_arch,
        )
    except BaseException as exc:
        print(f"{type(exc).__name__}: {exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
