"""Static tensor contracts for model recipes, without ML imports."""

import re


def _names(names, label):
    if (
        not isinstance(names, list)
        or not names
        or any(
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None
            for name in names
        )
        or len(set(names)) != len(names)
    ):
        raise ValueError(f"recipe requires unique safe {label}")
    return names


def _shape(shape, label):
    if (
        not isinstance(shape, list)
        or not shape
        or any(type(size) is not int or size < 1 for size in shape)
    ):
        raise ValueError(f"{label} requires a non-empty static positive shape")
    return list(shape)


def _descriptors(values, kind):
    if not isinstance(values, list) or not values:
        raise ValueError(f"recipe requires non-empty {kind} tensor descriptors")
    result = []
    for descriptor in values:
        if not isinstance(descriptor, dict) or set(descriptor) != {
            "name",
            "dtype",
            "shape",
        }:
            raise ValueError(f"{kind} tensor descriptors require name, dtype and shape")
        if descriptor["dtype"] != "float32":
            raise ValueError("recipe supports float32 tensors only")
        result.append(
            {
                "name": descriptor["name"],
                "dtype": "float32",
                "shape": _shape(descriptor["shape"], f"{kind} tensor"),
            }
        )
    _names([value["name"] for value in result], kind)
    return result


def tensor_contracts(recipe):
    """Resolve explicit v2 shapes or the unchanged shared-shape shorthand."""
    if type(recipe.get("format_version")) is not int or recipe[
        "format_version"
    ] not in (1, 2):
        raise ValueError("tensor contracts require recipe format_version 1 or 2")
    if "inputs" in recipe or "outputs" in recipe:
        if recipe["format_version"] != 2:
            raise ValueError("named tensor descriptors require recipe format_version 2")
        if any(
            key in recipe for key in ("input_names", "output_names", "dtype", "shape")
        ):
            raise ValueError(
                "named tensors cannot be combined with legacy names, dtype or shape"
            )
        return _descriptors(recipe.get("inputs"), "inputs"), _descriptors(
            recipe.get("outputs"), "outputs"
        )
    input_names = _names(recipe.get("input_names"), "input_names")
    output_names = _names(recipe.get("output_names"), "output_names")
    if recipe.get("dtype") != "float32":
        raise ValueError("recipe supports float32 tensors only")
    shape = _shape(recipe.get("shape"), "recipe")
    return (
        [
            {"name": name, "dtype": "float32", "shape": list(shape)}
            for name in input_names
        ],
        [{"name": name, "dtype": "float32", "shape": None} for name in output_names],
    )


def tensor_names(recipe, kind):
    if kind not in ("inputs", "outputs"):
        raise ValueError("tensor kind must be inputs or outputs")
    if "inputs" in recipe or "outputs" in recipe:
        inputs, outputs = tensor_contracts(recipe)
        return [value["name"] for value in (inputs if kind == "inputs" else outputs)]
    field = "input_names" if kind == "inputs" else "output_names"
    return _names(recipe.get(field), field)
