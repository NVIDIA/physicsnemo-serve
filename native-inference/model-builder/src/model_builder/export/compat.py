"""Reusable, explicitly selected graph compatibility passes."""


class DoMINOExactBoundaryPass:
    """Route captured FP32 arithmetic through the tensor-only ATen sidecar.

    The eager reference remains untouched. Tensor-only schemas avoid AOTI proxy
    scalar conversions while retaining PyTorch's original arithmetic order.
    """

    def __call__(self, module):
        import torch

        aten = torch.ops.aten
        ops = torch.ops.pnmir_domino
        scalar_ops = {
            aten.add.Tensor: "tensor_scalar_add",
            aten.div.Tensor: "tensor_scalar_div",
            aten.mul.Tensor: "tensor_scalar_mul",
        }
        count = 0
        for node in list(module.graph.nodes):
            if node.op != "call_function" or not node.args:
                continue
            source = node.args[0]
            value = (
                source.meta.get("val") if isinstance(source, torch.fx.Node) else None
            )
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
                continue
            replacement = None
            args = node.args
            if node.target in scalar_ops and len(args) == 2:
                scalar = args[1]
                if type(scalar) not in (int, float) or node.kwargs.get("alpha", 1) != 1:
                    continue
                with module.graph.inserting_before(node):
                    constant = module.graph.call_function(
                        aten.scalar_tensor.default,
                        (scalar,),
                        {"dtype": value.dtype, "device": value.device},
                    )
                constant.meta = node.meta.copy()
                constant.meta["val"] = value.new_full((), scalar)
                args = (source, constant)
                replacement = scalar_ops[node.target]
            elif node.target == aten.reciprocal.default:
                replacement = "reciprocal"
            elif node.target == aten.sub.Tensor and len(args) == 2:
                other = (
                    args[1].meta.get("val")
                    if isinstance(args[1], torch.fx.Node)
                    else None
                )
                if (
                    not isinstance(other, torch.Tensor)
                    or other.shape != value.shape
                    or other.dtype != value.dtype
                    or node.kwargs.get("alpha", 1) != 1
                ):
                    continue
                replacement = "tensor_sub"
            elif node.target == aten.linalg_vector_norm.default:
                order = args[1] if len(args) > 1 else node.kwargs.get("ord", 2)
                dims = args[2] if len(args) > 2 else node.kwargs.get("dim")
                keepdim = (
                    args[3] if len(args) > 3 else node.kwargs.get("keepdim", False)
                )
                if (
                    order != 2
                    or dims not in ([-1], (-1,))
                    or keepdim is not True
                    or node.kwargs.get("dtype") is not None
                ):
                    continue
                replacement, args = "vector_norm_last_dim", (source,)
            if replacement is not None:
                node.target = getattr(ops, replacement).default
                node.args, node.kwargs = args, {}
                count += 1
        module.graph.lint()
        module.recompile()
        return count


class FreezeScalarSigmoidGates:
    """Evaluate scalar parameter gates with PyTorch before ONNX constant folding.

    GeoTransolver's GALE mixing gates must use the reference device's sigmoid
    rounding. Only captured scalar parameters are constants; input-dependent
    sigmoids and vector parameters remain in the graph. The live model and the
    captured state dictionary keep their original logits.
    """

    def __call__(self, module):
        import torch

        changed = 0
        for node in list(module.graph.nodes):
            if (
                node.op != "call_function"
                or node.target != torch.ops.aten.sigmoid.default
            ):
                continue
            source = node.args[0] if node.args else node.kwargs.get("self")
            value = (
                source.meta.get("pnmir_parameter_value")
                if isinstance(source, torch.fx.Node) and source.op == "placeholder"
                else None
            )
            if not isinstance(value, torch.Tensor) or value.ndim != 0:
                continue
            if value.dtype != torch.float32 or not bool(torch.isfinite(value)):
                raise ValueError(
                    "FreezeScalarSigmoidGates requires finite FP32 scalar parameters"
                )
            with torch.no_grad():
                gate = torch.sigmoid(value.detach()).item()
            node.target = torch.ops.aten.scalar_tensor.default
            node.args = (gate,)
            node.kwargs = {"dtype": value.dtype, "device": value.device}
            changed += 1
        module.graph.lint()
        module.recompile()
        return changed


class NormalizeClampBounds:
    """Normalize scalar bounds in captured FP32 clamps to tensor bounds.

    ONNX decomposition can turn mixed float/integer bounds into mixed
    scalar/tensor bounds without changing aten.clamp.default's overload.
    Materialize scalars on the input's device before decomposition, preserving
    the tensor bound, broadcasting and clamp's lower-then-upper ordering.
    """

    def __call__(self, module):
        import torch

        changed = 0
        for node in list(module.graph.nodes):
            if node.op != "call_function" or node.target not in (
                torch.ops.aten.clamp.default,
                torch.ops.aten.clamp.Tensor,
            ):
                continue
            value = node.args[0] if node.args else node.kwargs.get("self")
            lower = node.args[1] if len(node.args) > 1 else node.kwargs.get("min")
            upper = node.args[2] if len(node.args) > 2 else node.kwargs.get("max")
            bounds = [lower, upper]
            scalar_indices = [
                i for i, bound in enumerate(bounds) if type(bound) in (int, float)
            ]
            tensor_indices = [
                i for i, bound in enumerate(bounds) if isinstance(bound, torch.fx.Node)
            ]
            if not scalar_indices or any(
                bound is not None
                and type(bound) not in (int, float)
                and not isinstance(bound, torch.fx.Node)
                for bound in bounds
            ):
                continue
            source = value.meta.get("val") if isinstance(value, torch.fx.Node) else None
            tensor_values = [bounds[i].meta.get("val") for i in tensor_indices]
            if (
                not isinstance(source, torch.Tensor)
                or source.dtype != torch.float32
                or any(
                    not isinstance(bound, torch.Tensor)
                    or bound.dtype != source.dtype
                    or bound.device != source.device
                    for bound in tensor_values
                )
            ):
                raise ValueError(
                    f"NormalizeClampBounds: {node.name} requires FP32 tensor metadata on the same device"
                )
            for index in scalar_indices:
                scalar = bounds[index]
                with module.graph.inserting_before(node):
                    constant = module.graph.call_function(
                        torch.ops.aten.scalar_tensor.default,
                        (scalar,),
                        {"dtype": source.dtype, "device": source.device},
                    )
                constant.meta = node.meta.copy()
                constant.meta["val"] = source.new_full((), scalar)
                bounds[index] = constant
            node.target = torch.ops.aten.clamp.Tensor
            node.args = (value, *bounds)
            node.kwargs = {}
            changed += 1
        module.graph.lint()
        module.recompile()
        return changed
