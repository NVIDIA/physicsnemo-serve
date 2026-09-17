"""Named, scoped AOTInductor compilation profiles."""

from contextlib import contextmanager
import hashlib
import marshal
import sys

EXACT_PROFILES = ("aten-boundary-exact-v2", "aten-boundary-exact-v3")


def validate_aoti_profile(name) -> str:
    if not isinstance(name, str) or name not in ("baseline", *EXACT_PROFILES):
        raise ValueError("AOTI profile must be baseline, aten-boundary-exact-v2 or aten-boundary-exact-v3")
    return name


def _make_exact_pass():
    import torch
    from torch._inductor.custom_graph_pass import CustomGraphPass

    aten = torch.ops.aten

    class ExactPass(CustomGraphPass):
        def __call__(self, graph):
            for node in list(graph.nodes):
                if node.op != "call_function":
                    continue
                if node.target is aten.addmm.default:
                    if (
                        node.kwargs.get("alpha", 1) != 1
                        or node.kwargs.get("beta", 1) != 1
                    ):
                        continue
                    bias, value, transposed = node.args[:3]
                elif node.target is aten.mm.default:
                    value, transposed = node.args[:2]
                    bias = None
                else:
                    continue
                if (
                    not isinstance(transposed, torch.fx.Node)
                    or transposed.op != "call_function"
                    or transposed.target is not aten.t.default
                ):
                    continue
                weight = transposed.args[0]
                if not isinstance(weight, torch.fx.Node) or weight.op != "get_attr":
                    continue
                with graph.inserting_before(node):
                    linear = graph.call_function(
                        aten.linear.default, (value, weight, bias)
                    )
                linear.meta = node.meta.copy()
                node.replace_all_uses_with(linear)
                graph.erase_node(node)
            graph.eliminate_dead_code()
            for node in graph.nodes:
                scalar_full = (
                    node.target is aten.full.default
                    and node.args
                    and isinstance(node.args[0], (tuple, list))
                    and not node.args[0]
                )
                if (
                    node.target in (aten.clone.default, aten._unsafe_view.default)
                    or scalar_full
                ):
                    node.meta.setdefault("custom", {})["compile_with_inductor"] = {}
            graph.lint()

        def uuid(self):
            return "pnmir-aten-boundary-exact-v2"

    return ExactPass()


@contextmanager
def compiler_profile(name="baseline"):
    name = validate_aoti_profile(name)
    metadata = {
        "name": name,
        "version": {"baseline": 1, "aten-boundary-exact-v2": 2, "aten-boundary-exact-v3": 3}[name],
        "requested_compiler_settings": {},
        "applied_compiler_settings": {},
        "graph_pass_uuid": None,
    }
    if name == "baseline":
        yield metadata
        return

    import torch
    from torch._inductor import config

    settings = {
        # The misspelling is the actual upstream Torch configuration key.
        "emulate_divison_rounding": True,
        "fallback_by_default": True,
        "selective_decompose": True,
        "post_grad_custom_pre_pass": None,
    }
    if name == "aten-boundary-exact-v3":
        settings["shape_padding"] = False
    missing = [key for key in settings if not hasattr(config, key)]
    if missing:
        raise ValueError(
            f"AOTI profile {name} requires unavailable Torch {torch.__version__} "
            f"compiler settings: {', '.join(missing)}"
        )
    graph_pass = _make_exact_pass()
    settings["post_grad_custom_pre_pass"] = graph_pass
    recorded_settings = {**settings, "post_grad_custom_pre_pass": graph_pass.uuid()}
    metadata.update(
        requested_compiler_settings=recorded_settings,
        graph_pass_uuid=graph_pass.uuid(),
        graph_pass_sha256=hashlib.sha256(
            marshal.dumps(type(graph_pass).__call__.__code__)
        ).hexdigest(),
        graph_pass_identity_format="python-code-marshal-v1",
        compiler={
            "python_version": sys.version,
            "torch_version": str(torch.__version__),
            "torch_git_version": torch.version.git_version,
            "cuda_version": torch.version.cuda,
        },
    )
    with config.patch(settings):
        metadata["applied_compiler_settings"] = dict(recorded_settings)
        yield metadata
