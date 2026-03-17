"""
PyTorch FX -> IR direct conversion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import torch
import torch.fx

from CIMA_TC.Compiler.IR_tool.core.ir import BaseIR, save_ir
from CIMA_TC.Compiler.IR_tool.core.jsonable import SerializationConfig

from CIMA_TC.Compiler.frontend.utils.weight_export import export_weights as _export_weights_impl
from CIMA_TC.Compiler.frontend.utils.ir_rewrite import fuse_sigmoid_mul_to_silu, rename_batchnorm_layers

from .config import FXConversionConfig
from .handlers import FX_OP_HANDLERS
from .shape_utils import get_shape_from_meta, shape_to_input_info, shape_to_output_info


def _check_torch() -> None:
    try:
        import torch.fx
    except ImportError as e:
        raise RuntimeError("PyTorch with FX is required. Install: pip install torch") from e


def _get_op_type(target: Any) -> str:
    """Get aten op name string from target."""
    if hasattr(target, "__name__"):
        return str(target.__name__)
    return str(target)


_OP_TO_LAYER_NAME = {
    "aten.conv2d.default": "Conv",
    "aten.relu.default": "Relu",
    "aten.adaptive_avg_pool2d.default": "AdaptiveAvgPool2d",
    "aten.flatten.using_ints": "Flatten",
    "aten.flatten": "Flatten",
    "aten.addmm.default": "Linear",
    "aten.linear.default": "Linear",
    "aten.max_pool2d.default": "MaxPool2d",
    "aten.sigmoid.default": "Sigmoid",
}

# call_module: nn.Module class -> layer name prefix
_MODULE_TO_LAYER_NAME = {
    "Linear": "Linear",
    "Conv2d": "Conv",
    "ReLU": "Relu",
    "AdaptiveAvgPool2d": "AdaptiveAvgPool2d",
    "Flatten": "Flatten",
}


class _FXContext:
    """Context for FX -> IR conversion."""

    def __init__(self, gm: torch.fx.GraphModule, prefix: str = ""):
        self.gm = gm
        self.prefix = prefix
        self.shape_map: Dict[str, List[int]] = {}
        self.name_map: Dict[str, str] = {}
        self.placeholder_order: List[str] = []
        self._op_count: Dict[str, int] = {}

    def _make_layer_name(self, node: torch.fx.Node, aten_op: str) -> str:
        """Generate IR-safe layer name from aten op."""
        base = _OP_TO_LAYER_NAME.get(aten_op, aten_op.split(".")[0].replace("aten_", "").title())
        cnt = self._op_count.get(base, 0)
        self._op_count[base] = cnt + 1
        name = f"{base}_{cnt}" if cnt > 0 else base
        return self.prefix + name


def _handle_call_method(ir: Any, ctx: _FXContext, node: torch.fx.Node) -> Optional[str]:
    """Handle call_method: flatten, etc. Returns layer name or None."""
    import CIMA_TC.Compiler.IR_tool.ops  # noqa: F401
    from CIMA_TC.Compiler.IR_tool.core import make_op
    from .shape_utils import shape_to_input_info, shape_to_output_info

    target = str(node.target)
    if target != "flatten":
        raise NotImplementedError(f"Unsupported call_method: {target} (node {node.name})")

    in_node = node.args[0] if node.args else None
    start_dim = node.args[1] if len(node.args) > 1 else 1
    in_shape = ctx.shape_map.get(in_node.name) if in_node and hasattr(in_node, "name") else None
    out_shape = ctx.shape_map.get(node.name)
    if not in_shape:
        in_shape = out_shape
    if not out_shape:
        out_shape = in_shape
    if not in_shape or not out_shape:
        raise ValueError(f"call_method flatten {node.name}: missing shape")

    ref = ""
    if in_node and hasattr(in_node, "name"):
        if hasattr(in_node, "op") and in_node.op == "placeholder":
            order = getattr(ctx, "placeholder_order", [])
            idx = order.index(in_node.name) if in_node.name in order else 0
            ref = f"graph_input:{idx}"
        else:
            ref = ctx.name_map.get(in_node.name, in_node.name)

    layer_name = ctx._make_layer_name(node, "Flatten")
    op = make_op("flatten", start_dim=start_dim)
    ir.add_layer(
        layer_name,
        type="op",
        op=op,
        inputs=[shape_to_input_info(in_shape, ref)],
        outputs=shape_to_output_info(out_shape),
    )
    return layer_name


def _handle_call_module(ir: Any, ctx: _FXContext, node: torch.fx.Node, gm: torch.fx.GraphModule) -> Optional[str]:
    """Handle call_module: nn.Linear, nn.Conv2d, nn.ReLU, nn.AdaptiveAvgPool2d. Returns layer name or None."""
    import CIMA_TC.Compiler.IR_tool.ops  # noqa: F401
    from CIMA_TC.Compiler.IR_tool.core import make_op
    from .shape_utils import shape_to_input_info, shape_to_output_info, get_weight_info

    try:
        submod = gm.get_submodule(node.target)
    except Exception:
        return None
    mod_type = type(submod).__name__
    layer_base = _MODULE_TO_LAYER_NAME.get(mod_type)
    if layer_base is None:
        raise NotImplementedError(f"Unsupported call_module: {mod_type} (node {node.name})")

    in_node = node.args[0] if node.args else None
    in_shape = ctx.shape_map.get(in_node.name) if in_node and hasattr(in_node, "name") else None
    out_shape = ctx.shape_map.get(node.name)
    if not in_shape:
        in_shape = out_shape
    if not out_shape:
        out_shape = in_shape
    if not in_shape or not out_shape:
        raise ValueError(f"call_module {node.name} ({mod_type}): missing shape")

    ref = ""
    if in_node and hasattr(in_node, "name"):
        if hasattr(in_node, "op") and in_node.op == "placeholder":
            order = getattr(ctx, "placeholder_order", [])
            idx = order.index(in_node.name) if in_node.name in order else 0
            ref = f"graph_input:{idx}"
        else:
            ref = ctx.name_map.get(in_node.name, in_node.name)

    layer_name = ctx._make_layer_name(node, mod_type)

    if mod_type == "Linear":
        in_ch, out_ch = submod.in_features, submod.out_features
        bias = submod.bias is not None
        op = make_op("linear", in_channel=in_ch, out_channel=out_ch, bias=bias)
        weight_shape = [out_ch, in_ch]
        bias_shape = [out_ch] if bias else None
        ir.add_layer(
            layer_name,
            type="op",
            op=op,
            inputs=[shape_to_input_info(in_shape, ref)],
            outputs=shape_to_output_info(out_shape),
            weights=get_weight_info(weight_shape, bias_shape),
        )
    elif mod_type == "Conv2d":
        out_ch = submod.out_channels
        in_ch = submod.in_channels
        k = submod.kernel_size
        kernel = k[0] if isinstance(k, (tuple, list)) else k
        s = submod.stride
        stride = s[0] if isinstance(s, (tuple, list)) else s
        p = submod.padding
        padding = p[0] if isinstance(p, (tuple, list)) else p
        op = make_op(
            "conv2d",
            in_channel=in_ch,
            out_channel=out_ch,
            kernel=kernel,
            stride=stride,
            padding=padding,
            bias=submod.bias is not None,
        )
        weight_shape = [out_ch, in_ch, kernel, kernel] if isinstance(kernel, int) else list(submod.weight.shape)
        bias_shape = list(submod.bias.shape) if submod.bias is not None else None
        ir.add_layer(
            layer_name,
            type="op",
            op=op,
            inputs=[shape_to_input_info(in_shape, ref)],
            outputs=shape_to_output_info(out_shape),
            weights=get_weight_info(weight_shape, bias_shape),
        )
    elif mod_type == "ReLU":
        op = make_op("relu")
        ir.add_layer(
            layer_name,
            type="op",
            op=op,
            inputs=[shape_to_input_info(in_shape, ref)],
            outputs=shape_to_output_info(out_shape),
        )
    elif mod_type == "AdaptiveAvgPool2d":
        op = make_op("global_avg_pool2d")
        ir.add_layer(
            layer_name,
            type="op",
            op=op,
            inputs=[shape_to_input_info(in_shape, ref)],
            outputs=shape_to_output_info(out_shape),
        )
    else:
        return None
    return layer_name


def _run_shape_prop(gm: torch.fx.GraphModule, example_input: Any) -> None:
    """Run shape propagation on the graph."""
    try:
        from torch.fx.passes.shape_prop import ShapeProp
        ShapeProp(gm).propagate(example_input)
    except Exception:
        pass  # Some PyTorch versions may not have ShapeProp or API differs


def _collect_shapes(ctx: _FXContext) -> None:
    """Populate ctx.shape_map from node meta."""
    for node in ctx.gm.graph.nodes:
        shape = get_shape_from_meta(node)
        if shape is not None:
            try:
                ctx.shape_map[node.name] = [int(s) for s in shape]
            except (TypeError, ValueError):
                pass
        # get_attr: get shape from module parameters/buffers
        if node.op == "get_attr":
            try:
                target = str(node.target)
                val = ctx.gm
                for part in target.split("."):
                    val = getattr(val, part)
                if hasattr(val, "shape"):
                    ctx.shape_map[node.name] = [int(s) for s in val.shape]
            except Exception:
                pass


class ConvertFX:
    """
    Convert PyTorch nn.Module to IR via torch.fx (no ONNX).
    """

    def __init__(
        self,
        config: Optional[FXConversionConfig] = None,
        **kwargs: Any,
    ) -> None:
        _check_torch()
        if config is not None:
            self.config = config
        else:
            allowed = {"example_input", "ir_file", "layer_name_prefix"}
            self.config = FXConversionConfig(**{k: v for k, v in kwargs.items() if k in allowed})
        self.ir: Optional[BaseIR] = None
        self._state_dict: Optional[Dict[str, Any]] = None

    def convert(
        self,
        model: torch.nn.Module,
        example_input: Optional[Any] = None,
    ) -> BaseIR:
        """
        Convert PyTorch model to IR using torch.fx.
        """
        cfg = self.config
        inp = example_input if example_input is not None else cfg.example_input
        if inp is None:
            raise ValueError("Provide example_input or set config.example_input")

        # Trace
        gm = torch.fx.symbolic_trace(model)
        _run_shape_prop(gm, inp)
        self._state_dict = {k: v.clone() for k, v in model.state_dict().items()}

        ctx = _FXContext(gm, prefix=cfg.layer_name_prefix)
        _collect_shapes(ctx)

        self.ir = BaseIR.make_ir()

        # Placeholders -> graph_input
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        ctx.placeholder_order = [p.name for p in placeholders]
        g_inputs = []
        for i, ph in enumerate(placeholders):
            shape = ctx.shape_map.get(ph.name)
            if shape and len(shape) == 4:
                g_inputs.append(dict(channel=shape[1], height=shape[2], width=shape[3], channel_last=True))
            elif shape and len(shape) == 2:
                g_inputs.append(dict(channel=shape[1], height=1, width=1, channel_last=True))
            else:
                g_inputs.append(dict(channel=1, height=1, width=1, channel_last=True))
        self.ir.add_layer("graph_input", type="input", outputs=g_inputs)

        # Build ref for placeholder consumers
        for ph in placeholders:
            ctx.name_map[ph.name] = "graph_input"

        # call_function, call_module, call_method nodes -> op layers
        for node in gm.graph.nodes:
            if node.op == "call_function":
                op_name = _get_op_type(node.target)
                handler = FX_OP_HANDLERS.get(op_name)
                if handler is None:
                    raise NotImplementedError(f"Unsupported FX op: {op_name} (node {node.name})")
                layer_name = ctx._make_layer_name(node, op_name)
                ctx.name_map[node.name] = layer_name
                handler(self.ir, ctx, node)
            elif node.op == "call_module":
                layer_name = _handle_call_module(self.ir, ctx, node, gm)
                if layer_name is not None:
                    ctx.name_map[node.name] = layer_name
            elif node.op == "call_method":
                layer_name = _handle_call_method(self.ir, ctx, node)
                if layer_name is not None:
                    ctx.name_map[node.name] = layer_name

        # Fix input refs: we assigned name_map after adding layers, but handlers run in order
        # so when conv runs, relu's name_map might not be set yet. Actually we process in
        # graph order - so when we process conv2d, we've already processed its input (relu).
        # The issue: we set ctx.name_map[node.name] = layer_name INSIDE the loop, but we call
        # handler which uses _get_input_ref. So when we process node B which consumes A,
        # we've already processed A and set name_map[A.name] = layer_A. Good.

        # But wait - we set name_map for the current node AFTER we might need it. Actually
        # _get_input_ref looks at node.args[arg_idx] - the producer. The producer was
        # processed in a previous iteration. So when we process conv2d, its input is relu.
        # We've already processed relu and set ctx.name_map[relu.name] = "Relu_1". Good.
        # But we set ctx.name_map[node.name] = layer_name at the start of the handler call...
        # No, we set it right before calling handler. So when handler runs, the current node's
        # name_map is set. But the producer's name_map was set when we processed the producer.
        # So we need to set name_map for the current node BEFORE calling handler, because
        # the handler might not need it - it's for future consumers. So the order is correct.

        # Output -> graph_output
        output_node = None
        for node in gm.graph.nodes:
            if node.op == "output":
                output_node = node
                break
        if output_node is not None and output_node.args:
            out_args = output_node.args[0]
            if isinstance(out_args, (tuple, list)):
                out_tensors = list(out_args)
            else:
                out_tensors = [out_args]
            g_outputs = []
            for t in out_tensors:
                if hasattr(t, "name"):
                    ref = ctx.name_map.get(t.name, t.name)
                    shape = ctx.shape_map.get(t.name)
                    if shape and len(shape) == 4:
                        g_outputs.append(dict(ref=ref, channel=shape[1], height=shape[2], width=shape[3], channel_last=True))
                    elif shape and len(shape) == 2:
                        g_outputs.append(dict(ref=ref, channel=shape[1], height=1, width=1, channel_last=True))
                    else:
                        g_outputs.append(dict(ref=ref, shape=shape or []))
                else:
                    g_outputs.append(dict(ref="", shape=[]))
            self.ir.add_layer("graph_output", type="output", inputs=g_outputs)

        # Frontend-shared rewrites (default enabled)
        if getattr(self.config, "enable_ir_rewrite", True):
            fuse_sigmoid_mul_to_silu(self.ir)
            rename_batchnorm_layers(self.ir)

        return self.ir

    def dump(
        self,
        ir_file: Optional[Union[str, Path]] = None,
        *,
        as_yaml: bool = True,
    ) -> Optional[str]:
        """Write IR to file. Returns string if ir_file is None."""
        if self.ir is None:
            raise RuntimeError("Run convert() first")
        path = ir_file or getattr(self.config, "ir_file", None)
        kwargs = {}
        if as_yaml:
            kwargs["config"] = SerializationConfig(default_flow_style=False)
        if path is not None:
            save_ir(self.ir, file=str(path), **kwargs)
            return None
        return save_ir(self.ir, **kwargs)

    def export_weights(
        self,
        path: Union[str, Path],
        state_dict: Optional[Dict[str, Any]] = None,
        *,
        format: Optional[str] = None,
    ) -> None:
        """
        Export PyTorch state_dict (conv/fc weights, BN parameters, etc.) to a separate file.
        format: None（按扩展名推断）/ "pt" / "npz" / "npy"（单数组时）。详见 utils.weight_export.export_weights。
        """
        sd = state_dict if state_dict is not None else self._state_dict
        if sd is None:
            raise RuntimeError("Run convert(model) first so state_dict is captured, or pass state_dict=...")
        _export_weights_impl(sd, path, format=format)
