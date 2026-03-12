"""
FX aten op -> IR layer handlers.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import CIMA_TC.Compiler.IR_tool.ops  # noqa: F401
from CIMA_TC.Compiler.IR_tool.core import make_op

from .shape_utils import (
    get_shape_from_meta,
    shape_to_input_info,
    shape_to_output_info,
    get_weight_info,
)

FX_OP_HANDLERS: Dict[str, Callable[..., None]] = {}


def register_fx_op(*names: str):
    """Decorator to register handler for aten op(s)."""

    def deco(fn: Callable[..., None]):
        for n in names:
            FX_OP_HANDLERS[n] = fn
        return fn

    return deco


def _get_input_ref(ctx: Any, node: Any, arg_idx: int = 0) -> str:
    """Resolve FX node arg to producer layer name (graph_input:0 or layer name)."""
    args = list(node.args)
    if arg_idx >= len(args):
        return ""
    arg = args[arg_idx]
    if not hasattr(arg, "name"):
        return ""
    if hasattr(arg, "op") and arg.op == "placeholder":
        order = getattr(ctx, "placeholder_order", [])
        idx = order.index(arg.name) if arg.name in order else 0
        return f"graph_input:{idx}"
    return ctx.name_map.get(arg.name, arg.name)


def _get_shape(ctx: Any, node: Any) -> Optional[list]:
    """Get output shape of node from meta or ctx.shape_map."""
    shape = get_shape_from_meta(node)
    if shape is not None:
        return shape
    return ctx.shape_map.get(node.name) if hasattr(ctx, "shape_map") else None


# ---------- aten.conv2d.default ----------
@register_fx_op("aten.conv2d.default")
def _conv2d(ir: Any, ctx: Any, node: Any) -> None:
    args = node.args
    inp = args[0]
    weight = args[1]
    bias = args[2] if len(args) > 2 else None
    stride = args[3] if len(args) > 3 else (1, 1)
    padding = args[4] if len(args) > 4 else (0, 0)
    dilation = args[5] if len(args) > 5 else (1, 1)
    groups = args[6] if len(args) > 6 else 1

    in_shape = ctx.shape_map.get(inp.name) if hasattr(inp, "name") else None
    out_shape = _get_shape(ctx, node)
    weight_shape = ctx.shape_map.get(weight.name) if hasattr(weight, "name") else None
    bias_shape = None
    if bias is not None and hasattr(bias, "name"):
        bias_shape = ctx.shape_map.get(bias.name)
        if bias_shape and (len(bias_shape) == 1 or all(s == 1 for s in bias_shape)):
            bias_shape = list(bias_shape)
        else:
            bias_shape = None

    if not in_shape or not out_shape or not weight_shape:
        raise ValueError(f"conv2d {node.name}: missing shape info")

    stride_h = stride[0] if isinstance(stride, (list, tuple)) else stride
    stride_w = stride[1] if isinstance(stride, (list, tuple)) and len(stride) > 1 else stride_h
    pad_h = padding[0] if isinstance(padding, (list, tuple)) else padding
    pad_w = padding[1] if isinstance(padding, (list, tuple)) and len(padding) > 1 else pad_h
    dil_h = dilation[0] if isinstance(dilation, (list, tuple)) else dilation
    dil_w = dilation[1] if isinstance(dilation, (list, tuple)) and len(dilation) > 1 else dil_h

    op = make_op(
        "conv2d",
        in_channel=weight_shape[1],
        out_channel=weight_shape[0],
        kernel=weight_shape[2] if len(weight_shape) >= 3 else 3,
        stride=stride_h,
        padding=pad_h,
        bias=bias_shape is not None,
        dilation=dil_h,
    )
    ref = _get_input_ref(ctx, node, 0)
    if not ref:
        ref = f"graph_input:0"
    inputs = [shape_to_input_info(in_shape, ref)]
    outputs = shape_to_output_info(out_shape)
    weights = get_weight_info(tuple(weight_shape), tuple(bias_shape) if bias_shape else None)
    layer_name = ctx.name_map.get(node.name, node.name)
    ir.add_layer(layer_name, type="op", op=op, inputs=inputs, outputs=outputs, weights=weights)


# ---------- aten.relu.default ----------
@register_fx_op("aten.relu.default")
def _relu(ir: Any, ctx: Any, node: Any) -> None:
    in_shape = _get_shape(ctx, node.args[0]) if hasattr(node.args[0], "name") else None
    out_shape = _get_shape(ctx, node)
    if not in_shape:
        in_shape = out_shape
    if not out_shape:
        out_shape = in_shape
    if not in_shape or not out_shape:
        raise ValueError(f"relu {node.name}: missing shape")

    op = make_op("relu")
    ref = _get_input_ref(ctx, node, 0)
    inputs = [shape_to_input_info(in_shape, ref)]
    outputs = shape_to_output_info(out_shape)
    layer_name = ctx.name_map.get(node.name, node.name)
    ir.add_layer(layer_name, type="op", op=op, inputs=inputs, outputs=outputs)


# ---------- aten.adaptive_avg_pool2d.default ----------
@register_fx_op("aten.adaptive_avg_pool2d.default")
def _adaptive_avg_pool2d(ir: Any, ctx: Any, node: Any) -> None:
    inp = node.args[0]
    out_shape = _get_shape(ctx, node)
    in_shape = ctx.shape_map.get(inp.name) if hasattr(inp, "name") else out_shape
    if not out_shape:
        out_shape = in_shape
    if not in_shape or not out_shape:
        raise ValueError(f"adaptive_avg_pool2d {node.name}: missing shape")

    op = make_op("global_avg_pool2d")
    ref = _get_input_ref(ctx, node, 0)
    inputs = [shape_to_input_info(in_shape, ref)]
    outputs = shape_to_output_info(out_shape)
    layer_name = ctx.name_map.get(node.name, node.name)
    ir.add_layer(layer_name, type="op", op=op, inputs=inputs, outputs=outputs)


# ---------- aten.flatten.using_ints / aten.flatten ----------
@register_fx_op("aten.flatten.using_ints", "aten.flatten")
def _flatten(ir: Any, ctx: Any, node: Any) -> None:
    inp = node.args[0]
    start_dim = node.args[1] if len(node.args) > 1 else 1
    end_dim = node.args[2] if len(node.args) > 2 else -1
    in_shape = ctx.shape_map.get(inp.name) if hasattr(inp, "name") else None
    out_shape = _get_shape(ctx, node)
    if not out_shape:
        out_shape = in_shape
    if not in_shape or not out_shape:
        raise ValueError(f"flatten {node.name}: missing shape")

    op = make_op("flatten", start_dim=start_dim)
    ref = _get_input_ref(ctx, node, 0)
    inputs = [shape_to_input_info(in_shape, ref)]
    outputs = shape_to_output_info(out_shape)
    layer_name = ctx.name_map.get(node.name, node.name)
    ir.add_layer(layer_name, type="op", op=op, inputs=inputs, outputs=outputs)


# ---------- aten.addmm.default (bias + input @ weight) -> linear ----------
@register_fx_op("aten.addmm.default")
def _addmm(ir: Any, ctx: Any, node: Any) -> None:
    # addmm(bias, input, weight) -> bias + input @ weight^T
    bias = node.args[0]
    inp = node.args[1]
    weight = node.args[2]
    in_shape = ctx.shape_map.get(inp.name) if hasattr(inp, "name") else None
    out_shape = _get_shape(ctx, node)
    weight_shape = ctx.shape_map.get(weight.name) if hasattr(weight, "name") else None
    bias_shape = ctx.shape_map.get(bias.name) if hasattr(bias, "name") else None
    if not in_shape or not out_shape or not weight_shape:
        raise ValueError(f"addmm {node.name}: missing shape")

    in_ch, out_ch = weight_shape[1], weight_shape[0]
    if bias_shape is None:
        bias_shape = [out_ch]
    elif len(bias_shape) > 1:
        bias_shape = [out_ch]
    op = make_op("linear", in_channel=in_ch, out_channel=out_ch, bias=True)
    ref = _get_input_ref(ctx, node, 1)
    inputs = [shape_to_input_info(in_shape, ref)]
    outputs = shape_to_output_info(out_shape)
    weights = get_weight_info(weight_shape, bias_shape)
    layer_name = ctx.name_map.get(node.name, node.name)
    ir.add_layer(layer_name, type="op", op=op, inputs=inputs, outputs=outputs, weights=weights)


# ---------- aten.linear.default ----------
@register_fx_op("aten.linear.default")
def _linear(ir: Any, ctx: Any, node: Any) -> None:
    inp = node.args[0]
    weight = node.args[1]
    bias = node.args[2] if len(node.args) > 2 else None
    in_shape = ctx.shape_map.get(inp.name) if hasattr(inp, "name") else None
    out_shape = _get_shape(ctx, node)
    weight_shape = ctx.shape_map.get(weight.name) if hasattr(weight, "name") else None
    if not in_shape or not out_shape or not weight_shape:
        raise ValueError(f"linear {node.name}: missing shape")

    in_ch, out_ch = weight_shape[1], weight_shape[0]
    bias_shape = None
    if bias is not None and hasattr(bias, "name"):
        bias_shape = ctx.shape_map.get(bias.name)
    if bias_shape is None and bias is not None:
        bias_shape = [out_ch]
    op = make_op("linear", in_channel=in_ch, out_channel=out_ch, bias=bias_shape is not None)
    ref = _get_input_ref(ctx, node, 0)
    inputs = [shape_to_input_info(in_shape, ref)]
    outputs = shape_to_output_info(out_shape)
    weights = get_weight_info(weight_shape, bias_shape)
    layer_name = ctx.name_map.get(node.name, node.name)
    ir.add_layer(layer_name, type="op", op=op, inputs=inputs, outputs=outputs, weights=weights)


# ---------- aten.max_pool2d.default ----------
@register_fx_op("aten.max_pool2d.default")
def _max_pool2d(ir: Any, ctx: Any, node: Any) -> None:
    args = node.args
    inp = args[0]
    kernel = args[1] if len(args) > 1 else (2, 2)
    stride = args[2] if len(args) > 2 else kernel
    padding = args[3] if len(args) > 3 else (0, 0)
    dilation = args[4] if len(args) > 4 else (1, 1)
    ceil_mode = args[5] if len(args) > 5 else False

    in_shape = ctx.shape_map.get(inp.name) if hasattr(inp, "name") else None
    out_shape = _get_shape(ctx, node)
    if not in_shape or not out_shape:
        raise ValueError(f"max_pool2d {node.name}: missing shape")

    k = kernel[0] if isinstance(kernel, (list, tuple)) else kernel
    s = stride[0] if isinstance(stride, (list, tuple)) else stride
    op = make_op("max_pool2d", kernel=[k, k], stride=[s, s])
    ref = _get_input_ref(ctx, node, 0)
    inputs = [shape_to_input_info(in_shape, ref)]
    outputs = shape_to_output_info(out_shape)
    layer_name = ctx.name_map.get(node.name, node.name)
    ir.add_layer(layer_name, type="op", op=op, inputs=inputs, outputs=outputs)


# ---------- aten.sigmoid.default ----------
@register_fx_op("aten.sigmoid.default")
def _sigmoid(ir: Any, ctx: Any, node: Any) -> None:
    in_shape = _get_shape(ctx, node.args[0]) if hasattr(node.args[0], "name") else None
    out_shape = _get_shape(ctx, node)
    if not in_shape:
        in_shape = out_shape
    if not out_shape:
        out_shape = in_shape
    if not in_shape or not out_shape:
        raise ValueError(f"sigmoid {node.name}: missing shape")

    op = make_op("sigmoid")
    ref = _get_input_ref(ctx, node, 0)
    inputs = [shape_to_input_info(in_shape, ref)]
    outputs = shape_to_output_info(out_shape)
    layer_name = ctx.name_map.get(node.name, node.name)
    ir.add_layer(layer_name, type="op", op=op, inputs=inputs, outputs=outputs)
