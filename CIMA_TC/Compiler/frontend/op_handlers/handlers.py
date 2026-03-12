"""
ONNX -> IR layer handlers. Uses IR_tool.core make_op, make_layer; GraphLayer.add_layer.
"""

from __future__ import annotations

import numpy as np
from typing import Any, List

try:
    from onnx import numpy_helper
except ImportError:
    numpy_helper = None

from CIMA_TC.Compiler.IR_tool.core import make_op
from CIMA_TC.Compiler.IR_tool.core.layer import BaseLayer

# Import ops so all BaseOp subclasses (ReluOp, Conv2dOp, etc.) are registered
import CIMA_TC.Compiler.IR_tool.ops  # noqa: F401

from ..utils import dim_to_list, get_input_info, get_output_info, get_weight_info
from ..utils.attr_reader import (
    get_axis,
    get_axes,
    get_perm,
    get_alpha,
    get_keepdims,
    get_conv_node_attr,
    get_conv_node_dilation,
    get_split,
    get_node_pads,
    get_node_epsilon,
    get_pad_mode,
    get_pad_value,
    get_resize_mode,
)
from .common import resolve_ref, ir_inputs_for_node, single_input_output, get_layer_name
from .registry import register_op


def _to_list(val: Any) -> Any:
    if hasattr(val, "tolist"):
        return val.tolist()
    if isinstance(val, (list, tuple)):
        return list(val)
    return val


# ---------- Conv ----------
@register_op("Conv")
def _conv(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    weight_name = node.input[1]
    weight_shape = dim_to_list(parser.value_infos[weight_name].type.tensor_type.shape.dim)
    in_shape = dim_to_list(parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
    out_shape = dim_to_list(parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
    stride, pad, kernel = get_conv_node_attr(node.attribute)
    dilation = get_conv_node_dilation(node.attribute)
    dilation = dilation[0] if isinstance(dilation, (list, tuple)) else dilation
    bias = False
    bias_shape = None
    if len(node.input) == 3:
        bkey = f"{node_name}.bias"
        if bkey in parser.weight_numpy:
            b = parser.weight_numpy[bkey]
            if np.mean(b) != 0 or np.std(b) != 0:
                bias = True
                bias_shape = b.shape
    op = make_op(
        "conv2d",
        in_channel=weight_shape[1],
        out_channel=weight_shape[0],
        kernel=kernel[0] if kernel else 3,
        stride=stride[0] if stride else 1,
        padding=pad[0] if pad else 0,
        bias=bias,
        dilation=dilation,
    )
    ref = resolve_ref(parser, node, node.input[0])
    inputs = [get_input_info(in_shape, ref)]
    outputs = get_output_info(out_shape)
    weights = get_weight_info(weight_shape, bias_shape)
    ir.add_layer(get_layer_name(parser, node_name), type="op", op=op, inputs=inputs, outputs=outputs, weights=weights)


# ---------- ConvTranspose ----------
@register_op("ConvTranspose")
def _conv_transpose(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    weight_name = node.input[1]
    weight_shape = dim_to_list(parser.value_infos[weight_name].type.tensor_type.shape.dim)
    in_shape = dim_to_list(parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
    out_shape = dim_to_list(parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
    stride, pad, kernel = get_conv_node_attr(node.attribute)
    bias = False
    bias_shape = None
    if len(node.input) == 3 and f"{node_name}.bias" in parser.weight_numpy:
        b = parser.weight_numpy[f"{node_name}.bias"]
        if np.mean(b) != 0 or np.std(b) != 0:
            bias = True
            bias_shape = b.shape
    op = make_op(
        "conv_transpose2d",
        in_channel=weight_shape[0],
        out_channel=weight_shape[1],
        kernel=kernel[0] if kernel else 3,
        stride=stride[0] if stride else 1,
        padding=pad[0] if pad else 0,
        bias=bias,
    )
    ref = resolve_ref(parser, node, node.input[0])
    inputs = [get_input_info(in_shape, ref)]
    outputs = get_output_info(out_shape)
    weights = get_weight_info(weight_shape, bias_shape)
    ir.add_layer(get_layer_name(parser, node_name), type="op", op=op, inputs=inputs, outputs=outputs, weights=weights)


# ---------- MatMul: FC/linear (one input + weight) vs matmul (two dynamic inputs) ----------
@register_op("MatMul")
def _matmul(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    in_shape = dim_to_list(parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
    out_shape = dim_to_list(parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
    weight_name = node.input[1]
    ref = resolve_ref(parser, node, node.input[0])

    if weight_name in parser.parameters:
        # Static weight -> linear/fc (one input)
        weight_shape = dim_to_list(parser.value_infos[weight_name].type.tensor_type.shape.dim)
        in_ch, out_ch = weight_shape[1], weight_shape[0]
        op = make_op("linear", in_channel=in_ch, out_channel=out_ch, bias=False)
        inputs = [get_input_info(in_shape, ref)]
        outputs = get_output_info(out_shape)
        weights = get_weight_info(weight_shape)
        ir.add_layer(get_layer_name(parser, node_name), type="op", op=op, inputs=inputs, outputs=outputs, weights=weights)
    else:
        # Two dynamic inputs -> matmul
        w_shape = dim_to_list(parser.value_infos[node.input[1]].type.tensor_type.shape.dim)
        ref_b = resolve_ref(parser, node, node.input[1])
        op = make_op("matmul")
        inputs = [get_input_info(in_shape, ref), get_input_info(w_shape, ref_b)]
        outputs = get_output_info(out_shape)
        ir.add_layer(get_layer_name(parser, node_name), type="op", op=op, inputs=inputs, outputs=outputs)


# ---------- Gemm -> linear ----------
@register_op("Gemm")
def _gemm(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    weight_shape = dim_to_list(parser.value_infos[node.input[1]].type.tensor_type.shape.dim)
    in_shape = dim_to_list(parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
    out_shape = dim_to_list(parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
    bias = False
    bias_shape = None
    if len(node.input) == 3 and f"{node_name}.bias" in parser.weight_numpy:
        b = parser.weight_numpy[f"{node_name}.bias"]
        if np.mean(b) != 0 or np.std(b) != 0:
            bias = True
            bias_shape = b.shape
    op = make_op("linear", in_channel=weight_shape[1], out_channel=weight_shape[0], bias=bias)
    ref = resolve_ref(parser, node, node.input[0])
    inputs = [get_input_info(in_shape, ref)]
    outputs = get_output_info(out_shape)
    weights = get_weight_info(weight_shape, bias_shape)
    ir.add_layer(get_layer_name(parser, node_name), type="op", op=op, inputs=inputs, outputs=outputs, weights=weights)


# ---------- Add (with optional constant injection) ----------
@register_op("Add")
def _add(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    op = make_op("add")
    input_list: List[dict] = []
    const_idx = 0
    for inp in node.input:
        if inp in parser.parameters or inp in parser.constant:
            val = parser.parameters.get(inp) or parser.constant.get(inp)
            if numpy_helper and hasattr(val, "dims"):
                val = numpy_helper.to_array(val)
            arr = np.array(val) if not hasattr(val, "shape") else val
            shape = list(arr.shape) if arr.shape else [1]
            const_name = f"{get_layer_name(parser, node_name)}_const_{const_idx}"
            const_idx += 1
            ir.add_layer(const_name, type="op", op=make_op("constant", value=_to_list(arr)), outputs=get_output_info(shape))
            input_list.append(get_input_info(shape, const_name))
        else:
            shape = dim_to_list(parser.value_infos[inp].type.tensor_type.shape.dim)
            if not shape:
                shape = [1]
            ref = resolve_ref(parser, node, inp)
            input_list.append(get_input_info(shape, ref))
    out_shape = dim_to_list(parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
    ir.add_layer(get_layer_name(parser, node_name), type="op", op=op, inputs=input_list, outputs=get_output_info(out_shape))


# ---------- Constant ----------
@register_op("Constant")
def _constant(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    out_name = node.output[0]
    val = parser.constant.get(out_name) or parser.parameters.get(out_name)
    if val is None:
        for attr in node.attribute:
            if attr.name == "value" and numpy_helper and attr.t:
                val = numpy_helper.to_array(attr.t)
                break
    if val is None:
        raise ValueError(f"Constant node {node_name} has no value")
    value = _to_list(val)
    if value == []:
        value = 0
    out_shape = dim_to_list(parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
    if not out_shape or out_shape == [0]:
        out_shape = [1]
    ir.add_layer(get_layer_name(parser, node_name), type="op", op=make_op("constant", value=value), outputs=get_output_info(out_shape))


# ---------- Unary / single-input (Relu, Sigmoid, Tanh, etc.) ----------
def _unary(ir: Any, parser: Any, node_name: str, op_id: str, **op_kwargs: Any) -> None:
    node = parser.nodes[node_name]
    inputs, outputs = single_input_output(parser, node)
    op = make_op(op_id, **op_kwargs)
    ir.add_layer(get_layer_name(parser, node_name), type="op", op=op, inputs=inputs, outputs=outputs)


@register_op("Relu")
def _relu(ir: Any, parser: Any, node_name: str) -> None:
    _unary(ir, parser, node_name, "relu")


@register_op("Sigmoid")
def _sigmoid(ir: Any, parser: Any, node_name: str) -> None:
    _unary(ir, parser, node_name, "sigmoid")


@register_op("Tanh")
def _tanh(ir: Any, parser: Any, node_name: str) -> None:
    _unary(ir, parser, node_name, "tanh")


@register_op("LeakyRelu")
def _leaky_relu(ir: Any, parser: Any, node_name: str) -> None:
    alpha = get_alpha(parser.nodes[node_name])
    _unary(ir, parser, node_name, "leaky_relu", alpha=alpha)


@register_op("Softmax")
def _softmax(ir: Any, parser: Any, node_name: str) -> None:
    axis = get_axis(parser.nodes[node_name])
    _unary(ir, parser, node_name, "softmax", axis=axis)


@register_op("LogSoftmax")
def _log_softmax(ir: Any, parser: Any, node_name: str) -> None:
    _unary(ir, parser, node_name, "log_softmax")


@register_op("Erf")
def _erf(ir: Any, parser: Any, node_name: str) -> None:
    _unary(ir, parser, node_name, "erf")


# ---------- Mul, Div (binary, with constant injection) ----------
def _binary(ir: Any, parser: Any, node_name: str, op_id: str) -> None:
    node = parser.nodes[node_name]
    input_list = ir_inputs_for_node(parser, node, list(node.input))
    for inp in node.input:
        if inp in parser.parameters or inp in parser.constant:
            val = parser.parameters.get(inp) or parser.constant.get(inp)
            if numpy_helper and hasattr(val, "dims"):
                val = numpy_helper.to_array(val)
            arr = np.array(val) if not hasattr(val, "shape") else val
            shape = list(arr.shape) if arr.shape else [1]
            const_name = f"{get_layer_name(parser, node_name)}_const_{inp}"
            ir.add_layer(const_name, type="op", op=make_op("constant", value=_to_list(arr)), outputs=get_output_info(shape))
            input_list.append(get_input_info(shape, const_name))
            break
    out_shape = dim_to_list(parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
    ir.add_layer(get_layer_name(parser, node_name), type="op", op=make_op(op_id), inputs=input_list, outputs=get_output_info(out_shape))


@register_op("Mul")
def _mul(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    input_list: List[dict] = []
    const_idx = 0
    for inp in node.input:
        if inp in parser.parameters or inp in parser.constant:
            val = parser.parameters.get(inp) or parser.constant.get(inp)
            if numpy_helper and hasattr(val, "dims"):
                val = numpy_helper.to_array(val)
            arr = np.array(val) if not hasattr(val, "shape") else val
            shape = list(arr.shape) if arr.shape else [1]
            const_name = f"{get_layer_name(parser, node_name)}_const_{const_idx}"
            const_idx += 1
            ir.add_layer(const_name, type="op", op=make_op("constant", value=_to_list(arr)), outputs=get_output_info(shape))
            input_list.append(get_input_info(shape, const_name))
        else:
            shape = dim_to_list(parser.value_infos[inp].type.tensor_type.shape.dim) or [1]
            input_list.append(get_input_info(shape, resolve_ref(parser, node, inp)))
    out_shape = dim_to_list(parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
    ir.add_layer(get_layer_name(parser, node_name), type="op", op=make_op("mul"), inputs=input_list, outputs=get_output_info(out_shape))


@register_op("Div")
def _div(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    input_list = ir_inputs_for_node(parser, node, list(node.input))
    div_const_idx = 0
    for inp in node.input:
        if inp in parser.parameters or inp in parser.constant:
            val = parser.parameters.get(inp) or parser.constant.get(inp)
            if numpy_helper and hasattr(val, "dims"):
                val = numpy_helper.to_array(val)
            arr = np.array(val) if not hasattr(val, "shape") else val
            shape = list(arr.shape) if arr.shape else [1]
            const_name = f"{get_layer_name(parser, node_name)}_const_{div_const_idx}"
            div_const_idx += 1
            ir.add_layer(const_name, type="op", op=make_op("constant", value=_to_list(arr)), outputs=get_output_info(shape))
            input_list.append(get_input_info(shape, const_name))
            break
    out_shape = dim_to_list(parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
    ir.add_layer(get_layer_name(parser, node_name), type="op", op=make_op("div"), inputs=input_list, outputs=get_output_info(out_shape))


# ---------- Transpose ----------
@register_op("Transpose")
def _transpose(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    perm = get_perm(node)
    op = make_op("transpose", perm=perm)
    input_list = ir_inputs_for_node(parser, node, list(node.input))
    out_shape = dim_to_list(parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
    ir.add_layer(get_layer_name(parser, node_name), type="op", op=op, inputs=input_list, outputs=get_output_info(out_shape))


# ---------- Reshape ----------
@register_op("Reshape")
def _reshape(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    shape = dim_to_list(parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
    shape = list(shape)
    if shape:
        shape[0] = -1
    op = make_op("reshape", shape=shape)
    input_list = [inp for inp in node.input if inp not in parser.parameters and inp in parser.value_infos]
    inputs = ir_inputs_for_node(parser, node, input_list)
    if not inputs:
        in_shape = dim_to_list(parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
        inputs = [get_input_info(in_shape, resolve_ref(parser, node, node.input[0]))]
    out_shape = dim_to_list(parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
    ir.add_layer(get_layer_name(parser, node_name), type="op", op=op, inputs=inputs, outputs=get_output_info(out_shape))


# ---------- Flatten ----------
@register_op("Flatten")
def _flatten(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    axis = get_axis(node)
    op = make_op("flatten", start_dim=axis)
    inputs, outputs = single_input_output(parser, node)
    ir.add_layer(get_layer_name(parser, node_name), type="op", op=op, inputs=inputs, outputs=outputs)


# ---------- Concat ----------
@register_op("Concat")
def _concat(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    axis = get_axis(node)
    op = make_op("concat", axis=axis)
    input_list: List[dict] = []
    idx = 0
    for inp in node.input:
        if inp in parser.parameters or inp in parser.constant:
            val = parser.parameters.get(inp) or parser.constant.get(inp)
            if numpy_helper and hasattr(val, "dims"):
                val = numpy_helper.to_array(val)
            arr = np.array(val) if not hasattr(val, "shape") else val
            shape = list(arr.shape) if arr.shape else [1]
            const_name = f"{get_layer_name(parser, node_name)}_const_{idx}"
            idx += 1
            ir.add_layer(const_name, type="op", op=make_op("constant", value=_to_list(arr)), outputs=get_output_info(shape))
            input_list.append(get_input_info(shape, const_name))
        else:
            shape = dim_to_list(parser.value_infos[inp].type.tensor_type.shape.dim)
            input_list.append(get_input_info(shape, resolve_ref(parser, node, inp)))
    out_shape = dim_to_list(parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
    ir.add_layer(get_layer_name(parser, node_name), type="op", op=op, inputs=input_list, outputs=get_output_info(out_shape))


# ---------- Pool ----------
@register_op("MaxPool")
def _max_pool(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    stride, pad, kernel = get_conv_node_attr(node.attribute)
    op = make_op("max_pool2d", kernel=kernel[0] if kernel else 2, stride=stride[0] if stride else 2, padding=pad[0] if pad else 0)
    inputs, outputs = single_input_output(parser, node)
    ir.add_layer(get_layer_name(parser, node_name), type="op", op=op, inputs=inputs, outputs=outputs)


@register_op("AveragePool")
def _avg_pool(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    stride, pad, kernel = get_conv_node_attr(node.attribute)
    op = make_op("avg_pool2d", kernel=kernel[0] if kernel else 2, stride=stride[0] if stride else 2, padding=pad[0] if pad else 0)
    inputs, outputs = single_input_output(parser, node)
    ir.add_layer(get_layer_name(parser, node_name), type="op", op=op, inputs=inputs, outputs=outputs)


@register_op("GlobalAveragePool")
def _global_avg_pool(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    in_shape = dim_to_list(parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
    op = make_op("global_avg_pool2d", kernel=in_shape[2] if len(in_shape) > 2 else 1, stride=1, padding=0)
    inputs, outputs = single_input_output(parser, node)
    ir.add_layer(get_layer_name(parser, node_name), type="op", op=op, inputs=inputs, outputs=outputs)


# ---------- Split ----------
@register_op("Split")
def _split(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    axis = get_axis(node)
    if axis == -1:
        in_shape = dim_to_list(parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
        axis = len(in_shape) - 1
    split = get_split(node)
    if split is None and len(node.input) > 1 and node.input[1] in parser.parameters:
        t = parser.parameters[node.input[1]]
        if numpy_helper and hasattr(t, "dims"):
            split = numpy_helper.to_array(t).tolist()
        else:
            split = _to_list(t)
    if split is None:
        raise ValueError(f"Split node {node_name}: split attribute or input shape required")
    op = make_op("split", axis=axis, split=split)
    in_ref = resolve_ref(parser, node, node.input[0])
    in_shape = dim_to_list(parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
    inputs = [get_input_info(in_shape, in_ref)]
    outputs = [get_output_info(dim_to_list(parser.value_infos[o].type.tensor_type.shape.dim))[0] for o in node.output]
    ir.add_layer(get_layer_name(parser, node_name), type="op", op=op, inputs=inputs, outputs=outputs)


# ---------- ReduceMean, Sqrt, Pow ----------
@register_op("ReduceMean")
def _reduce_mean(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    axes = get_axes(node)
    ax = axes[0] if axes else 0
    keepdims = get_keepdims(node)
    op = make_op("reduce_mean", axes=[ax], keepdims=bool(keepdims))
    inputs = ir_inputs_for_node(parser, node, list(node.input))
    out_shape = dim_to_list(parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
    ir.add_layer(get_layer_name(parser, node_name), type="op", op=op, inputs=inputs, outputs=get_output_info(out_shape))


@register_op("Sqrt")
def _sqrt(ir: Any, parser: Any, node_name: str) -> None:
    _unary(ir, parser, node_name, "sqrt")


@register_op("Pow")
def _pow(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    input_list: List[dict] = []
    for inp in node.input:
        if inp in parser.parameters or inp in parser.constant:
            val = parser.parameters.get(inp) or parser.constant.get(inp)
            if numpy_helper and hasattr(val, "dims"):
                val = numpy_helper.to_array(val)
            arr = np.array(val) if not hasattr(val, "shape") else val
            shape = list(arr.shape) if arr.shape else [1]
            const_name = f"{get_layer_name(parser, node_name)}_const_exp"
            ir.add_layer(const_name, type="op", op=make_op("constant", value=_to_list(arr)), outputs=get_output_info(shape))
            input_list.append(get_input_info(shape, const_name))
        else:
            shape = dim_to_list(parser.value_infos[inp].type.tensor_type.shape.dim)
            input_list.append(get_input_info(shape, resolve_ref(parser, node, inp)))
    out_shape = dim_to_list(parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
    ir.add_layer(get_layer_name(parser, node_name), type="op", op=make_op("pow"), inputs=input_list, outputs=get_output_info(out_shape))


# ---------- BatchNormalization, LayerNormalization ----------
@register_op("BatchNormalization")
def _batch_norm(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    in_shape = dim_to_list(parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
    ch = in_shape[1]
    eps = get_node_epsilon(node)
    scale = _to_list(parser.weight_numpy.get(f"{node_name}.weight", [1.0]))
    bias = _to_list(parser.weight_numpy.get(f"{node_name}.bias", [0.0]))
    mean = _to_list(parser.weight_numpy.get(f"{node_name}.running_mean", [0.0]))
    var = _to_list(parser.weight_numpy.get(f"{node_name}.running_var", [1.0]))
    op = make_op("batch_norm2d", channel=ch, epsilon=float(eps), scale=scale, bias=bias, input_mean=mean, input_var=var)
    inputs, outputs = single_input_output(parser, node)
    ir.add_layer(get_layer_name(parser, node_name), type="op", op=op, inputs=inputs, outputs=outputs)


@register_op("LayerNormalization")
def _layer_norm(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    axis = get_axis(node)
    eps = get_node_epsilon(node)
    scale = _to_list(parser.weight_numpy.get(f"{node_name}.weight", [1.0]))
    bias = _to_list(parser.weight_numpy.get(f"{node_name}.bias", [0.0]))
    op = make_op("layer_norm", axis=axis, epsilon=float(eps), scale=scale, bias=bias)
    inputs, outputs = single_input_output(parser, node)
    ir.add_layer(get_layer_name(parser, node_name), type="op", op=op, inputs=inputs, outputs=outputs)


# ---------- Pad, Resize ----------
@register_op("Pad")
def _pad(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    if len(node.input) > 1 and node.input[1] in parser.constant:
        pads = _to_list(parser.constant[node.input[1]])
    else:
        pads = get_node_pads(node)
    mode = get_pad_mode(node)
    if "constant" not in mode.lower():
        raise ValueError(f"Pad mode {mode!r} not supported")
    value = get_pad_value(node)
    op = make_op("pad", pads=pads, value=value)
    inputs, outputs = single_input_output(parser, node)
    ir.add_layer(get_layer_name(parser, node_name), type="op", op=op, inputs=inputs, outputs=outputs)


@register_op("Resize")
def _resize(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    scale: List[float] = []
    if len(node.input) >= 3 and node.input[2] in parser.parameters:
        t = parser.parameters[node.input[2]]
        if numpy_helper and hasattr(t, "dims"):
            scale = numpy_helper.to_array(t).astype(np.int32).tolist()
        else:
            scale = _to_list(t)
    elif len(node.input) >= 2 and node.input[1] and node.input[1] in parser.parameters:
        t = parser.parameters[node.input[1]]
        if numpy_helper and hasattr(t, "dims"):
            scale = numpy_helper.to_array(t).astype(np.int32).tolist()
        else:
            scale = _to_list(t)
    in_shape = dim_to_list(parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
    out_shape = dim_to_list(parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
    if not scale or (hasattr(np, "mean") and np.mean(scale) == 0):
        scale = [out_shape[i] // in_shape[i] if in_shape[i] else 1 for i in range(len(in_shape))]
    mode = get_resize_mode(node)
    op = make_op("resize", scale=scale, mode=mode)
    inputs, outputs = single_input_output(parser, node)
    ir.add_layer(get_layer_name(parser, node_name), type="op", op=op, inputs=inputs, outputs=outputs)


# ---------- Squeeze, Unsqueeze, Gather, Slice ----------
@register_op("Squeeze")
def _squeeze(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    axes = get_axes(node)
    if not axes and len(node.input) > 1 and node.input[1] in parser.parameters:
        t = parser.parameters[node.input[1]]
        if numpy_helper and hasattr(t, "dims"):
            axes = numpy_helper.to_array(t).tolist()
    op = make_op("squeeze", axes=axes or [0])
    inputs, outputs = single_input_output(parser, node)
    ir.add_layer(get_layer_name(parser, node_name), type="op", op=op, inputs=inputs, outputs=outputs)


@register_op("Unsqueeze")
def _unsqueeze(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    axes = get_axes(node)
    if not axes and len(node.input) > 1 and node.input[1] in parser.parameters:
        t = parser.parameters[node.input[1]]
        if numpy_helper and hasattr(t, "dims"):
            axes = numpy_helper.to_array(t).tolist()
    op = make_op("unsqueeze", axes=axes or [0])
    inputs, outputs = single_input_output(parser, node)
    ir.add_layer(get_layer_name(parser, node_name), type="op", op=op, inputs=inputs, outputs=outputs)


@register_op("Gather")
def _gather(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    axis = get_axis(node)
    idx = node.input[1]
    if idx in parser.constant:
        indices = _to_list(parser.constant[idx])
    elif idx in parser.parameters:
        t = parser.parameters[idx]
        if numpy_helper and hasattr(t, "dims"):
            indices = numpy_helper.to_array(t).tolist()
        else:
            indices = _to_list(t)
    else:
        pred = parser.predecessors.get(idx, [None])[0]
        if pred and hasattr(pred, "name") and pred.name in parser.constant:
            indices = _to_list(parser.constant[pred.name])
        else:
            raise ValueError(f"Gather {node_name}: indices not found")
    op = make_op("gather", axis=axis, indices=indices)
    inputs, outputs = single_input_output(parser, node)
    ir.add_layer(get_layer_name(parser, node_name), type="op", op=op, inputs=inputs, outputs=outputs)


@register_op("Slice")
def _slice(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    starts = _to_list(parser.parameters.get(node.input[1], []))
    ends = _to_list(parser.parameters.get(node.input[2], []))
    axes = _to_list(parser.parameters.get(node.input[3], []))
    steps = _to_list(parser.parameters.get(node.input[4], [1] * len(axes))) if len(node.input) > 4 else [1] * len(axes)
    if numpy_helper:
        for i, name in enumerate([node.input[1], node.input[2], node.input[3], node.input[4] if len(node.input) > 4 else None]):
            if name and name in parser.parameters:
                t = parser.parameters[name]
                if hasattr(t, "dims"):
                    arr = numpy_helper.to_array(t)
                    if i == 0:
                        starts = arr.tolist()
                    elif i == 1:
                        ends = arr.tolist()
                    elif i == 2:
                        axes = arr.tolist()
                    else:
                        steps = arr.tolist()
    op = make_op("slice", starts=starts, ends=ends, axes=axes, steps=steps)
    inputs, outputs = single_input_output(parser, node)
    ir.add_layer(get_layer_name(parser, node_name), type="op", op=op, inputs=inputs, outputs=outputs)


# ---------- Silu ----------
@register_op("Silu")
def _silu(ir: Any, parser: Any, node_name: str) -> None:
    _unary(ir, parser, node_name, "silu")
