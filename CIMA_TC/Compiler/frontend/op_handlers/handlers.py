"""ONNX op handlers for building IR layers."""

from __future__ import annotations

from typing import Any, List, Optional

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised only without numpy installed
    np = None

from CIMA_TC.Compiler.IR_tool.core import make_op

# Import ops so all BaseOp subclasses are registered before handlers instantiate them.
import CIMA_TC.Compiler.IR_tool.ops  # noqa: F401

from ..utils import get_weight_info
from ..utils.attr_reader import (
    get_alpha,
    get_axes,
    get_axis,
    get_conv_node_attr,
    get_conv_node_dilation,
    get_keepdims,
    get_node_epsilon,
    get_node_pads,
    get_pad_mode,
    get_pad_value,
    get_perm,
    get_resize_mode,
    get_split,
)
from .common import (
    get_layer_name,
    input_info_for_tensor,
    inputs_with_constants,
    ir_inputs_for_node,
    node_input_shape,
    node_output_shape,
    output_info_for_node,
    resolve_ref,
    single_input_output,
    static_value,
    tensor_shape,
    to_list,
    to_numpy_value,
)
from .registry import register_op


def _stored_bias_shape(parser: Any, node_name: str) -> Optional[Any]:
    """Return a non-zero stored bias shape for a node, if present."""
    key = f"{node_name}.bias"
    if key not in parser.weight_numpy:
        return None
    bias = parser.weight_numpy[key]
    if np is not None and np.mean(bias) == 0 and np.std(bias) == 0:
        return None
    return bias.shape


def _first(values: Any, default: Any) -> Any:
    return values[0] if values else default


def _numeric_mean_is_zero(values: Any) -> bool:
    if np is not None:
        return bool(np.mean(values) == 0)
    plain = to_list(values)
    if not isinstance(plain, list):
        plain = [plain]
    return all(float(x) == 0 for x in plain)


def _to_int_list(value: Any) -> List[int]:
    arr = to_numpy_value(value)
    if np is not None:
        return np.asarray(arr).astype(np.int32).tolist()
    plain = to_list(arr)
    if isinstance(plain, list):
        return [int(x) for x in plain]
    return [int(plain)]


def _add_op_layer(
    ir: Any,
    parser: Any,
    node_name: str,
    *,
    op: Any,
    inputs: Optional[List[dict]] = None,
    outputs: Optional[List[dict]] = None,
    weights: Optional[dict] = None,
) -> None:
    kwargs: dict[str, Any] = {
        "type": "op",
        "op": op,
        "inputs": inputs,
        "outputs": outputs,
    }
    if weights is not None:
        kwargs["weights"] = weights
    ir.add_layer(get_layer_name(parser, node_name), **kwargs)


@register_op("Conv")
def _conv(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    weight_shape = tensor_shape(parser, node.input[1])
    stride, pad, kernel = get_conv_node_attr(node.attribute)
    dilation = get_conv_node_dilation(node.attribute)
    dilation = _first(dilation, dilation) if isinstance(dilation, (list, tuple)) else dilation
    bias_shape = _stored_bias_shape(parser, node_name) if len(node.input) == 3 else None

    op = make_op(
        "conv2d",
        in_channel=weight_shape[1],
        out_channel=weight_shape[0],
        kernel=_first(kernel, 3),
        stride=_first(stride, 1),
        padding=_first(pad, 0),
        bias=bias_shape is not None,
        dilation=dilation,
    )
    _add_op_layer(
        ir,
        parser,
        node_name,
        op=op,
        inputs=[input_info_for_tensor(parser, node, node.input[0])],
        outputs=output_info_for_node(parser, node),
        weights=get_weight_info(weight_shape, bias_shape),
    )


@register_op("ConvTranspose")
def _conv_transpose(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    weight_shape = tensor_shape(parser, node.input[1])
    stride, pad, kernel = get_conv_node_attr(node.attribute)
    bias_shape = _stored_bias_shape(parser, node_name) if len(node.input) == 3 else None

    op = make_op(
        "conv_transpose2d",
        in_channel=weight_shape[0],
        out_channel=weight_shape[1],
        kernel=_first(kernel, 3),
        stride=_first(stride, 1),
        padding=_first(pad, 0),
        bias=bias_shape is not None,
    )
    _add_op_layer(
        ir,
        parser,
        node_name,
        op=op,
        inputs=[input_info_for_tensor(parser, node, node.input[0])],
        outputs=output_info_for_node(parser, node),
        weights=get_weight_info(weight_shape, bias_shape),
    )


@register_op("MatMul")
def _matmul(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    out_shape = output_info_for_node(parser, node)

    if node.input[1] in parser.parameters:
        weight_shape = tensor_shape(parser, node.input[1])
        op = make_op("linear", in_channel=weight_shape[1], out_channel=weight_shape[0], bias=False)
        _add_op_layer(
            ir,
            parser,
            node_name,
            op=op,
            inputs=[input_info_for_tensor(parser, node, node.input[0])],
            outputs=out_shape,
            weights=get_weight_info(weight_shape),
        )
        return

    _add_op_layer(
        ir,
        parser,
        node_name,
        op=make_op("matmul"),
        inputs=[
            input_info_for_tensor(parser, node, node.input[0]),
            input_info_for_tensor(parser, node, node.input[1]),
        ],
        outputs=out_shape,
    )


@register_op("Gemm")
def _gemm(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    weight_shape = tensor_shape(parser, node.input[1])
    bias_shape = _stored_bias_shape(parser, node_name) if len(node.input) == 3 else None
    op = make_op("linear", in_channel=weight_shape[1], out_channel=weight_shape[0], bias=bias_shape is not None)
    _add_op_layer(
        ir,
        parser,
        node_name,
        op=op,
        inputs=[input_info_for_tensor(parser, node, node.input[0])],
        outputs=output_info_for_node(parser, node),
        weights=get_weight_info(weight_shape, bias_shape),
    )


@register_op("Add")
def _add(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    _add_op_layer(
        ir,
        parser,
        node_name,
        op=make_op("add"),
        inputs=inputs_with_constants(ir, parser, node_name),
        outputs=output_info_for_node(parser, node),
    )


@register_op("Constant")
def _constant(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    found, value = static_value(parser, node.output[0])
    if not found:
        for attr in node.attribute:
            if attr.name == "value" and attr.t:
                value = to_numpy_value(attr.t)
                found = True
                break
    if not found:
        raise ValueError(f"Constant node {node_name} has no value")

    value = to_list(value)
    if value == []:
        value = 0
    out_shape = node_output_shape(parser, node, default=[1]) or [1]
    if out_shape == [0]:
        out_shape = [1]
    _add_op_layer(ir, parser, node_name, op=make_op("constant", value=value), outputs=[output_info_for_node(parser, node)[0] if out_shape != [1] else {"channel": 1, "height": 1, "width": 1, "channel_last": True}])


def _unary(ir: Any, parser: Any, node_name: str, op_id: str, **op_kwargs: Any) -> None:
    node = parser.nodes[node_name]
    inputs, outputs = single_input_output(parser, node)
    _add_op_layer(ir, parser, node_name, op=make_op(op_id, **op_kwargs), inputs=inputs, outputs=outputs)


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
    _unary(ir, parser, node_name, "leaky_relu", alpha=get_alpha(parser.nodes[node_name]))


@register_op("Softmax")
def _softmax(ir: Any, parser: Any, node_name: str) -> None:
    _unary(ir, parser, node_name, "softmax", axis=get_axis(parser.nodes[node_name]))


@register_op("LogSoftmax")
def _log_softmax(ir: Any, parser: Any, node_name: str) -> None:
    _unary(ir, parser, node_name, "log_softmax")


@register_op("Erf")
def _erf(ir: Any, parser: Any, node_name: str) -> None:
    _unary(ir, parser, node_name, "erf")


@register_op("Mul")
def _mul(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    _add_op_layer(
        ir,
        parser,
        node_name,
        op=make_op("mul"),
        inputs=inputs_with_constants(ir, parser, node_name),
        outputs=output_info_for_node(parser, node),
    )


@register_op("Div")
def _div(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    _add_op_layer(
        ir,
        parser,
        node_name,
        op=make_op("div"),
        inputs=inputs_with_constants(ir, parser, node_name),
        outputs=output_info_for_node(parser, node),
    )


@register_op("Transpose")
def _transpose(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    _add_op_layer(
        ir,
        parser,
        node_name,
        op=make_op("transpose", perm=get_perm(node)),
        inputs=ir_inputs_for_node(parser, node, list(node.input)),
        outputs=output_info_for_node(parser, node),
    )


@register_op("Reshape")
def _reshape(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    shape = node_output_shape(parser, node)
    if shape:
        shape = [-1] + list(shape[1:])
    inputs = ir_inputs_for_node(
        parser,
        node,
        [inp for inp in node.input if inp not in parser.parameters and inp in parser.value_infos],
    )
    if not inputs:
        inputs = [input_info_for_tensor(parser, node, node.input[0])]
    _add_op_layer(
        ir,
        parser,
        node_name,
        op=make_op("reshape", shape=shape),
        inputs=inputs,
        outputs=output_info_for_node(parser, node),
    )


@register_op("Flatten")
def _flatten(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    inputs, outputs = single_input_output(parser, node)
    _add_op_layer(ir, parser, node_name, op=make_op("flatten", start_dim=get_axis(node)), inputs=inputs, outputs=outputs)


@register_op("Concat")
def _concat(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    _add_op_layer(
        ir,
        parser,
        node_name,
        op=make_op("concat", axis=get_axis(node)),
        inputs=inputs_with_constants(ir, parser, node_name),
        outputs=output_info_for_node(parser, node),
    )


@register_op("MaxPool")
def _max_pool(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    stride, pad, kernel = get_conv_node_attr(node.attribute)
    inputs, outputs = single_input_output(parser, node)
    _add_op_layer(
        ir,
        parser,
        node_name,
        op=make_op("max_pool2d", kernel=_first(kernel, 2), stride=_first(stride, 2), padding=_first(pad, 0)),
        inputs=inputs,
        outputs=outputs,
    )


@register_op("AveragePool")
def _avg_pool(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    stride, pad, kernel = get_conv_node_attr(node.attribute)
    inputs, outputs = single_input_output(parser, node)
    _add_op_layer(
        ir,
        parser,
        node_name,
        op=make_op("avg_pool2d", kernel=_first(kernel, 2), stride=_first(stride, 2), padding=_first(pad, 0)),
        inputs=inputs,
        outputs=outputs,
    )


@register_op("GlobalAveragePool")
def _global_avg_pool(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    in_shape = node_input_shape(parser, node)
    inputs, outputs = single_input_output(parser, node)
    _add_op_layer(
        ir,
        parser,
        node_name,
        op=make_op("global_avg_pool2d", kernel=in_shape[2] if len(in_shape) > 2 else 1, stride=1, padding=0),
        inputs=inputs,
        outputs=outputs,
    )


@register_op("Split")
def _split(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    axis = get_axis(node)
    in_shape = node_input_shape(parser, node)
    if axis == -1:
        axis = len(in_shape) - 1

    split = get_split(node)
    if split is None and len(node.input) > 1:
        found, value = static_value(parser, node.input[1])
        if found:
            split = to_list(to_numpy_value(value))
    if split is None:
        raise ValueError(f"Split node {node_name}: split attribute or input shape required")

    outputs = [output_info_for_node(parser, node, i)[0] for i, _ in enumerate(node.output)]
    _add_op_layer(
        ir,
        parser,
        node_name,
        op=make_op("split", axis=axis, split=split),
        inputs=[input_info_for_tensor(parser, node, node.input[0])],
        outputs=outputs,
    )


@register_op("ReduceMean")
def _reduce_mean(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    axes = get_axes(node)
    _add_op_layer(
        ir,
        parser,
        node_name,
        op=make_op("reduce_mean", axes=[axes[0] if axes else 0], keepdims=bool(get_keepdims(node))),
        inputs=ir_inputs_for_node(parser, node, list(node.input)),
        outputs=output_info_for_node(parser, node),
    )


@register_op("Sqrt")
def _sqrt(ir: Any, parser: Any, node_name: str) -> None:
    _unary(ir, parser, node_name, "sqrt")


@register_op("Pow")
def _pow(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    _add_op_layer(
        ir,
        parser,
        node_name,
        op=make_op("pow"),
        inputs=inputs_with_constants(ir, parser, node_name, constant_suffix="exp"),
        outputs=output_info_for_node(parser, node),
    )


@register_op("BatchNormalization")
def _batch_norm(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    in_shape = node_input_shape(parser, node)
    channel = in_shape[1]
    inputs, outputs = single_input_output(parser, node)
    _add_op_layer(
        ir,
        parser,
        node_name,
        op=make_op("batch_norm2d", channel=channel, epsilon=float(get_node_epsilon(node))),
        inputs=inputs,
        outputs=outputs,
        weights={
            "weight": {"shape": [channel]},
            "bias": {"shape": [channel]},
            "running_mean": {"shape": [channel]},
            "running_var": {"shape": [channel]},
        },
    )


@register_op("LayerNormalization")
def _layer_norm(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    axis = get_axis(node)
    in_shape = node_input_shape(parser, node)
    num_features = in_shape[axis] if -len(in_shape) <= axis < len(in_shape) else in_shape[-1]
    inputs, outputs = single_input_output(parser, node)
    _add_op_layer(
        ir,
        parser,
        node_name,
        op=make_op("layer_norm", axis=axis, epsilon=float(get_node_epsilon(node))),
        inputs=inputs,
        outputs=outputs,
        weights={"weight": {"shape": [num_features]}, "bias": {"shape": [num_features]}},
    )


@register_op("Pad")
def _pad(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    found, value = static_value(parser, node.input[1]) if len(node.input) > 1 else (False, None)
    pads = to_list(to_numpy_value(value)) if found else get_node_pads(node)
    mode = get_pad_mode(node)
    if "constant" not in mode.lower():
        raise ValueError(f"Pad mode {mode!r} not supported")
    inputs, outputs = single_input_output(parser, node)
    _add_op_layer(
        ir,
        parser,
        node_name,
        op=make_op("pad", pads=pads, value=get_pad_value(node)),
        inputs=inputs,
        outputs=outputs,
    )


@register_op("Resize")
def _resize(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    scale: List[float] = []
    for idx in (2, 1):
        if len(node.input) > idx and node.input[idx]:
            found, value = static_value(parser, node.input[idx])
            if found:
                scale = _to_int_list(value)
                break

    in_shape = node_input_shape(parser, node)
    out_shape = node_output_shape(parser, node)
    if not scale or _numeric_mean_is_zero(scale):
        scale = [out_shape[i] // in_shape[i] if in_shape[i] else 1 for i in range(len(in_shape))]

    inputs, outputs = single_input_output(parser, node)
    _add_op_layer(
        ir,
        parser,
        node_name,
        op=make_op("resize", scale=scale, mode=get_resize_mode(node)),
        inputs=inputs,
        outputs=outputs,
    )


@register_op("Squeeze")
def _squeeze(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    axes = _axes_from_attr_or_input(parser, node)
    inputs, outputs = single_input_output(parser, node)
    _add_op_layer(ir, parser, node_name, op=make_op("squeeze", axes=axes or [0]), inputs=inputs, outputs=outputs)


@register_op("Unsqueeze")
def _unsqueeze(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    axes = _axes_from_attr_or_input(parser, node)
    inputs, outputs = single_input_output(parser, node)
    _add_op_layer(ir, parser, node_name, op=make_op("unsqueeze", axes=axes or [0]), inputs=inputs, outputs=outputs)


def _axes_from_attr_or_input(parser: Any, node: Any) -> List[int]:
    axes = get_axes(node)
    if not axes and len(node.input) > 1:
        found, value = static_value(parser, node.input[1])
        if found:
            axes = to_list(to_numpy_value(value))
    return axes


@register_op("Gather")
def _gather(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    found, value = static_value(parser, node.input[1])
    if found:
        indices = to_list(to_numpy_value(value))
    else:
        pred = parser.predecessors.get(node.input[1], [None])[0]
        if pred and hasattr(pred, "name") and pred.name in parser.constant:
            indices = to_list(parser.constant[pred.name])
        else:
            raise ValueError(f"Gather {node_name}: indices not found")
    inputs, outputs = single_input_output(parser, node)
    _add_op_layer(ir, parser, node_name, op=make_op("gather", axis=get_axis(node), indices=indices), inputs=inputs, outputs=outputs)


@register_op("Slice")
def _slice(ir: Any, parser: Any, node_name: str) -> None:
    node = parser.nodes[node_name]
    axes = _static_list_arg(parser, node, 3, [])
    op = make_op(
        "slice",
        starts=_static_list_arg(parser, node, 1, []),
        ends=_static_list_arg(parser, node, 2, []),
        axes=axes,
        steps=_static_list_arg(parser, node, 4, [1] * len(axes)) if len(node.input) > 4 else [1] * len(axes),
    )
    inputs, outputs = single_input_output(parser, node)
    _add_op_layer(ir, parser, node_name, op=op, inputs=inputs, outputs=outputs)


def _static_list_arg(parser: Any, node: Any, index: int, default: Any) -> Any:
    if len(node.input) <= index:
        return default
    found, value = static_value(parser, node.input[index])
    return to_list(to_numpy_value(value)) if found else default


@register_op("Silu")
def _silu(ir: Any, parser: Any, node_name: str) -> None:
    _unary(ir, parser, node_name, "silu")
