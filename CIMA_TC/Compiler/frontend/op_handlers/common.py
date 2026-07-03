"""Shared helpers for ONNX op handlers."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised only without numpy installed
    np = None

try:
    from onnx import numpy_helper
except ImportError:  # pragma: no cover - exercised only without onnx installed
    numpy_helper = None

from CIMA_TC.Compiler.IR_tool.core import make_op

from ..utils.shape_utils import dim_to_list, get_input_info, get_output_info


ValueResult = Tuple[bool, Any]


def get_layer_name(parser: Any, node_name: str) -> str:
    """Return the IR layer name, using parser.name_map when available."""
    name_map = getattr(parser, "name_map", None)
    return name_map.get(node_name, node_name) if name_map else node_name


def resolve_ref(parser: Any, node: Any, input_name: str) -> str:
    """
    Resolve an ONNX tensor input to an IR ref.

    Graph inputs become `graph_input:index`; Split outputs become
    `producer:index`; all other produced tensors resolve to the producer layer.
    """
    if input_name in parser.graph_input:
        idx = parser.inputs.index(input_name) if input_name in parser.inputs else parser.graph_input.index(input_name)
        return f"graph_input:{idx}"

    result = input_name
    pred_list = parser.predecessors.get(input_name, [])
    if pred_list:
        pred = pred_list[0]
        if hasattr(pred, "name"):
            pred_name = pred.name
            result = _split_ref(parser, pred_name, input_name) or pred_name

    return _map_ref_name(parser, result)


def _split_ref(parser: Any, pred_name: str, input_name: str) -> Optional[str]:
    pred_node = parser.nodes.get(pred_name)
    if pred_node is None or pred_node.op_type != "Split":
        return None
    outputs = list(pred_node.output)
    if input_name not in outputs:
        return pred_name
    return f"{pred_name}:{outputs.index(input_name)}"


def _map_ref_name(parser: Any, ref: str) -> str:
    name_map = getattr(parser, "name_map", None)
    if not name_map or not ref or ref.startswith("graph_input:"):
        return ref
    if ":" not in ref:
        return name_map.get(ref, ref)
    name, _, index = ref.partition(":")
    return f"{name_map.get(name, name)}:{index}"


def tensor_shape(parser: Any, tensor_name: str, *, default: Optional[List[int]] = None) -> List[int]:
    """Return an ONNX tensor shape from parser.value_infos."""
    vi = parser.value_infos.get(tensor_name)
    if vi is None:
        if default is not None:
            return list(default)
        raise ValueError(f"Missing value_info for tensor {tensor_name!r}")
    return dim_to_list(vi.type.tensor_type.shape.dim)


def node_input_shape(parser: Any, node: Any, index: int = 0, *, default: Optional[List[int]] = None) -> List[int]:
    return tensor_shape(parser, node.input[index], default=default)


def node_output_shape(parser: Any, node: Any, index: int = 0, *, default: Optional[List[int]] = None) -> List[int]:
    return tensor_shape(parser, node.output[index], default=default)


def input_info_for_tensor(parser: Any, node: Any, tensor_name: str) -> dict:
    """Build an IR input spec for one non-static ONNX tensor input."""
    shape = tensor_shape(parser, tensor_name, default=[1]) or [1]
    return get_input_info(shape, resolve_ref(parser, node, tensor_name))


def ir_inputs_for_node(parser: Any, node: Any, input_names: List[str]) -> List[dict]:
    """Build IR input specs for non-static inputs, skipping missing tensors."""
    inputs = []
    for inp in input_names:
        if is_static_input(parser, inp) or inp not in parser.value_infos:
            continue
        inputs.append(input_info_for_tensor(parser, node, inp))
    return inputs


def single_input_output(parser: Any, node: Any) -> tuple:
    """Return ([input spec], [output spec]) for a single-input single-output op."""
    return [input_info_for_tensor(parser, node, node.input[0])], get_output_info(node_output_shape(parser, node))


def is_static_input(parser: Any, tensor_name: str) -> bool:
    """True when a tensor name resolves to an initializer or Constant value."""
    return tensor_name in parser.parameters or tensor_name in parser.constant


def static_value(parser: Any, tensor_name: str) -> ValueResult:
    """Return (found, value) for static initializer/constant tensors."""
    if tensor_name in parser.parameters:
        return True, parser.parameters[tensor_name]
    if tensor_name in parser.constant:
        return True, parser.constant[tensor_name]
    return False, None


def to_numpy_value(value: Any) -> Any:
    """Convert ONNX TensorProto-like values to numpy arrays when possible."""
    if numpy_helper is not None and hasattr(value, "dims"):
        return numpy_helper.to_array(value)
    return value if hasattr(value, "shape") else np.array(value)


def to_list(value: Any) -> Any:
    """Convert numpy-like values to plain Python containers for ConstantOp."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return value


def constant_input_info(
    ir: Any,
    parser: Any,
    node_name: str,
    value: Any,
    *,
    suffix: str | int,
) -> dict:
    """Insert a Constant layer and return an input spec pointing at it."""
    arr = to_numpy_value(value)
    shape = list(arr.shape) if getattr(arr, "shape", ()) else [1]
    const_name = f"{get_layer_name(parser, node_name)}_const_{suffix}"
    ir.add_layer(
        const_name,
        type="op",
        op=make_op("constant", value=to_list(arr)),
        outputs=get_output_info(shape),
    )
    return get_input_info(shape, const_name)


def inputs_with_constants(ir: Any, parser: Any, node_name: str, *, constant_suffix: str | None = None) -> List[dict]:
    """Build input specs for all node inputs, injecting constants where needed."""
    node = parser.nodes[node_name]
    inputs: List[dict] = []
    const_index = 0
    for inp in node.input:
        found, value = static_value(parser, inp)
        if found:
            suffix = constant_suffix if constant_suffix is not None else const_index
            inputs.append(constant_input_info(ir, parser, node_name, value, suffix=suffix))
            const_index += 1
            continue
        inputs.append(input_info_for_tensor(parser, node, inp))
    return inputs


def output_info_for_node(parser: Any, node: Any, index: int = 0) -> List[dict]:
    return get_output_info(node_output_shape(parser, node, index))
