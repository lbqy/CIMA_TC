"""
Common helpers for op handlers: resolve input ref (graph_input, Split branch, or predecessor).
"""

from __future__ import annotations

from typing import Any, List

from ..utils.shape_utils import dim_to_list, get_input_info, get_output_info


def get_layer_name(parser: Any, node_name: str) -> str:
    """Return IR layer name for this node (sanitized if parser.name_map is set)."""
    name_map = getattr(parser, "name_map", None)
    if name_map:
        return name_map.get(node_name, node_name)
    return node_name


def resolve_ref(parser: Any, node: Any, input_name: str) -> str:
    """
    Resolve an input tensor name to the IR ref string for the producer.
    - If input is graph input -> 'graph_input:{index}'.
    - If producer is Split -> 'producer_name:output_index'.
    - Else -> producer node name.
    If parser.name_map is set, node names are mapped to sanitized IR layer names.
    """
    if input_name in parser.graph_input:
        idx = parser.inputs.index(input_name) if input_name in parser.inputs else parser.graph_input.index(input_name)
        return f"graph_input:{idx}"
    pred_list = parser.predecessors.get(input_name, [])
    if not pred_list:
        result = input_name
    else:
        pred = pred_list[0]
        if hasattr(pred, "name"):
            pred_name = pred.name
            if pred_name in parser.nodes and parser.nodes[pred_name].op_type == "Split":
                out_list = list[Any](parser.nodes[pred_name].output)
                if input_name in out_list:
                    result = f"{pred_name}:{out_list.index(input_name)}"
                else:
                    result = pred_name
            else:
                result = pred_name
        else:
            result = input_name
    name_map = getattr(parser, "name_map", None)
    if name_map and result:
        if ":" in result and not result.startswith("graph_input:"):
            part, _, idx = result.partition(":")
            result = f"{name_map.get(part, part)}:{idx}"
        else:
            result = name_map.get(result, result)
    return result


def ir_inputs_for_node(parser: Any, node: Any, input_names: List[str]) -> List[dict]:
    """Build list of IR input dicts for the given node input names (skip parameters/constants)."""
    value_infos = parser.value_infos
    inputs = []
    for inp in input_names:
        if inp in parser.parameters or inp in parser.constant:
            continue
        if inp not in value_infos:
            continue
        shape = dim_to_list(value_infos[inp].type.tensor_type.shape.dim)
        if not shape:
            shape = [1]
        ref = resolve_ref(parser, node, inp)
        inputs.append(get_input_info(shape, ref))
    return inputs


def single_input_output(parser: Any, node: Any) -> tuple:
    """(inputs list of one dict, outputs list) for single-input single-output op."""
    in_shape = dim_to_list(parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
    out_shape = dim_to_list(parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
    ref = resolve_ref(parser, node, node.input[0])
    inputs = [get_input_info(in_shape, ref)]
    outputs = get_output_info(out_shape)
    return inputs, outputs
