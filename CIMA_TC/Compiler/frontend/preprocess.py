"""ONNX preprocessing: shape inference and small graph cleanups."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from .utils.onnx_io import add_value_info_for_constants, save_onnx

try:
    import onnx
    from onnx import helper, numpy_helper, shape_inference

    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False


def value_info_has_shape(vi: Any) -> bool:
    """Return True when value_info carries at least one shape dimension."""
    return bool(list(vi.type.tensor_type.shape.dim))


def load_onnx_model(
    model: Any,
    *,
    fix_layer_name: bool = False,
    store_intermediate_model: bool = False,
    label_file: Optional[str] = None,
    output_npu_model_filename: Optional[str] = None,
) -> Tuple[Any, Optional[Dict[str, str]]]:
    """
    Run ONNX shape inference, add constant value_info, remove simple no-op nodes,
    and optionally normalize node names.
    """
    if not _ONNX_AVAILABLE:
        raise RuntimeError("onnx package is required for preprocessing")

    model = shape_inference.infer_shapes(model)
    model = add_value_info_for_constants(model)
    model = _meaningless_op_pass(model, _get_model_constant_nodes(model), _get_initializer_dict(model))

    updated_name_dict: Optional[Dict[str, str]] = None
    if fix_layer_name:
        model, updated_name_dict = _fix_node_names(model)

    if store_intermediate_model:
        save_onnx(model, os.path.join(os.getcwd(), "intermediate_convert_model.onnx"))

    return model, updated_name_dict


def _get_model_constant_nodes(model: Any) -> Dict[str, Any]:
    return {node.output[0]: node for node in model.graph.node if node.op_type == "Constant"}


def _get_initializer_dict(model: Any) -> Dict[str, Any]:
    graph_inputs = {inp.name for inp in model.graph.input}
    return {init.name: init for init in model.graph.initializer if init.name not in graph_inputs}


def _meaningless_op_pass(model: Any, constant_dict: Dict[str, Any], initializer_dict: Dict[str, Any]) -> Any:
    """Remove zero Pad/Add nodes when their single output feeds a normal node."""
    import numpy as np

    nodes = list(model.graph.node)
    consumer_by_input = _first_consumer_by_input(nodes)
    to_remove: set[str] = set()
    replacements: Dict[str, Tuple[List[str], Any, Any]] = {}

    for node in nodes:
        if node.op_type == "Pad" and _is_zero_value(_get_node_pads(node, constant_dict), np):
            _mark_passthrough(node, node.input[0], consumer_by_input, replacements, to_remove)
            if len(node.input) > 1 and node.input[1] in constant_dict:
                to_remove.add(constant_dict[node.input[1]].name)

        elif node.op_type == "Add":
            add_val = _add_rhs_value(node, constant_dict, initializer_dict)
            if add_val is not None and _is_zero_value(add_val, np):
                _mark_passthrough(node, node.input[0], consumer_by_input, replacements, to_remove)
                if len(node.input) >= 2 and node.input[1] in constant_dict:
                    to_remove.add(constant_dict[node.input[1]].name)

    if not to_remove and not replacements:
        return model

    new_nodes = [node for node in nodes if node.name not in to_remove]
    _apply_replacements(new_nodes, replacements)
    graph = helper.make_graph(
        new_nodes,
        model.graph.name,
        model.graph.input,
        model.graph.output,
        model.graph.initializer,
        value_info=model.graph.value_info,
    )
    return helper.make_model(graph)


def _first_consumer_by_input(nodes: List[Any]) -> Dict[str, Any]:
    consumers: Dict[str, Any] = {}
    for node in nodes:
        if node.op_type == "Constant":
            continue
        for input_name in node.input:
            consumers.setdefault(input_name, node)
    return consumers


def _mark_passthrough(
    node: Any,
    replacement_input: str,
    consumer_by_input: Dict[str, Any],
    replacements: Dict[str, Tuple[List[str], Any, Any]],
    to_remove: set[str],
) -> None:
    out_name = node.output[0]
    consumer = consumer_by_input.get(out_name)
    if consumer is not None:
        new_inputs = [replacement_input if x == out_name else x for x in consumer.input]
        replacements[consumer.name] = (new_inputs, consumer.output, consumer)
    to_remove.add(node.name)


def _apply_replacements(new_nodes: List[Any], replacements: Dict[str, Tuple[List[str], Any, Any]]) -> None:
    for name, (new_inputs, outputs, old_node) in replacements.items():
        new_node = helper.make_node(old_node.op_type, new_inputs, list(outputs), name=name, **_get_node_attrs(old_node))
        for index, node in enumerate(new_nodes):
            if node.name == name:
                new_nodes[index] = new_node
                break


def _is_zero_value(value: Any, np: Any) -> bool:
    if value is None:
        return False
    try:
        return len(value) > 0 and np.mean(value) == 0 and np.std(value) == 0
    except TypeError:
        return np.mean(value) == 0 and np.std(value) == 0


def _add_rhs_value(node: Any, constant_dict: Dict[str, Any], initializer_dict: Dict[str, Any]) -> Optional[Any]:
    if len(node.input) < 2:
        return None
    rhs = node.input[1]
    if rhs in constant_dict:
        return _constant_tensor_value(constant_dict[rhs])
    if rhs in initializer_dict:
        return numpy_helper.to_array(initializer_dict[rhs])
    return None


def _constant_tensor_value(node: Any) -> Optional[Any]:
    for attr in node.attribute:
        if attr.t:
            return numpy_helper.to_array(attr.t)
    return None


def _get_node_pads(node: Any, constant_dict: Dict[str, Any]) -> Optional[Any]:
    if len(node.input) > 1 and node.input[1] in constant_dict:
        return _constant_tensor_value(constant_dict[node.input[1]])
    for attr in node.attribute:
        if attr.name == "pads":
            return list(attr.ints)
    return None


def _get_node_attrs(node: Any) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}
    for attr in node.attribute:
        if attr.name in ("group", "ceil_mode", "axis"):
            attrs[attr.name] = attr.i
        elif attr.type == 7:
            attrs[attr.name] = list(attr.ints)
        elif attr.type == 4:
            attrs[attr.name] = attr.t
        elif attr.type == 1:
            attrs[attr.name] = attr.f
        elif attr.type == 3:
            attrs[attr.name] = attr.s
    return attrs


def _fix_node_names(model: Any) -> Tuple[Any, Dict[str, str]]:
    """Assign op_type_0, op_type_1, ... names to ONNX nodes."""
    updated: Dict[str, str] = {}
    new_nodes: List[Any] = []
    for index, node in enumerate(model.graph.node):
        new_name = f"{node.op_type}_{index}"
        if node.name:
            updated[node.name] = new_name
        new_nodes.append(
            helper.make_node(
                node.op_type,
                list(node.input),
                list(node.output),
                new_name,
                **_get_node_attrs(node),
            )
        )
    graph = helper.make_graph(
        new_nodes,
        model.graph.name,
        model.graph.input,
        model.graph.output,
        model.graph.initializer,
        value_info=model.graph.value_info,
    )
    return helper.make_model(graph), updated
