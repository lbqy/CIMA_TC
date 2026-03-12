"""
ONNX preprocessing: shape inference, optional constant folding, and meaningless-op removal.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from .utils.onnx_io import load_onnx, save_onnx, add_value_info_for_constants
from .utils.shape_utils import dim_to_list

try:
    import onnx
    from onnx import shape_inference
    from onnx import helper
    from onnx import numpy_helper
    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False


def value_info_has_shape(vi: Any) -> bool:
    """True if value_info has at least one dimension."""
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
    Preprocess ONNX model: shape inference, add value_info for constants,
    optional custom shape pass, and MeaninglessOpPass (remove zero Pad/Add).
    Returns (preprocessed_model, updated_name_dict or None).
    """
    if not _ONNX_AVAILABLE:
        raise RuntimeError("onnx package is required for preprocessing")

    model = shape_inference.infer_shapes(model)
    model = add_value_info_for_constants(model)

    # Optional: run custom shape inference if needed (e.g. after constant folding)
    # model = custom_shape_inference(model)

    constant_dict = _get_model_constant_nodes(model)
    initializer_dict = _get_initializer_dict(model)
    model = _meaningless_op_pass(model, constant_dict, initializer_dict)

    updated_name_dict: Optional[Dict[str, str]] = None
    if fix_layer_name:
        model, updated_name_dict = _fix_node_names(model)

    if store_intermediate_model:
        save_onnx(model, os.path.join(os.getcwd(), "intermediate_convert_model.onnx"))

    return model, updated_name_dict


def _get_model_constant_nodes(model: Any) -> Dict[str, Any]:
    out = {}
    for node in model.graph.node:
        if node.op_type == "Constant":
            out[node.output[0]] = node
    return out


def _get_initializer_dict(model: Any) -> Dict[str, Any]:
    return {init.name: init for init in model.graph.initializer if init.name not in [i.name for i in model.graph.input]}


def _meaningless_op_pass(model: Any, constant_dict: Dict, initializer_dict: Dict) -> Any:
    """Remove Pad with zero pads and Add with zero addend where possible."""
    import numpy as np
    nodes = list[Any](model.graph.node)
    input_node = {n.input[0]: n for n in nodes if n.op_type != "Constant"}
    output_node = {n.output[0]: n for n in nodes if n.op_type != "Constant"}
    valueinfo = {v.name: v for v in model.graph.value_info}
    for inp in model.graph.input:
        valueinfo[inp.name] = inp
    for out in model.graph.output:
        valueinfo[out.name] = out

    to_remove = set[Any]()
    replacements = {}

    for i, node in enumerate[Any](nodes):
        if node.op_type == "Pad":
            pads = _get_node_pads(node, constant_dict)
            if pads is not None and len(pads) and np.mean(pads) == 0 and np.std(pads) == 0:
                out_name = node.output[0]
                in_name = node.input[0]
                if out_name in output_node:
                    succ = output_node[out_name]
                    repl = list[Any](succ.input)
                    for j, x in enumerate[Any](repl):
                        if x == out_name:
                            repl[j] = in_name
                    replacements[succ.name] = (repl, succ.output, succ)
                to_remove.add(node.name)
                if len(node.input) > 1 and node.input[1] in constant_dict:
                    to_remove.add(constant_dict[node.input[1]].name)

        elif node.op_type == "Add":
            add_val = None
            if len(node.input) >= 2:
                if node.input[1] in constant_dict:
                    c = constant_dict[node.input[1]]
                    for attr in c.attribute:
                        if attr.t:
                            add_val = numpy_helper.to_array(attr.t)
                            break
                elif node.input[1] in initializer_dict:
                    add_val = numpy_helper.to_array(initializer_dict[node.input[1]])
            if add_val is not None and np.mean(add_val) == 0 and np.std(add_val) == 0:
                out_name = node.output[0]
                in_name = node.input[0]
                if out_name in output_node:
                    succ = output_node[out_name]
                    repl = [in_name] + list(succ.input[1:])
                    replacements[succ.name] = (repl, succ.output, succ)
                to_remove.add(node.name)
                if len(node.input) >= 2 and node.input[1] in constant_dict:
                    to_remove.add(constant_dict[node.input[1]].name)

    if not to_remove and not replacements:
        return model

    new_nodes = [n for n in nodes if n.name not in to_remove]
    for name, (new_inputs, outputs, succ) in replacements.items():
        attr = _get_node_attrs(succ)
        new_node = helper.make_node(succ.op_type, new_inputs, list[str](outputs), name=name, **attr)
        for idx, n in enumerate[Any](new_nodes):
            if n.name == name:
                new_nodes[idx] = new_node
                break
    graph = helper.make_graph(
        new_nodes,
        model.graph.name,
        model.graph.input,
        model.graph.output,
        model.graph.initializer,
        value_info=model.graph.value_info,
    )
    return helper.make_model(graph)


def _get_node_pads(node: Any, constant_dict: Dict) -> Optional[Any]:
    if len(node.input) > 1 and node.input[1] in constant_dict:
        c = constant_dict[node.input[1]]
        for attr in c.attribute:
            if attr.t:
                return numpy_helper.to_array(attr.t).tolist()
    for a in node.attribute:
        if a.name == "pads":
            return list[Any](a.ints)
    return None


def _get_node_attrs(node: Any) -> Dict[str, Any]:
    attr = {}
    for a in node.attribute:
        if a.name in ("group", "ceil_mode", "axis"):
            attr[a.name] = a.i
        elif a.type == 7:
            attr[a.name] = a.ints
        elif a.type == 4:
            attr[a.name] = a.t
        elif a.type == 1:
            attr[a.name] = a.f
        elif a.type == 3:
            attr[a.name] = a.s
    return attr


def _fix_node_names(model: Any) -> Tuple[Any, Dict[str, str]]:
    """Assign op_type_0, op_type_1, ... to nodes; return (model, old_name -> new_name)."""
    from onnx import helper
    updated: Dict[str, str] = {}
    new_list: List[Any] = []
    for c, node in enumerate(model.graph.node):
        new_name = f"{node.op_type}_{c}"
        if node.name:
            updated[node.name] = new_name
        attr_dict = {}
        for a in node.attribute:
            if node.op_type == "Concat":
                attr_dict[a.name] = a.i
            elif a.name in ("group", "ceil_mode", "axis"):
                attr_dict[a.name] = a.i
            else:
                if a.type == 7:
                    attr_dict[a.name] = a.ints
                elif a.type == 4:
                    attr_dict[a.name] = a.t
                elif a.type == 1:
                    attr_dict[a.name] = a.f
                elif a.type == 3:
                    attr_dict[a.name] = a.s
        new_node = helper.make_node(node.op_type, list(node.input), list(node.output), new_name, **attr_dict)
        new_list.append(new_node)
    graph = helper.make_graph(
        new_list,
        model.graph.name,
        model.graph.input,
        model.graph.output,
        model.graph.initializer,
        value_info=model.graph.value_info,
    )
    return helper.make_model(graph), updated
