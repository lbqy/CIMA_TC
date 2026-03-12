"""
ONNX graph parser: value_infos, parameters, nodes, predecessors/successors, weights.
Simplified and typed; no quantization or LSTM weight splitting in this module.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

try:
    from onnx import numpy_helper
    _NP_HELPER = True
except ImportError:
    _NP_HELPER = False

from .utils.shape_utils import dim_to_list


class OnnxParser:
    """
    Parses an ONNX model into value_infos, parameters, nodes, graph I/O,
    predecessors/successors, weight_numpy (for Conv/MatMul/Gemm/ConvTranspose/BN/LN), and constant.
    """

    def __init__(
        self,
        onnx_model: Any,
        weight_half_level: Optional[int] = None,
        weight_scale: Optional[Dict[str, Any]] = None,
        data_clamp_std: float = 0.0,
        data_range_specify: Optional[Any] = None,
    ) -> None:
        self.model = onnx_model
        self.weight_half_level = weight_half_level
        self.weight_scale = weight_scale or {}
        self.data_clamp_std = data_clamp_std
        self.data_range_specify = data_range_specify

        self.value_infos: Dict[str, Any] = {}
        self.parameters: Dict[str, Any] = {}
        self.nodes: Dict[str, Any] = {}
        self.inputs: List[str] = []
        self.weight_numpy: Dict[str, Any] = {}
        self.constant: Dict[str, Any] = {}
        self.predecessors: Dict[str, List[Any]] = defaultdict[str, List[Any]](list)
        self.successors: Dict[str, List[Any]] = defaultdict[str, List[Any]](list)
        self.node_weight_name: Dict[str, str] = {}

        self._map_value_infos()
        self._map_nodes()
        self._map_initializer_parameters()
        self._map_orders()
        self._save_constant_parameters()
        self._map_node_weight_name()

        self.inputs = list[str](self.graph_input)

    @property
    def graph(self) -> Any:
        return self.model.graph

    @property
    def graph_input(self) -> List[str]:
        return [i.name for i in self.graph.input]

    @property
    def graph_output(self) -> List[str]:
        return [o.name for o in self.graph.output]

    def _map_value_infos(self) -> None:
        for vi in list(self.graph.value_info) + list(self.graph.input) + list(self.graph.output):
            self.value_infos[vi.name] = vi

    def _map_nodes(self) -> None:
        for node in self.graph.node:
            self.nodes[node.name] = node

    def _map_initializer_parameters(self) -> None:
        for tensor in self.graph.initializer:
            if tensor.name not in self.graph_input:
                self.parameters[tensor.name] = tensor

        for node in self.graph.node:
            if node.op_type == "Constant":
                for attr in node.attribute:
                    if attr.name == "value" and attr.t:
                        if _NP_HELPER:
                            self.parameters[node.output[0]] = numpy_helper.to_array(attr.t)
                        else:
                            self.parameters[node.output[0]] = attr.t

        # Build weight_numpy for layers that have stored weights
        for node_name, node in self.nodes.items():
            if node.op_type in ("Conv", "ConvTranspose") and len(node.input) >= 2:
                wname = node.input[1]
                if wname in self.parameters:
                    arr = self._to_numpy(self.parameters[wname])
                    if arr is not None:
                        self.weight_numpy[f"{node_name}.weight"] = arr
                if len(node.input) == 3 and node.input[2] in self.parameters:
                    arr = self._to_numpy(self.parameters[node.input[2]])
                    if arr is not None:
                        self.weight_numpy[f"{node_name}.bias"] = arr

            elif node.op_type == "MatMul" and len(node.input) >= 2:
                wname = node.input[1]
                if wname in self.parameters:
                    arr = self._to_numpy(self.parameters[wname])
                    if arr is not None:
                        # FC: weight is (out, in); store as (out, in)
                        self.weight_numpy[f"{node_name}.weight"] = arr.T

            elif node.op_type == "Gemm" and len(node.input) >= 2:
                wname = node.input[1]
                if wname in self.parameters:
                    arr = self._to_numpy(self.parameters[wname])
                    if arr is not None:
                        self.weight_numpy[f"{node_name}.weight"] = arr.T if arr.shape[0] != arr.shape[1] else arr
                if len(node.input) == 3 and node.input[2] in self.parameters:
                    arr = self._to_numpy(self.parameters[node.input[2]])
                    if arr is not None:
                        self.weight_numpy[f"{node_name}.bias"] = arr

            elif node.op_type == "BatchNormalization" and len(node.input) >= 5:
                for i, key in enumerate(["weight", "bias", "running_mean", "running_var"]):
                    if node.input[i + 1] in self.parameters:
                        arr = self._to_numpy(self.parameters[node.input[i + 1]])
                        if arr is not None:
                            self.weight_numpy[f"{node_name}.{key}"] = arr

            elif node.op_type == "LayerNormalization" and len(node.input) >= 3:
                if node.input[1] in self.parameters:
                    arr = self._to_numpy(self.parameters[node.input[1]])
                    if arr is not None:
                        self.weight_numpy[f"{node_name}.weight"] = arr
                if node.input[2] in self.parameters:
                    arr = self._to_numpy(self.parameters[node.input[2]])
                    if arr is not None:
                        self.weight_numpy[f"{node_name}.bias"] = arr

    def _to_numpy(self, val: Any) -> Optional[Any]:
        if hasattr(val, "shape") and hasattr(val, "dtype"):
            return val
        if _NP_HELPER and hasattr(val, "dims"):
            return numpy_helper.to_array(val)
        return None

    def _map_orders(self) -> None:
        for node in self.graph.node:
            for i in node.input:
                if i in self.graph_input:
                    self.predecessors[i].append(self.value_infos.get(i))
                self.successors[i].append(node)
            for o in node.output:
                if o in self.graph_output:
                    self.successors[o].append(self.value_infos.get(o))
                self.predecessors[o].append(node)

    def _save_constant_parameters(self) -> None:
        for node in self.graph.node:
            if node.op_type == "Constant":
                for attr in node.attribute:
                    if attr.name == "value" and attr.t:
                        if _NP_HELPER:
                            self.constant[node.output[0]] = numpy_helper.to_array(attr.t)
                        else:
                            self.constant[node.output[0]] = attr.t
                        break

    def _map_node_weight_name(self) -> None:
        for node in self.graph.node:
            if node.op_type in ("Conv", "MatMul") and len(node.input) >= 2:
                self.node_weight_name[node.name] = node.input[1]
