"""
ONNX -> IR converter. Uses config, runs preprocessing, parser, and op handlers; builds flat GraphLayer.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

try:
    import numpy as np
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from CIMA_TC.Compiler.IR_tool.core.ir import BaseIR, save_ir
from CIMA_TC.Compiler.IR_tool.core import DataDef, OpLayer, make_op
from CIMA_TC.Compiler.IR_tool.core.jsonable import SerializationConfig

from .config import ConversionConfig
from .parser import OnnxParser
from .preprocess import load_onnx_model
from .utils.onnx_io import load_onnx
from .utils.ir_rewrite import fuse_sigmoid_mul_to_silu, rename_batchnorm_layers
from .utils.shape_utils import dim_to_list, get_input_info, get_output_info
from .utils.name_sanitize import build_layer_name_map
from .utils.weight_export import export_weights as _export_weights_impl
from .op_handlers import get_handler, UnsupportedONNXOpError


class ConvertONNX:
    """
    Convert ONNX model to IR (GraphLayer + DeviceTree).
    Construction only stores config; call convert() to run conversion.
    """

    def __init__(self, config: Optional[ConversionConfig] = None, **kwargs: Any) -> None:
        if config is not None:
            self.config = config
        else:
            allowed = {"onnx_file", "ir_file", "fix_layer_name", "store_intermediate_model", "specify_input_layer", "specify_output_layer", "weight_half_level", "weight_scale", "data_range_specify", "data_clamp_std"}
            self.config = ConversionConfig(**{k: v for k, v in kwargs.items() if k in allowed})
        self.ir: Optional[BaseIR] = None
        self.parser: Optional[OnnxParser] = None
        self.updated_name_dict: Optional[Dict[str, str]] = None
        self._redundant: Set[str] = set()

    def convert(self, onnx_file: Optional[str] = None, onnx_model: Any = None) -> BaseIR:
        """
        Run conversion. Uses config.onnx_file if onnx_file and onnx_model are None.
        """
        cfg = self.config
        if onnx_model is not None:
            model = onnx_model
        else:
            path = onnx_file or cfg.onnx_file
            if path is None:
                raise ValueError("Provide onnx_file path or onnx_model")
            model = load_onnx(path)

        model, self.updated_name_dict = load_onnx_model(
            model,
            fix_layer_name=cfg.fix_layer_name,
            store_intermediate_model=cfg.store_intermediate_model,
        )

        self.parser = OnnxParser(
            model,
            weight_half_level=cfg.weight_half_level,
            weight_scale=cfg.weight_scale,
            data_clamp_std=cfg.data_clamp_std,
            data_range_specify=cfg.data_range_specify,
        )
        parser = self.parser

        if cfg.specify_input_layer:
            parser.inputs = []
            for sil in cfg.specify_input_layer:
                if sil in parser.nodes:
                    parser.inputs.append(parser.nodes[sil].input[0])
            self._mark_redundant_pre(parser.inputs)
        else:
            parser.inputs = list[str](parser.graph_input)

        if cfg.specify_output_layer:
            self._mark_redundant_post(parser, cfg.specify_output_layer)

        self.ir = BaseIR.make_ir()

        # Map ONNX node names to IR-safe layer names (e.g. /Conv_0/Conv -> Conv_0_Conv)
        parser.name_map = build_layer_name_map(list(parser.nodes.keys()))

        # Graph input layer: one layer with multiple outputs (one per graph input)
        g_inputs = []
        for input_name in parser.inputs:
            vi = parser.value_infos.get(input_name)
            if vi is None:
                raise ValueError(f"Missing value_info for graph input {input_name!r}")
            shape = dim_to_list(vi.type.tensor_type.shape.dim)
            if len(shape) == 4:
                d = dict[str, int](channel=shape[1], height=shape[2], width=shape[3], channel_last=True)
            elif len(shape) == 2:
                d = dict[str, int](channel=shape[1], height=1, width=1, channel_last=True)
            elif len(shape) == 3:
                d = dict[str, int](channel=shape[0], height=shape[1], width=shape[2], channel_last=True)
            else:
                raise ValueError(f"Unsupported input rank: {shape}")
            g_inputs.append(d)
        self.ir.add_layer("graph_input", type="input", outputs=g_inputs)

        # Op layers
        for node_name in parser.nodes:
            if node_name in self._redundant:
                continue
            node = parser.nodes[node_name]
            op_type = node.op_type
            handler = get_handler(op_type)
            if handler is None:
                raise UnsupportedONNXOpError(op_type, node_name)
            handler(self.ir, parser, node_name)

        # Graph output layer
        if cfg.specify_output_layer:
            output_names = []
            for ol in cfg.specify_output_layer:
                if ol in parser.nodes:
                    for o in parser.nodes[ol].output:
                        output_names.append(o)
        else:
            output_names = list[str](parser.graph_output)

        g_outputs = []
        for out_name in output_names:
            vi = parser.value_infos.get(out_name)
            if vi is None:
                continue
            shape = dim_to_list(vi.type.tensor_type.shape.dim)
            pred_list = parser.predecessors.get(out_name, [])
            ref_name = pred_list[0].name if pred_list and hasattr(pred_list[0], "name") else out_name
            ref_name = parser.name_map.get(ref_name, ref_name)
            if len(shape) == 4:
                d = dict[str, str | int](ref=ref_name, channel=shape[1], height=shape[2], width=shape[3], channel_last=True)
            elif len(shape) == 2:
                d = dict[str, str | int](ref=ref_name, channel=shape[1], height=1, width=1, channel_last=True)
            elif len(shape) == 3:
                d = dict[str, str | int](ref=ref_name, channel=shape[0], height=shape[1], width=shape[2], channel_last=True)
            else:
                d = dict[str, str | List[int]](ref=ref_name, shape=shape)
            g_outputs.append(d)
        self.ir.add_layer("graph_output", type="output", inputs=g_outputs)

        # Remove dead op layers (no successor)
        self._remove_redundant_layers()

        # Optional frontend IR rewrite passes (default enabled)
        if getattr(cfg, "enable_ir_rewrite", True):
            # Pattern: Y = Mul(X, Sigmoid(X)) -> Silu(X)
            self._fuse_sigmoid_mul_to_silu()
            # Rename BN to "{nearest_conv_or_fc}_bn" and update parser.name_map
            self._rename_batchnorm_layers()

        # Attach runtime-only weight/BN stores onto IR for later mapping passes.
        # These are not serialized by dump(), so they will not affect readability.
        if self.ir is not None and self.parser is not None:
            name_map = getattr(self.parser, "name_map", {}) or {}
            ws = getattr(self.ir, "weight_store", None)
            bs = getattr(self.ir, "bn_store", None)
            if isinstance(ws, dict) and isinstance(bs, dict):
                for key, arr in self.parser.weight_numpy.items():
                    if "." not in key:
                        continue
                    node_name, suffix = key.rsplit(".", 1)
                    ir_name = name_map.get(node_name, node_name)
                    new_key = f"{ir_name}.{suffix}"
                    if suffix in ("running_mean", "running_var"):
                        bs[new_key] = arr
                    else:
                        ws[new_key] = arr

        return self.ir

    def _fuse_sigmoid_mul_to_silu(self) -> None:
        if self.ir is None:
            return
        fuse_sigmoid_mul_to_silu(self.ir, name_map=getattr(self.parser, "name_map", None))

    def _rename_batchnorm_layers(self) -> None:
        if self.ir is None:
            return
        rename_batchnorm_layers(self.ir, name_map=getattr(self.parser, "name_map", None))

    def _mark_redundant_pre(self, input_tensor_names: List[str]) -> None:
        """Mark nodes that are predecessors of the given input tensors (traverse backward)."""
        parser = self.parser
        if not parser:
            return
        queue = list[str](input_tensor_names)
        seen = set[str](queue)
        while queue:
            tensor_name = queue.pop(0)
            for pred in parser.predecessors.get(tensor_name, []):
                if not hasattr(pred, "name"):
                    continue
                if pred.name in self._redundant:
                    continue
                self._redundant.add(pred.name)
                for inp in getattr(pred, "input", []):
                    if inp in parser.graph_input or inp in parser.parameters or inp in parser.constant:
                        continue
                    if inp not in seen:
                        seen.add(inp)
                        queue.append(inp)

    def _mark_redundant_post(self, parser: OnnxParser, output_layer_names: List[str]) -> None:
        out_tensors = []
        for ol in output_layer_names:
            if ol in parser.nodes:
                for o in parser.nodes[ol].output:
                    out_tensors.append(o)
        queue = list(out_tensors)
        while queue:
            name = queue.pop(0)
            for succ in parser.successors.get(name, []):
                if hasattr(succ, "name") and succ.name not in self._redundant:
                    self._redundant.add(succ.name)
                    for o in getattr(succ, "output", []):
                        queue.append(o)

    def _remove_redundant_layers(self) -> None:
        if self.ir is None or not self.ir.layers:
            return
        while True:
            next_map: Dict[str, List[str]] = {}
            for name, layer in self.ir.layers.items():
                if not getattr(layer, "inputs", None):
                    continue
                for dd in layer.inputs:
                    if dd.ref is None or not dd.ref.segments:
                        continue
                    prod = dd.ref.segments[0].name
                    if prod in self.ir.layers:
                        next_map.setdefault(prod, []).append(name)
            dead = [k for k, layer in self.ir.layers.items() if getattr(layer, "type", None) == "op" and k not in next_map]
            if not dead:
                break
            for k in dead:
                self.ir.layers.pop(k, None)

    def dump(self, ir_file: Optional[str] = None, *, as_yaml: bool = True) -> Optional[str]:
        """Write IR to file (YAML by default). Returns JSON string if ir_file is None and as_yaml is False."""
        if self.ir is None:
            raise RuntimeError("Run convert() first")
        path = ir_file or self.config.ir_file
        kwargs = {}
        if as_yaml:
            kwargs["config"] = SerializationConfig(default_flow_style=False)
        if path is not None:
            save_ir(self.ir, file=path, **kwargs)
            return None
        return save_ir(self.ir, **kwargs)

    def export_weights(
        self,
        path: Union[str, Path],
        *,
        format: Optional[str] = None,
    ) -> None:
        """
        Export conv/fc weights and BN parameters (from parser.weight_numpy) to a separate file.

        - format 为 None：按 path 扩展名推断（.npz / .npy -> numpy，否则 .pt）。
        - format "pt" / "npz" / "npy"：见 utils.weight_export.export_weights（.npy 仅支持单数组）。

        Keys are mapped to IR layer names (e.g. "Conv_0.weight", "Conv_0.bias").
        Load: .pt -> torch.load(path); .npz -> np.load(path) 得到 NpzFile，keys 为下划线版。
        """
        if self.parser is None:
            raise RuntimeError("Run convert() first")
        if not _TORCH_AVAILABLE:
            raise RuntimeError("export_weights requires torch and numpy. Install: pip install torch")
        name_map = getattr(self.parser, "name_map", {}) or {}
        weight_numpy = self.parser.weight_numpy
        out: Dict[str, Any] = {}
        for key, arr in weight_numpy.items():
            if "." not in key:
                continue
            node_name, suffix = key.rsplit(".", 1)
            ir_name = name_map.get(node_name, node_name)
            new_key = f"{ir_name}.{suffix}"
            if hasattr(arr, "__array__"):
                out[new_key] = np.asarray(arr)
            else:
                out[new_key] = arr
        _export_weights_impl(out, path, format=format)
