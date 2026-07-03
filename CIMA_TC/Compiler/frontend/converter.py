"""ONNX to IR converter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

try:
    import numpy as np
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from CIMA_TC.Compiler.IR_tool.core.ir import BaseIR, save_ir
from CIMA_TC.Compiler.IR_tool.core.jsonable import SerializationConfig

from .config import ConversionConfig
from .op_handlers import UnsupportedONNXOpError, get_handler
from .parser import OnnxParser
from .preprocess import load_onnx_model
from .utils.ir_rewrite import fuse_sigmoid_mul_to_silu, rename_batchnorm_layers
from .utils.name_sanitize import build_layer_name_map
from .utils.onnx_io import load_onnx
from .utils.shape_utils import dim_to_list, tensor_info_from_shape
from .utils.weight_export import export_weights as _export_weights_impl


_CONFIG_KWARGS = {
    "onnx_file",
    "ir_file",
    "fix_layer_name",
    "store_intermediate_model",
    "specify_input_layer",
    "specify_output_layer",
    "weight_half_level",
    "weight_scale",
    "data_range_specify",
    "data_clamp_std",
    "enable_ir_rewrite",
}


class ConvertONNX:
    """Convert an ONNX model into the project IR."""

    def __init__(self, config: Optional[ConversionConfig] = None, **kwargs: Any) -> None:
        self.config = config if config is not None else ConversionConfig(**{k: v for k, v in kwargs.items() if k in _CONFIG_KWARGS})
        self.ir: Optional[BaseIR] = None
        self.parser: Optional[OnnxParser] = None
        self.updated_name_dict: Optional[Dict[str, str]] = None
        self._redundant: Set[str] = set()

    def convert(self, onnx_file: Optional[str] = None, onnx_model: Any = None) -> BaseIR:
        """Run ONNX preprocessing, parsing, and IR construction."""
        model = self._load_and_preprocess(onnx_file=onnx_file, onnx_model=onnx_model)
        self.parser = self._build_parser(model)
        parser = self.parser
        self._configure_graph_boundaries(parser)

        self.ir = BaseIR.make_ir()
        parser.name_map = build_layer_name_map(list(parser.nodes.keys()))

        self._add_graph_input_layer(parser)
        self._add_op_layers(parser)
        self._add_graph_output_layer(parser)
        self._remove_redundant_layers()
        self._apply_ir_rewrites()
        self._attach_runtime_weight_stores()
        return self.ir

    def _load_and_preprocess(self, *, onnx_file: Optional[str], onnx_model: Any) -> Any:
        cfg = self.config
        model = onnx_model
        if model is None:
            path = onnx_file or cfg.onnx_file
            if path is None:
                raise ValueError("Provide onnx_file path or onnx_model")
            model = load_onnx(path)

        model, self.updated_name_dict = load_onnx_model(
            model,
            fix_layer_name=cfg.fix_layer_name,
            store_intermediate_model=cfg.store_intermediate_model,
        )
        return model

    def _build_parser(self, model: Any) -> OnnxParser:
        cfg = self.config
        return OnnxParser(
            model,
            weight_half_level=cfg.weight_half_level,
            weight_scale=cfg.weight_scale,
            data_clamp_std=cfg.data_clamp_std,
            data_range_specify=cfg.data_range_specify,
        )

    def _configure_graph_boundaries(self, parser: OnnxParser) -> None:
        cfg = self.config
        if cfg.specify_input_layer:
            parser.inputs = [parser.nodes[name].input[0] for name in cfg.specify_input_layer if name in parser.nodes]
            self._mark_redundant_pre(parser.inputs)
        else:
            parser.inputs = list(parser.graph_input)

        if cfg.specify_output_layer:
            self._mark_redundant_post(parser, cfg.specify_output_layer)

    def _add_graph_input_layer(self, parser: OnnxParser) -> None:
        assert self.ir is not None
        inputs = []
        for input_name in parser.inputs:
            shape = self._shape_for_value(parser, input_name)
            if len(shape) not in (2, 3, 4):
                raise ValueError(f"Unsupported input rank: {shape}")
            inputs.append(tensor_info_from_shape(shape))
        self.ir.add_layer("graph_input", type="input", outputs=inputs)

    def _add_op_layers(self, parser: OnnxParser) -> None:
        assert self.ir is not None
        for node_name, node in parser.nodes.items():
            if node_name in self._redundant:
                continue
            handler = get_handler(node.op_type)
            if handler is None:
                raise UnsupportedONNXOpError(node.op_type, node_name)
            handler(self.ir, parser, node_name)

    def _add_graph_output_layer(self, parser: OnnxParser) -> None:
        assert self.ir is not None
        outputs = []
        for out_name in self._output_tensor_names(parser):
            vi = parser.value_infos.get(out_name)
            if vi is None:
                continue
            shape = dim_to_list(vi.type.tensor_type.shape.dim)
            outputs.append(tensor_info_from_shape(shape, ref=self._output_ref(parser, out_name)))
        self.ir.add_layer("graph_output", type="output", inputs=outputs)

    def _output_tensor_names(self, parser: OnnxParser) -> List[str]:
        cfg = self.config
        if not cfg.specify_output_layer:
            return list(parser.graph_output)
        names: List[str] = []
        for layer_name in cfg.specify_output_layer:
            if layer_name in parser.nodes:
                names.extend(parser.nodes[layer_name].output)
        return names

    @staticmethod
    def _output_ref(parser: OnnxParser, output_name: str) -> str:
        pred_list = parser.predecessors.get(output_name, [])
        ref_name = pred_list[0].name if pred_list and hasattr(pred_list[0], "name") else output_name
        return parser.name_map.get(ref_name, ref_name)

    @staticmethod
    def _shape_for_value(parser: OnnxParser, value_name: str) -> List[int]:
        vi = parser.value_infos.get(value_name)
        if vi is None:
            raise ValueError(f"Missing value_info for graph input {value_name!r}")
        return dim_to_list(vi.type.tensor_type.shape.dim)

    def _apply_ir_rewrites(self) -> None:
        if not getattr(self.config, "enable_ir_rewrite", True):
            return
        self._fuse_sigmoid_mul_to_silu()
        self._rename_batchnorm_layers()

    def _fuse_sigmoid_mul_to_silu(self) -> None:
        if self.ir is not None:
            fuse_sigmoid_mul_to_silu(self.ir, name_map=getattr(self.parser, "name_map", None))

    def _rename_batchnorm_layers(self) -> None:
        if self.ir is not None:
            rename_batchnorm_layers(self.ir, name_map=getattr(self.parser, "name_map", None))

    def _attach_runtime_weight_stores(self) -> None:
        if self.ir is None or self.parser is None:
            return
        weight_store = getattr(self.ir, "weight_store", None)
        bn_store = getattr(self.ir, "bn_store", None)
        if not isinstance(weight_store, dict) or not isinstance(bn_store, dict):
            return

        for key, arr in self._mapped_weight_numpy().items():
            if key.endswith((".running_mean", ".running_var")):
                bn_store[key] = arr
            else:
                weight_store[key] = arr

    def _mapped_weight_numpy(self) -> Dict[str, Any]:
        if self.parser is None:
            return {}
        name_map = getattr(self.parser, "name_map", {}) or {}
        out: Dict[str, Any] = {}
        for key, arr in self.parser.weight_numpy.items():
            if "." not in key:
                continue
            node_name, suffix = key.rsplit(".", 1)
            out[f"{name_map.get(node_name, node_name)}.{suffix}"] = arr
        return out

    def _mark_redundant_pre(self, input_tensor_names: List[str]) -> None:
        parser = self.parser
        if parser is None:
            return
        queue = list(input_tensor_names)
        seen = set(queue)
        while queue:
            tensor_name = queue.pop(0)
            for pred in parser.predecessors.get(tensor_name, []):
                if not hasattr(pred, "name") or pred.name in self._redundant:
                    continue
                self._redundant.add(pred.name)
                for inp in getattr(pred, "input", []):
                    if inp in parser.graph_input or inp in parser.parameters or inp in parser.constant:
                        continue
                    if inp not in seen:
                        seen.add(inp)
                        queue.append(inp)

    def _mark_redundant_post(self, parser: OnnxParser, output_layer_names: List[str]) -> None:
        queue = self._selected_output_tensors(parser, output_layer_names)
        while queue:
            name = queue.pop(0)
            for succ in parser.successors.get(name, []):
                if hasattr(succ, "name") and succ.name not in self._redundant:
                    self._redundant.add(succ.name)
                    queue.extend(getattr(succ, "output", []))

    @staticmethod
    def _selected_output_tensors(parser: OnnxParser, output_layer_names: List[str]) -> List[str]:
        out_tensors: List[str] = []
        for layer_name in output_layer_names:
            if layer_name in parser.nodes:
                out_tensors.extend(parser.nodes[layer_name].output)
        return out_tensors

    def _remove_redundant_layers(self) -> None:
        if self.ir is None or not self.ir.layers:
            return
        while True:
            next_map: Dict[str, List[str]] = {}
            for name, layer in self.ir.layers.items():
                for dd in getattr(layer, "inputs", None) or []:
                    if dd.ref is None or not dd.ref.segments:
                        continue
                    prod = dd.ref.segments[0].name
                    if prod in self.ir.layers:
                        next_map.setdefault(prod, []).append(name)
            dead = [
                name
                for name, layer in self.ir.layers.items()
                if getattr(layer, "type", None) == "op" and name not in next_map
            ]
            if not dead:
                return
            for name in dead:
                self.ir.layers.pop(name, None)

    def dump(self, ir_file: Optional[str] = None, *, as_yaml: bool = True) -> Optional[str]:
        """Write IR to file, or return a serialized string when no file is given."""
        if self.ir is None:
            raise RuntimeError("Run convert() first")
        path = ir_file or self.config.ir_file
        kwargs = {"config": SerializationConfig(default_flow_style=False)} if as_yaml else {}
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
        """Export converted layer weights using IR layer names as keys."""
        if self.parser is None:
            raise RuntimeError("Run convert() first")
        if not _TORCH_AVAILABLE:
            raise RuntimeError("export_weights requires torch and numpy. Install: pip install torch")

        out: Dict[str, Any] = {}
        for key, arr in self._mapped_weight_numpy().items():
            out[key] = np.asarray(arr) if hasattr(arr, "__array__") else arr
        _export_weights_impl(out, path, format=format)
