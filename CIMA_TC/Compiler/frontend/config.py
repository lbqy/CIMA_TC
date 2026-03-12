"""
Conversion configuration for ONNX -> IR frontend.
Separates options from execution so conversion can be run explicitly via convert().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional


@dataclass
class ConversionConfig:
    """
    Options for ONNX to IR conversion.

    - onnx_file: path to ONNX model file, or None when model is passed in memory.
    - ir_file: path to write IR (YAML/JSON); None = do not save.
    - fix_layer_name: if True, normalize node names (op_type_0, op_type_1, ...).
    - store_intermediate_model: if True, keep intermediate ONNX after preprocessing.
    - specify_input_layer: optional list of node names to use as graph inputs (trim prefix).
    - specify_output_layer: optional list of node names to use as graph outputs (trim suffix).
    - weight_half_level: optional int for symmetric quantization (e.g. 15 for int8).
    - weight_scale: optional dict mapping weight name -> scale (alternative to weight_half_level).
    - data_range_specify: optional for quantization.
    - data_clamp_std: optional; clamp data to ± N*std before quantization.
    """

    onnx_file: Optional[str] = None
    ir_file: Optional[str] = None
    fix_layer_name: bool = False
    store_intermediate_model: bool = False
    specify_input_layer: Optional[List[str]] = None
    specify_output_layer: Optional[List[str]] = None
    weight_half_level: Optional[int] = None
    weight_scale: Optional[dict[str, Any]] = None
    data_range_specify: Optional[Any] = None
    data_clamp_std: float = 0.0

