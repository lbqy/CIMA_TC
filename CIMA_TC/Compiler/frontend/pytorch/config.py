"""
Conversion configuration for PyTorch -> IR frontend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class TorchConversionConfig:
    """
    Options for PyTorch model to IR conversion.

    - example_input: required; tensor(s) for tracing/export. Single tensor or tuple of tensors.
    - ir_file: path to write IR (YAML/JSON); None = do not save.
    - fix_layer_name: if True, normalize ONNX node names (op_type_0, op_type_1, ...).
    - store_intermediate_model: if True, keep intermediate ONNX after preprocessing.
    - opset_version: ONNX opset for export (default 11).
    - dynamic_axes: optional dict for dynamic batch/sequence axes.
    - do_constant_folding: whether to fold constants during ONNX export (default True).
    - input_names: optional list of input names for ONNX.
    - output_names: optional list of output names for ONNX.
    """

    example_input: Any = None
    ir_file: Optional[str] = None
    fix_layer_name: bool = True
    store_intermediate_model: bool = False
    opset_version: int = 11
    dynamic_axes: Optional[dict[str, Any]] = None
    do_constant_folding: bool = True
    input_names: Optional[List[str]] = None
    output_names: Optional[List[str]] = None

    # Frontend IR rewrite passes (default enabled):
    # - fuse Sigmoid+Mul -> Silu
    # - rename BN to "{nearest_conv_or_fc}_bn"
    enable_ir_rewrite: bool = True
