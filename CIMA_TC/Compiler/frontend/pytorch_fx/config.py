"""
Conversion configuration for PyTorch FX -> IR frontend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class FXConversionConfig:
    """
    Options for PyTorch FX model to IR conversion.

    - example_input: required; tensor(s) for tracing. Single tensor or tuple.
    - ir_file: path to write IR (YAML); None = do not save.
    - layer_name_prefix: optional prefix for layer names (e.g. "fx_").
    """

    example_input: Any = None
    ir_file: Optional[str] = None
    layer_name_prefix: str = ""
