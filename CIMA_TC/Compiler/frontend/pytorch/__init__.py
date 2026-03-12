"""
PyTorch -> IR frontend. Converts nn.Module to IR via ONNX export.
Separate from the ONNX frontend; uses it internally after export.
"""

from .convert import ConvertTorch
from .config import TorchConversionConfig

__all__ = ["ConvertTorch", "TorchConversionConfig"]
