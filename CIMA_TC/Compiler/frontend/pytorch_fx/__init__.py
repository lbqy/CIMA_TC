"""
PyTorch -> IR frontend via torch.fx. Direct conversion without ONNX.
"""

from .convert import ConvertFX
from .config import FXConversionConfig

__all__ = ["ConvertFX", "FXConversionConfig"]
