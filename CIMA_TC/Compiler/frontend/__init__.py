"""
ONNX -> IR frontend. Refactored from frontend_old with config, op registry, and typed parser.
"""

from .config import ConversionConfig
from .converter import ConvertONNX
from .parser import OnnxParser
from .preprocess import load_onnx_model
from .op_handlers import OP_HANDLERS, get_handler, register_op, UnsupportedONNXOpError

__all__ = [
    "ConversionConfig",
    "ConvertONNX",
    "OnnxParser",
    "load_onnx_model",
    "OP_HANDLERS",
    "get_handler",
    "register_op",
    "UnsupportedONNXOpError",
]
