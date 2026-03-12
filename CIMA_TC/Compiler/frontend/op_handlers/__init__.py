"""ONNX op -> IR layer handlers. Register handlers and look up by op_type."""

from .registry import OP_HANDLERS, register_op, UnsupportedONNXOpError, get_handler

# Import so all @register_op decorators run
from . import handlers  # noqa: F401

__all__ = ["OP_HANDLERS", "register_op", "UnsupportedONNXOpError", "get_handler"]
