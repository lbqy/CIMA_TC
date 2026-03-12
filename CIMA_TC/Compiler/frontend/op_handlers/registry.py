"""
Op handler registry: map ONNX op_type to handler(ir, parser, node_name).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


class UnsupportedONNXOpError(ValueError):
    """Raised when an ONNX op_type has no registered handler."""

    def __init__(self, op_type: str, node_name: Optional[str] = None) -> None:
        self.op_type = op_type
        self.node_name = node_name
        super().__init__(f"Unsupported ONNX op: {op_type}" + (f" (node {node_name})" if node_name else ""))


# op_type -> (ir, parser, node_name) -> None
OP_HANDLERS: Dict[str, Callable[[Any, Any, str], None]] = {}


def register_op(op_type: str) -> Callable[[Callable[[Any, Any, str], None]], Callable[[Any, Any, str], None]]:
    """Decorator to register a handler for an ONNX op_type."""

    def deco(fn: Callable[[Any, Any, str], None]) -> Callable[[Any, Any, str], None]:
        OP_HANDLERS[op_type] = fn
        return fn

    return deco


def get_handler(op_type: str) -> Optional[Callable[[Any, Any, str], None]]:
    """Return registered handler for op_type or None."""
    return OP_HANDLERS.get(op_type)
