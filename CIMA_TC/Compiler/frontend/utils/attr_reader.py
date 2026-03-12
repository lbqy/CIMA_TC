"""
Read ONNX node attributes by name.
Each function takes a node (NodeProto) or attribute list and returns the attribute value.
"""

from __future__ import annotations

from typing import Any, List, Optional

# Lazy import onnx to avoid hard dependency at import time
def _node_attr(node: Any, name: str, default: Any = None) -> Any:
    for a in node.attribute:
        if a.name == name:
            if a.type == 1:
                return a.f
            if a.type == 2:
                return a.i
            if a.type == 3:
                return a.s
            if a.type == 4:
                return a.t
            if a.type == 7:
                return list[Any](a.ints)
            if a.type == 6:
                return list[Any](a.floats)
            return getattr(a, "t", getattr(a, "i", getattr(a, "f", default)))
    return default


def get_axis(node: Any) -> int:
    """Axis attribute; 0 if missing."""
    return _node_attr(node, "axis", 0)


def get_axes(node: Any) -> List[int]:
    """Axes attribute; [] if missing."""
    v = _node_attr(node, "axes")
    return list[int](v) if v is not None else []


def get_perm(node: Any) -> List[int]:
    """Perm attribute for Transpose; [] if missing."""
    v = _node_attr(node, "perm")
    return list[int](v) if v is not None else []


def get_alpha(node: Any) -> float:
    """Alpha attribute (e.g. LeakyRelu); 0.0 if missing."""
    v = _node_attr(node, "alpha")
    return float(v) if v is not None else 0.0


def get_keepdims(node: Any) -> int:
    """Keepdims attribute; 1 if missing."""
    v = _node_attr(node, "keepdims")
    return int(v) if v is not None else 1


def get_conv_node_attr(attr: Any) -> tuple:
    """From node.attribute: (strides, pads, kernel_shape). Defaults [0,0],[0,0],[]."""
    stride, pad, kernel = [0, 0], [0, 0], []
    for a in attr:
        if a.name == "strides":
            stride = list[Any](a.ints)
        elif a.name == "pads":
            pad = list[Any](a.ints)
        elif a.name == "kernel_shape":
            kernel = list[Any](a.ints)
    return stride, pad, kernel


def get_conv_node_dilation(attr: Any) -> Any:
    """Dilations from node.attribute; 1 if missing. Returns list of int or 1."""
    for a in attr:
        if a.name == "dilations":
            d = a.ints
            if d is None:
                return 1
            if isinstance(d, (int, float)):
                return [int(d)]
            try:
                return [int(x) for x in d]
            except (TypeError, ValueError):
                return 1
    return 1


def get_split(node: Any) -> Optional[List[int]]:
    """Split attribute (list of sizes); None if missing."""
    v = _node_attr(node, "split")
    if v is None:
        return None
    return [int(x) for x in v] if hasattr(v, "__iter__") and not isinstance(v, (int, float)) else [int(v)]


def get_node_pads(node: Any) -> List[int]:
    """Pads attribute for Pad op."""
    v = _node_attr(node, "pads")
    if v is None:
        return []
    return [int(x) for x in v] if hasattr(v, "__iter__") and not isinstance(v, (int, float)) else [int(v)]


def get_node_epsilon(node: Any) -> float:
    """Epsilon attribute; 0.0 if missing."""
    v = _node_attr(node, "epsilon")
    return float(v) if v is not None else 0.0


def get_pad_mode(node: Any) -> str:
    """Pad mode; 'constant' if missing."""
    v = _node_attr(node, "mode")
    if v is None:
        return "constant"
    s = v.decode() if isinstance(v, bytes) else str(v)
    return "constant" if "constant" in s.lower() else s


def get_pad_value(node: Any) -> float:
    """Pad value (float); 0.0 if missing."""
    v = _node_attr(node, "value")
    return float(v) if v is not None else 0.0


def get_resize_mode(node: Any) -> str:
    """Resize mode; 'nearest' if missing."""
    v = _node_attr(node, "mode")
    if v is None:
        return "nearest"
    s = v.decode() if isinstance(v, bytes) else str(v)
    if "b'" in s:
        s = s[2:-1]
    return s
