"""Small readers for ONNX node attributes."""

from __future__ import annotations

from typing import Any, List, Optional


def _as_int_list(values: Any) -> List[int]:
    return [int(x) for x in values]


def _as_float_list(values: Any) -> List[float]:
    return [float(x) for x in values]


def _node_attr(node: Any, name: str, default: Any = None) -> Any:
    for attr in node.attribute:
        if attr.name != name:
            continue
        if attr.type == 1:
            return attr.f
        if attr.type == 2:
            return attr.i
        if attr.type == 3:
            return attr.s
        if attr.type == 4:
            return attr.t
        if attr.type == 6:
            return _as_float_list(attr.floats)
        if attr.type == 7:
            return _as_int_list(attr.ints)
        return getattr(attr, "t", getattr(attr, "i", getattr(attr, "f", default)))
    return default


def get_axis(node: Any) -> int:
    """Return axis, defaulting to 0."""
    return int(_node_attr(node, "axis", 0))


def get_axes(node: Any) -> List[int]:
    """Return axes, defaulting to an empty list."""
    value = _node_attr(node, "axes")
    return _as_int_list(value) if value is not None else []


def get_perm(node: Any) -> List[int]:
    """Return Transpose perm, defaulting to an empty list."""
    value = _node_attr(node, "perm")
    return _as_int_list(value) if value is not None else []


def get_alpha(node: Any) -> float:
    """Return alpha, defaulting to 0.0."""
    value = _node_attr(node, "alpha")
    return float(value) if value is not None else 0.0


def get_keepdims(node: Any) -> int:
    """Return keepdims, defaulting to 1."""
    value = _node_attr(node, "keepdims")
    return int(value) if value is not None else 1


def get_conv_node_attr(attr: Any) -> tuple:
    """Return (strides, pads, kernel_shape) from a Conv-like attr list."""
    stride, pad, kernel = [0, 0], [0, 0], []
    for item in attr:
        if item.name == "strides":
            stride = _as_int_list(item.ints)
        elif item.name == "pads":
            pad = _as_int_list(item.ints)
        elif item.name == "kernel_shape":
            kernel = _as_int_list(item.ints)
    return stride, pad, kernel


def get_conv_node_dilation(attr: Any) -> Any:
    """Return dilations as a list of ints, or 1 when missing/invalid."""
    for item in attr:
        if item.name != "dilations":
            continue
        values = item.ints
        if values is None:
            return 1
        if isinstance(values, (int, float)):
            return [int(values)]
        try:
            return _as_int_list(values)
        except (TypeError, ValueError):
            return 1
    return 1


def get_split(node: Any) -> Optional[List[int]]:
    """Return Split sizes, or None when missing."""
    value = _node_attr(node, "split")
    if value is None:
        return None
    return _as_int_list(value) if hasattr(value, "__iter__") and not isinstance(value, (int, float)) else [int(value)]


def get_node_pads(node: Any) -> List[int]:
    """Return Pad pads attribute."""
    value = _node_attr(node, "pads")
    if value is None:
        return []
    return _as_int_list(value) if hasattr(value, "__iter__") and not isinstance(value, (int, float)) else [int(value)]


def get_node_epsilon(node: Any) -> float:
    """Return epsilon, defaulting to 0.0."""
    value = _node_attr(node, "epsilon")
    return float(value) if value is not None else 0.0


def _decode_attr_string(value: Any, default: str) -> str:
    if value is None:
        return default
    text = value.decode() if isinstance(value, bytes) else str(value)
    if text.startswith("b'") and text.endswith("'"):
        return text[2:-1]
    return text


def get_pad_mode(node: Any) -> str:
    """Return Pad mode, defaulting to constant."""
    mode = _decode_attr_string(_node_attr(node, "mode"), "constant")
    return "constant" if "constant" in mode.lower() else mode


def get_pad_value(node: Any) -> float:
    """Return constant pad value, defaulting to 0.0."""
    value = _node_attr(node, "value")
    return float(value) if value is not None else 0.0


def get_resize_mode(node: Any) -> str:
    """Return Resize mode, defaulting to nearest."""
    return _decode_attr_string(_node_attr(node, "mode"), "nearest")
