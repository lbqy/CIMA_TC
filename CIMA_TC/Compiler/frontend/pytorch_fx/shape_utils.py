"""
Shape helpers for FX -> IR conversion.
"""

from __future__ import annotations

from typing import Any, List, Optional


def get_shape_from_meta(node: Any) -> Optional[List[int]]:
    """
    Extract shape from FX node meta (tensor_meta or val).
    Returns list of ints or None if not available.
    """
    meta = getattr(node, "meta", {})
    if not meta:
        return None
    # tensor_meta: TensorMetadata with .shape
    tm = meta.get("tensor_meta")
    if tm is not None and hasattr(tm, "shape"):
        s = tm.shape
        if s is not None:
            out = []
            for d in s:
                try:
                    out.append(int(d))
                except (TypeError, ValueError):
                    out.append(-1)
            return out
    # val: actual tensor (e.g. from ShapeProp)
    val = meta.get("val")
    if val is not None and hasattr(val, "shape"):
        return [int(x) for x in val.shape]
    return None


def shape_to_input_info(shape: List[int], ref: str) -> dict[str, Any]:
    """Build IR input spec from shape (NCHW or 2D)."""
    if len(shape) == 4:
        return dict(
            ref=ref,
            channel=shape[1],
            height=shape[2],
            width=shape[3],
            channel_last=True,
        )
    if len(shape) in (2, 1):
        return dict(
            ref=ref,
            channel=shape[-1],
            height=1,
            width=1,
            channel_last=True,
        )
    if len(shape) == 3:
        return dict(
            ref=ref,
            channel=shape[0],
            height=shape[1],
            width=shape[2],
            channel_last=True,
        )
    return dict(ref=ref, shape=shape)


def shape_to_output_info(shape: List[int]) -> List[dict[str, Any]]:
    """Build IR output spec from shape."""
    if len(shape) == 4:
        return [dict(channel=shape[1], height=shape[2], width=shape[3], channel_last=True)]
    if len(shape) == 2:
        return [dict(channel=shape[1], height=1, width=1, channel_last=True)]
    if len(shape) == 1:
        return [dict(channel=shape[0], height=1, width=1, channel_last=True)]
    if len(shape) == 3:
        return [dict(channel=shape[0], height=shape[1], width=shape[2], channel_last=True)]
    return [dict(shape=shape)]


def get_weight_info(weight_shape: tuple | list, bias_shape: Optional[tuple | list] = None) -> dict[str, Any]:
    """Build IR weights spec."""
    out: dict[str, Any] = {"weight": {"shape": list(weight_shape)}}
    if bias_shape is not None:
        out["bias"] = {"shape": list(bias_shape)}
    return out
