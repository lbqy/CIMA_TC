"""
Shape and IR input/output helpers.
Converts ONNX value_info dims to lists and builds IR-style input/output dicts.
"""

from __future__ import annotations

from typing import Any, List, Optional


def dim_to_list(dim: Any) -> List[int]:
    """
    Convert ONNX ValueInfoProto shape dim to list of int (dim_value).
    """
    return [int(d.dim_value) for d in dim]


def get_input_info(input_shape: List[int], ref_name: str) -> dict[str, Any]:
    """
    Build one IR input spec dict for a layer (ref + optional channel/height/width).
    Used when adding a layer that consumes one tensor from ref_name.
    """
    if len(input_shape) == 4:
        # NCHW or NHWC; assume channel_last=True means last is channel
        return dict(
            ref=ref_name,
            channel=input_shape[1],
            height=input_shape[2],
            width=input_shape[3],
            channel_last=True,
        )
    if len(input_shape) in (2, 1):
        return dict(
            ref=ref_name,
            channel=input_shape[-1],
            height=1,
            width=1,
            channel_last=True,
        )
    if len(input_shape) == 3:
        return dict(
            ref=ref_name,
            channel=input_shape[0],
            height=input_shape[1],
            width=input_shape[2],
            channel_last=True,
        )
    return dict(ref=ref_name, shape=input_shape)


def get_output_info(out_shape: List[int]) -> List[dict[str, Any]]:
    """
    Build IR output spec list (one element per output; single output = one dict).
    """
    if len(out_shape) == 4:
        return [dict(channel=out_shape[1], height=out_shape[2], width=out_shape[3], channel_last=True)]
    if len(out_shape) == 2:
        return [dict(channel=out_shape[1], height=1, width=1, channel_last=True)]
    if len(out_shape) == 1:
        return [dict(channel=out_shape[0], height=1, width=1, channel_last=True)]
    if len(out_shape) == 3:
        return [dict(channel=out_shape[0], height=out_shape[1], width=out_shape[2], channel_last=True)]
    return [dict(shape=out_shape)]


def get_weight_info(
    weight_shape: tuple | list,
    bias_shape: Optional[tuple | list] = None,
) -> dict[str, Any]:
    """
    Build IR weights spec: weight (and optionally bias) shape dicts.
    """
    out: dict[str, Any] = {"weight": {"shape": list(weight_shape)}}
    if bias_shape is not None:
        out["bias"] = {"shape": list(bias_shape)}
    return out
