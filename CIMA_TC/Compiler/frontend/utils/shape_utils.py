"""Helpers for converting framework tensor shapes into IR metadata dicts."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence


def dim_to_list(dim: Any) -> List[int]:
    """Convert an ONNX shape dim sequence to a list of integer dim values."""
    return [int(d.dim_value) for d in dim]


def tensor_info_from_shape(shape: Sequence[int], *, ref: Optional[str] = None) -> dict[str, Any]:
    """
    Build a DataDef-compatible dict from a tensor shape.

    The current IR tracks NCHW-like tensors as channel/height/width metadata and
    falls back to a raw `shape` field for ranks outside the common cases.
    """
    values = [int(x) for x in shape]
    info: dict[str, Any] = {}
    if ref is not None:
        info["ref"] = ref

    if len(values) == 4:
        info.update(channel=values[1], height=values[2], width=values[3], channel_last=True)
    elif len(values) == 3:
        info.update(channel=values[0], height=values[1], width=values[2], channel_last=True)
    elif len(values) in (1, 2):
        info.update(channel=values[-1], height=1, width=1, channel_last=True)
    else:
        info["shape"] = values
    return info


def get_input_info(input_shape: Sequence[int], ref_name: str) -> dict[str, Any]:
    """Build one IR input spec for a tensor consumed from `ref_name`."""
    return tensor_info_from_shape(input_shape, ref=ref_name)


def get_output_info(out_shape: Sequence[int]) -> List[dict[str, Any]]:
    """Build the single-output IR spec list used by layer constructors."""
    return [tensor_info_from_shape(out_shape)]


def get_weight_info(
    weight_shape: tuple | list,
    bias_shape: Optional[tuple | list] = None,
) -> dict[str, Any]:
    """Build IR weight specs for a required weight and optional bias."""
    out: dict[str, Any] = {"weight": {"shape": list(weight_shape)}}
    if bias_shape is not None:
        out["bias"] = {"shape": list(bias_shape)}
    return out
