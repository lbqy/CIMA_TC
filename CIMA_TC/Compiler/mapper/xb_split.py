"""
First-step mapping pass: adapt conv/linear ops to PE XB constraints.

This module currently focuses on computing a split *plan* based on XB size
and thread limits, and leaves IR graph rewriting for later passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

from ..IR_tool.core import BaseIR


@dataclass
class XBConfig:
    """
    Crossbar configuration for a single PE thread.

    rows:      rows per XB (e.g. 576)
    cols:      cols per XB (e.g. 128)
    max_xbs:   maximum number of XBs that can be used in parallel
               along the column dimension for one thread (e.g. 4).
    """

    rows: int
    cols: int
    max_xbs: int

    @property
    def max_rows(self) -> int:
        return self.rows

    @property
    def max_cols(self) -> int:
        return self.cols * self.max_xbs


def _split_ceil(x: int, limit: int) -> int:
    return (x + limit - 1) // limit


def compute_conv_fc_split_plan(
    weight_shape: List[int],
    *,
    is_conv: bool,
    xb: XBConfig,
) -> Tuple[int, int]:
    """
    Given a conv/fc weight shape and XB config, compute how many splits are
    needed along the *flattened* row and column dimensions.

    For conv:
        weight_shape = [out_c, in_c, kh, kw]
        flattened -> [kh * kw * in_c, out_c]

    For fc:
        weight_shape = [out_c, in_c], flattened -> [in_c, out_c].

    Returns:
        (row_splits, col_splits)
    """
    if is_conv:
        if len(weight_shape) != 4:
            raise ValueError(f"Expected conv weight shape [out_c, in_c, kh, kw], got {weight_shape}")
        out_c, in_c, kh, kw = weight_shape
        rows = kh * kw * in_c
        cols = out_c
    else:
        if len(weight_shape) != 2:
            raise ValueError(f"Expected fc weight shape [out_c, in_c], got {weight_shape}")
        out_c, in_c = weight_shape
        rows = in_c
        cols = out_c

    row_splits = _split_ceil(rows, xb.max_rows)
    col_splits = _split_ceil(cols, xb.max_cols)
    return row_splits, col_splits


def analyze_ir_for_xb_splits(
    ir: BaseIR,
    xb: XBConfig,
) -> Dict[str, Dict[str, Any]]:
    """
    Scan IR layers, find conv2d / linear layers with weight shapes,
    and compute their split plans.

    Returns:
        dict[layer_name] = {
            "op_id": str,
            "weight_shape": List[int],
            "is_conv": bool,
            "row_splits": int,
            "col_splits": int,
        }
    """
    plans: Dict[str, Dict[str, Any]] = {}
    if not getattr(ir, "layers", None):
        return plans

    for name, layer in ir.layers.items():
        op = getattr(layer, "op", None)
        if op is None:
            continue
        op_id = getattr(op, "op_id", None)
        if op_id not in ("conv2d", "linear"):
            continue
        weights = getattr(layer, "weights", None)
        if not weights or "weight" not in weights:
            continue
        w_shape = list(getattr(weights["weight"], "shape", []) or [])
        if not w_shape:
            continue
        is_conv = op_id == "conv2d" and len(w_shape) == 4
        row_splits, col_splits = compute_conv_fc_split_plan(
            w_shape,
            is_conv=is_conv,
            xb=xb,
        )
        plans[name] = {
            "op_id": op_id,
            "weight_shape": w_shape,
            "is_conv": is_conv,
            "row_splits": row_splits,
            "col_splits": col_splits,
        }
    return plans


__all__ = [
    "XBConfig",
    "compute_conv_fc_split_plan",
    "analyze_ir_for_xb_splits",
]

