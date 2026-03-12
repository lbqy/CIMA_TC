"""
MatMul vs FC/Linear operators.

- MatMulOp: two dynamic inputs (e.g. attention Q@K), op_id="matmul", num_inputs=2, no weights.
- LinearOp / FCOp: one data input + static weight (and optional bias), op_id="linear" / "fc",
  num_inputs=1, weights=(weight, bias).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from ..core import BaseOp, BinaryOp
from .abs import AbsDotOp


# ============================================================
# MatMul: two dynamic matrices (e.g. attention)
# ============================================================


class MatMulOp(BinaryOp):
    """
    Matrix multiplication of two dynamic inputs: out = A @ B.

    Used for e.g. attention (Q @ K^T), or any two-tensor matmul.
    No static weights; both inputs are runtime tensors.
    """

    op_id = "matmul"
    num_inputs = 2
    attrs: Tuple[str, ...] = ()
    weights: Tuple[str, ...] = ()


# ============================================================
# Linear / FC: one input + static weight (and optional bias)
# ============================================================


class LinearOp(AbsDotOp):
    """
    Fully connected (linear) layer: out = input @ weight + bias.

    One data input; weight and optional bias are stored parameters.
    op_id = "linear".
    """

    op_id = "linear"

    def weight_shapes(
        self,
        *,
        channel_last: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Optional[Tuple[int, ...]]]:
        if self.in_channel is None or self.out_channel is None:
            raise ValueError("in_channel and out_channel must be set")
        ci, co = self.in_channel, self.out_channel
        # Linear: weight (out_features, in_features) in common layout
        return {
            "weight": (co, ci) if not channel_last else (ci, co),
            "bias": (co,) if self.bias else None,
        }


class FCOp(LinearOp):
    """
    Alias for fully connected layer; same as LinearOp, op_id = "fc".
    """

    op_id = "fc"
