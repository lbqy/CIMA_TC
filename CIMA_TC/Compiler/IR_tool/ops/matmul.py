"""
MatMul / Linear / FC operator.
Aligned with current IR core interface.
"""

from typing import Dict, Any, Optional, Tuple

from .abs import AbsDotOp


class MatMulOp(AbsDotOp):
    """Matrix multiply / fully connected; single op_id for matmul, linear, fc."""

    op_id = 'matmul'

    def weight_shapes(
        self,
        *,
        channel_last: bool = False,
        **kwargs: Any
    ) -> Dict[str, Optional[Tuple[int, ...]]]:
        if self.in_channel is None or self.out_channel is None:
            raise ValueError("in_channel and out_channel must be set")
        co = self.out_channel
        ci = self.in_channel
        return dict[str, Tuple[int, ...] | None](
            weight=(ci, co) if channel_last else (co, ci),
            bias=None if not self.bias else (co,)
        )
