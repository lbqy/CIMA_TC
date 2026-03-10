"""
Pooling operators (avg/max, 1d/2d/3d, global).
Aligned with current IR core interface (BaseOp, type_utils).
"""

from typing import Optional, Any, Tuple

from .abs import AbsKernelOp
from ..core import UnaryOp, is_boolean, to_integer_tuple


class PoolOp(AbsKernelOp, UnaryOp):
    """Base pooling op with kernel, stride, padding, ceil_mode."""

    attrs: Tuple[str, ...] = (*AbsKernelOp.attrs, 'kernel', 'ceil_mode')
    ceil_mode: bool = False

    def __init__(
        self,
        *,
        ceil_mode: Optional[bool] = None,
        kernel: Optional[Any] = None,
        stride: Optional[Any] = None,
        **kwargs: Any
    ) -> None:
        if stride is None:
            stride = kernel
        super().__init__(kernel=kernel, stride=stride, **kwargs)
        self.set_attr(
            'ceil_mode',
            ceil_mode if ceil_mode is not None else self.ceil_mode,
            validator=is_boolean
        )


class AvgPoolOp(PoolOp):
    """Average pooling."""
    pass


class MaxPoolOp(PoolOp):
    """Max pooling."""
    pass


class AvgPool1dOp(AvgPoolOp):
    op_id = 'avg_pool1d'
    ndim = 1


class AvgPool2dOp(AvgPoolOp):
    op_id = 'avg_pool2d'
    ndim = 2


class AvgPool3dOp(AvgPoolOp):
    op_id = 'avg_pool3d'
    ndim = 3


class MaxPool1dOp(MaxPoolOp):
    op_id = 'max_pool1d'
    ndim = 1


class MaxPool2dOp(MaxPoolOp):
    op_id = 'max_pool2d'
    ndim = 2


class MaxPool3dOp(MaxPoolOp):
    op_id = 'max_pool3d'
    ndim = 3


class GlobalPoolOp(UnaryOp):
    """Base for global pooling (no kernel)."""
    ndim: Optional[int] = None


class GlobalAvgPoolOp(GlobalPoolOp):
    pass


class GlobalMaxPoolOp(GlobalPoolOp):
    pass


class GlobalAvgPool1dOp(GlobalAvgPoolOp):
    op_id = 'global_avg_pool1d'
    ndim = 1


class GlobalAvgPool2dOp(GlobalAvgPoolOp):
    op_id = 'global_avg_pool2d'
    ndim = 2


class GlobalAvgPool3dOp(GlobalAvgPoolOp):
    op_id = 'global_avg_pool3d'
    ndim = 3


class GlobalMaxPool1dOp(GlobalMaxPoolOp):
    op_id = 'global_max_pool1d'
    ndim = 1


class GlobalMaxPool2dOp(GlobalMaxPoolOp):
    op_id = 'global_max_pool2d'
    ndim = 2


class GlobalMaxPool3dOp(GlobalMaxPoolOp):
    op_id = 'global_max_pool3d'
    ndim = 3
