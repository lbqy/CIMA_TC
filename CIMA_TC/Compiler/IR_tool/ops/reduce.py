"""
Reduce operators (e.g. ReduceMean).
Aligned with current IR core interface.
"""

from typing import Optional, Any

from ..core import UnaryOp, is_integers, is_boolean


class ReduceMeanOp(UnaryOp):
    op_id = 'reduce_mean'
    attrs: tuple = ('axes', 'keepdims')
    axes: Optional[Any] = None
    keepdims: Optional[bool] = None

    def __init__(
        self,
        *,
        axes: Optional[Any] = None,
        keepdims: Optional[bool] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.set_attr('axes', axes, validator=is_integers, min_val=0)
        self.set_attr('keepdims', keepdims, validator=is_boolean)
