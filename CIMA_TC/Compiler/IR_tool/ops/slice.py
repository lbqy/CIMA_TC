"""
Slice operator (starts, ends, axes, steps).
Aligned with current IR core interface.
"""

from typing import Optional, Any

from ..core import UnaryOp


class SliceOp(UnaryOp):
    op_id = 'slice'
    attrs: tuple = ('starts', 'ends', 'axes', 'steps')
    starts: Optional[Any] = None
    ends: Optional[Any] = None
    axes: Optional[Any] = None
    steps: Optional[Any] = None

    def __init__(
        self,
        *,
        starts: Optional[Any] = None,
        ends: Optional[Any] = None,
        axes: Optional[Any] = None,
        steps: Optional[Any] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.set_attr('starts', starts)
        self.set_attr('ends', ends)
        self.set_attr('axes', axes)
        self.set_attr('steps', steps)
