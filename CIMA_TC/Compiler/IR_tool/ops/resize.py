"""
Resize and shape ops: Resize, Squeeze, Unsqueeze, Gather.
Aligned with current IR core interface.
"""

from typing import Optional, Any

from ..core import UnaryOp, is_integers, is_numbers, to_integer_tuple


class ResizeOp(UnaryOp):
    """Resize with size or scale; mode e.g. nearest."""

    op_id = 'resize'
    attrs: tuple = ('size', 'scale', 'mode')
    size: Optional[Any] = None
    scale: Optional[Any] = None
    mode: str = 'nearest'

    def __init__(
        self,
        *,
        size: Optional[Any] = None,
        scale: Optional[Any] = None,
        mode: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if (size is None) == (scale is None):
            raise ValueError(f"exactly one of size or scale must be set; got size={size!r}, scale={scale!r}")
        if size is not None:
            self.set_attr(
                'size',
                to_integer_tuple(size, keep_scalar=True),
                validator=is_integers,
                min_val=1,
                min_dim=0,
                max_dim=4
            )
            self.set_attr('scale', None)
        else:
            self.set_attr('scale', scale, validator=is_numbers, lower_limit=0, min_dim=0, max_dim=4)
            self.set_attr('size', None)
        self.set_attr('mode', (mode or self.mode).lower() if (mode or self.mode) else (mode or self.mode))


class SqueezeOp(UnaryOp):
    op_id = 'squeeze'
    attrs: tuple = ('axes',)
    axes: Optional[Any] = None

    def __init__(self, *, axes: Optional[Any] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.set_attr('axes', axes)


class UnsqueezeOp(UnaryOp):
    op_id = 'unsqueeze'
    attrs: tuple = ('axes',)
    axes: Optional[Any] = None

    def __init__(self, *, axes: Optional[Any] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.set_attr('axes', axes)


class GatherOp(UnaryOp):
    op_id = 'gather'
    attrs: tuple = ('axis', 'indices')
    axis: Optional[Any] = None
    indices: Optional[Any] = None

    def __init__(
        self,
        *,
        axis: Optional[Any] = None,
        indices: Optional[Any] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.set_attr('axis', axis)
        self.set_attr('indices', indices)
