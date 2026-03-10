"""
Shape/transform operators: Concat, Reshape, Flatten, Transpose, Pad.
Aligned with current IR core interface.
"""

from typing import Optional, Any, Tuple

from ..core import (
    BaseOp,
    UnaryOp,
    ValidationError,
    is_integer,
    is_integers,
    is_boolean,
    is_number,
    is_in_values,
    to_integer_tuple,
)


class ShapeOp(BaseOp):
    """Base for shape-related ops (with_batch)."""

    attrs: Tuple[str, ...] = ('with_batch',)
    with_batch: bool = True

    def __init__(self, *, with_batch: Optional[bool] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.set_attr(
            'with_batch',
            with_batch if with_batch is not None else self.with_batch,
            validator=is_boolean
        )


class ConcatOp(ShapeOp):
    """Concat on axis; channel_pos: first, last, ignore."""

    op_id = 'concat'
    attrs: Tuple[str, ...] = (*ShapeOp.attrs, 'axis', 'channel_pos')
    axis: Optional[int] = None
    channel_pos: str = 'first'
    CHANNEL_POS: Tuple[str, ...] = ('first', 'last', 'ignore')

    def __init__(
        self,
        *,
        axis: Optional[int] = None,
        channel_pos: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.set_attr('axis', axis, validator=is_integer, not_none=True)
        normalized = str(channel_pos).lower() if channel_pos is not None else self.channel_pos
        self.set_attr(
            'channel_pos',
            normalized,
            validator=is_in_values,
            values=self.CHANNEL_POS
        )


class ReshapeOp(ShapeOp, UnaryOp):
    """Reshape to given shape."""

    op_id = 'reshape'
    attrs: Tuple[str, ...] = (*ShapeOp.attrs, 'shape')
    shape: Optional[Any] = None

    def __init__(self, *, shape: Optional[Any] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.set_attr(
            'shape',
            to_integer_tuple(shape) if shape is not None else None,
            validator=is_integers,
            min_val=-1,
            not_none=True
        )


class FlattenOp(ShapeOp, UnaryOp):
    """Flatten from start_dim."""

    op_id = 'flatten'
    attrs: Tuple[str, ...] = (*ShapeOp.attrs, 'start_dim')
    start_dim: int = 1

    def __init__(self, *, start_dim: Optional[int] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.set_attr(
            'start_dim',
            start_dim if start_dim is not None else self.start_dim,
            validator=is_integer
        )


class TransposeOp(ShapeOp, UnaryOp):
    """Transpose with perm."""

    op_id = 'transpose'
    attrs: Tuple[str, ...] = (*ShapeOp.attrs, 'perm')
    perm: Optional[Any] = None

    def __init__(self, *, perm: Optional[Any] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.set_attr(
            'perm',
            to_integer_tuple(perm) if perm is not None else None,
            validator=is_integers,
            min_val=0,
            min_dim=2,
            not_none=True
        )

    def validate(self) -> None:
        super().validate()
        p = list(self.perm) if self.perm is not None else []
        p_sorted = sorted(p)
        if not p or p_sorted != list(range(len(p))):
            raise ValidationError(f"invalid perm={self.perm}")


class PadOp(UnaryOp):
    """Pad with pads and value."""

    op_id = 'pad'
    attrs: Tuple[str, ...] = ('pads', 'value')
    value: float = 0

    def __init__(
        self,
        *,
        pads: Optional[Any] = None,
        value: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.set_attr(
            'pads',
            to_integer_tuple(pads) if pads is not None else None,
            validator=is_integers,
            min_val=0,
            min_dim=2
        )
        self.set_attr(
            'value',
            value if value is not None else self.value,
            validator=is_number
        )

    def validate(self) -> None:
        super().validate()
        pads = self.pads
        if pads is not None and len(pads) not in (2, 4, 6, 8):
            raise ValidationError(f"pads length must be 2, 4, 6, or 8, got {len(pads)}")
