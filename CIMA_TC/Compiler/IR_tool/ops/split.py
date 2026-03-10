"""
Split operator (multi-branch output).
Aligned with current IR core interface.
"""

from typing import Optional, Any

from ..core import UnaryOp, is_integer, is_integers, to_integer_tuple


class SplitOp(UnaryOp):
    """Split on axis; split is number of outputs or section sizes."""

    op_id = 'split'
    attrs: tuple = ('axis', 'split', 'with_batch')
    axis: Optional[int] = None
    split: Optional[Any] = None
    with_batch: bool = True

    def __init__(
        self,
        *,
        axis: Optional[int] = None,
        split: Optional[Any] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.set_attr('axis', axis, validator=is_integer, min_val=0)
        self.set_attr(
            'split',
            to_integer_tuple(split, keep_scalar=True) if split is not None else None,
            validator=is_integers,
            min_val=1,
            allow_scalar=True
        )

    @property
    def num_outputs(self) -> int:
        if self.split is None:
            return 0
        if isinstance(self.split, int):
            return self.split
        return len(self.split)
