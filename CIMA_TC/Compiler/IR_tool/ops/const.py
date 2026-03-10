"""
Constant and Identity operators.
Aligned with current IR core interface.
"""

from ..core import BaseOp, UnaryOp


class ConstantOp(BaseOp):
    """Constant value; num_inputs=0."""

    op_id = 'constant'
    attrs: tuple = ('value',)
    num_inputs: int = 0
    value: object = None

    def __init__(self, *, value: object = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.set_attr('value', value, not_none=True)


class IdentityOp(UnaryOp):
    op_id = 'identity'
