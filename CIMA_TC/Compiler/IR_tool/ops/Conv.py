from typing import Optional, Tuple, Dict, Any

from .abs import AbsDotOp, AbsKernelOp
from ..core import (
    is_integers,
    to_integer_tuple,
    is_integer,
)


# ==========================================================
# Grouped Dot Operator
# ==========================================================

class GroupDotOp(AbsDotOp):
    """
    Extension of AbsDotOp to support grouped computation.

    Used by:
        - Grouped Convolution
        - Depthwise Convolution
        - Grouped Linear

    Enforces:
        - in_channel % group == 0
        - out_channel % group == 0
    """

    attrs: Tuple[str, ...] = (*AbsDotOp.attrs, 'group')

    group: int = 1

    def __init__(
        self,
        *,
        group: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.set_attr(
            'group',
            group if group is not None else self.group,
            validator=is_integer,
            min_val=1
        )

    def validate(self) -> None:
        """
        Validate group compatibility with channels.
        """
        super().validate()

        if self.in_channel is not None:
            assert self.in_channel % self.group == 0, \
                f'Invalid group={self.group} for in_channel={self.in_channel}'

        if self.out_channel is not None:
            assert self.out_channel % self.group == 0, \
                f'Invalid group={self.group} for out_channel={self.out_channel}'


# ==========================================================
# Convolution Operator
# ==========================================================

class ConvOp(AbsKernelOp, GroupDotOp):
    """
    General Convolution Operator.

    Combines:
        - Kernel management (AbsKernelOp)
        - Channel + group management (GroupDotOp)

    Multiple inheritance order ensures:
        - Kernel attributes initialized
        - Channel/group validated
    """

    attrs: Tuple[str, ...] = tuple(
        dict.fromkeys((*AbsKernelOp.attrs, *GroupDotOp.attrs))
    )

    ndim: Optional[int] = None

    # ------------------------------------------------------
    # Weight Shape Definition
    # ------------------------------------------------------

    def weight_shapes(
        self,
        channel_last: bool = False,
        **kwargs: Any
    ) -> Dict[str, Optional[Tuple[int, ...]]]:
        """
        Return expected weight shapes.

        Parameters
        ----------
        channel_last : bool
            If True:
                weight shape = (k..., in_channel_per_group, out_channel)
            Else:
                weight shape = (out_channel, in_channel_per_group, k...)

        Returns
        -------
        Dict[str, Optional[Tuple[int, ...]]]
        """

        assert self.in_channel is not None
        assert self.out_channel is not None
        assert self.group > 0

        co: int = self.out_channel
        ci: int = self.in_channel // self.group

        kernel_tuple: Tuple[int, ...] = to_integer_tuple(
            self.kernel,
            ndim=self.ndim
        )

        if channel_last:
            weight_shape = (*kernel_tuple, ci, co)
        else:
            weight_shape = (co, ci, *kernel_tuple)

        bias_shape = None if not self.bias else (co,)

        return dict(weight=weight_shape, bias=bias_shape)


# ==========================================================
# Concrete Conv Operators
# ==========================================================

class Conv1dOp(ConvOp):
    """1D Convolution"""
    op_id: str = 'conv1d'
    ndim: int = 1


class Conv2dOp(ConvOp):
    """2D Convolution"""
    op_id: str = 'conv2d'
    ndim: int = 2


class Conv3dOp(ConvOp):
    """3D Convolution"""
    op_id: str = 'conv3d'
    ndim: int = 3


# ==========================================================
# Transposed Convolution Operator
# ==========================================================

class ConvTransposeOp(AbsKernelOp, GroupDotOp):
    """
    Transposed Convolution (Deconvolution) Operator.

    Adds:
        - output_padding support
    """

    attrs: Tuple[str, ...] = tuple(
        dict.fromkeys(
            (*AbsKernelOp.attrs, *GroupDotOp.attrs, 'output_padding')
        )
    )

    output_padding: int = 0

    def __init__(
        self,
        *,
        output_padding: Optional[Any] = None,
        **kwargs: Any
    ) -> None:

        super().__init__(**kwargs)

        self.set_attr(
            'output_padding',
            to_integer_tuple(output_padding if output_padding is not None
                         else self.output_padding,
                         keep_scalar=True),
            validator=is_integers,
            min_val=0,
            ndims=(0, 1, self.ndim)
        )

    # ------------------------------------------------------
    # Formalized Attributes
    # ------------------------------------------------------

    def formalized_attrs(self) -> Dict[str, Any]:
        """
        Extend parent attributes with output_padding.
        """
        attrs = super().formalized_attrs()

        attrs['output_padding'] = to_integer_tuple(
            self.output_padding,
            ndim=self.ndim
        )

        return attrs

    # ------------------------------------------------------
    # Weight Shapes
    # ------------------------------------------------------

    def weight_shapes(
        self,
        channel_last: bool = False,
        **kwargs: Any
    ) -> Dict[str, Tuple[int, ...]]:
        """
        Return expected weight shapes for transposed convolution.

        Note:
            ConvTranspose swaps in/out channel ordering
            compared to normal Conv.
        """

        assert self.in_channel is not None
        assert self.out_channel is not None

        ci: int = self.in_channel // self.group
        co: int = self.out_channel

        kernel_tuple: Tuple[int, ...] = to_integer_tuple(
            self.kernel,
            ndim=self.ndim
        )

        if channel_last:
            weight_shape = (*kernel_tuple, co, ci)
        else:
            weight_shape = (ci, co, *kernel_tuple)

        result = dict(weight=weight_shape)

        if self.bias:
            result['bias'] = (co,)

        return result


# ==========================================================
# Concrete ConvTranspose Operators
# ==========================================================

class ConvTranspose1dOp(ConvTransposeOp):
    op_id: str = 'conv_transpose1d'
    ndim: int = 1


class ConvTranspose2dOp(ConvTransposeOp):
    op_id: str = 'conv_transpose2d'
    ndim: int = 2


class ConvTranspose3dOp(ConvTransposeOp):
    op_id: str = 'conv_transpose3d'
    ndim: int = 3