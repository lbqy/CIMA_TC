from typing import Optional, Tuple, Dict, Any

from ..core import (
    BaseOp, 
    UnaryOp,
    to_integer_tuple,
    is_integer,
    is_boolean,
    is_integers,
    ValidationError
)

# ==========================================================
# Abstract Dot-Based Operator
# ==========================================================

class AbsDotOp(UnaryOp):
    """
    Abstract base class for dot-based operators such as:

        - Fully Connected (Linear)
        - 1x1 Convolution

    Responsibilities:
        - Manage input/output channels
        - Manage weight definitions
        - Validate bias configuration

    This class does NOT perform computation.
    It only defines structural and attribute constraints.
    """

    __abstract__ = True
    # Required and optional weight names
    weights: Tuple[str, ...] = ('weight', 'bias')
    optional_weights: Tuple[str, ...] = ('bias',)

    # Structural attributes
    in_channel: Optional[int] = None
    out_channel: Optional[int] = None
    bias: bool = True

    def __init__(
        self,
        *,
        in_channel: Optional[int] = None,
        out_channel: Optional[int] = None,
        bias: Optional[bool] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize dot-based operator.

        Parameters
        ----------
        in_channel : Optional[int]
            Number of input channels. Must be >= 1.

        out_channel : Optional[int]
            Number of output channels. Must be >= 1.

        bias : Optional[bool]
            Whether bias term is used.
        """

        super().__init__(**kwargs)

        # Validate input channels
        self.set_attr(
            'in_channel',
            in_channel,
            validator=is_integer,
            min_val=1,
            not_none=True
        )

        # Validate output channels
        self.set_attr(
            'out_channel',
            out_channel,
            validator=is_integer,
            min_val=1,
            not_none=True
        )

        # Validate bias flag
        self.set_attr(
            'bias',
            bias if bias is not None else self.bias,
            validator=is_boolean
        )

    def validate(self) -> None:
        """
        Perform additional structural validation.
        """
        super().validate()

        if not self.in_channel > 0:
            raise ValidationError(f"Invalid in_channel={self.in_channel}")

        if not self.out_channel > 0:
            raise ValidationError(f"Invalid out_channel={self.out_channel}")


# ==========================================================
# Abstract Kernel-Based Operator
# ==========================================================

class AbsKernelOp(BaseOp):
    """
    Abstract base class for kernel-based operators such as:

        - Convolution (Conv1D / Conv2D / Conv3D)
        - Pooling layers

    Responsibilities:
        - Manage kernel-related attributes
        - Enforce dimensional consistency
        - Validate padding rules
        - Normalize attributes for export or comparison

    Subclasses MUST define:
        ndim (int) : Spatial dimension of the operator
                     e.g., 1, 2, or 3
    """

    __abstract__ = True
    attrs: Tuple[str, ...] = (
        'stride',
        'padding',
        'dilation',
        'auto_pad',
    )

    # Must be defined in subclass
    ndim: Optional[int] = None

    # Default attribute values
    kernel: Optional[Tuple[int, ...]] = None
    stride: int = 1
    padding: int = 0
    dilation: int = 1
    groups: int = 1
    auto_pad: Optional[str] = None

    AUTO_PADS: Tuple[str, ...] = ('VALID', 'SAME')

    def __init__(
        self,
        *,
        kernel: Optional[Tuple[int, ...]] = None,
        stride: Optional[Tuple[int, ...]] = None,
        padding: Optional[Tuple[int, ...]] = None,
        dilation: Optional[Tuple[int, ...]] = None,
        groups: Optional[int] = None,
        auto_pad: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize kernel-based operator.

        Parameters
        ----------
        kernel : Optional[Tuple[int, ...]]
            Kernel size. Must be positive integers.

        stride : Optional[Tuple[int, ...]]
            Stride values. Must be >= 1.

        padding : Optional[Tuple[int, ...]]
            Padding values. Must be >= 0.

        dilation : Optional[Tuple[int, ...]]
            Dilation factors. Must be >= 1.

        groups : Optional[int]
            Number of groups. Must be >= 1.

        auto_pad : Optional[str]
            Auto padding mode. Must be either 'VALID' or 'SAME'.
        """

        super().__init__(**kwargs)

        assert self.ndim is not None and self.ndim > 0, \
            "Subclass must define a positive ndim."

        allowed_dims = (0, 1, self.ndim)

        # Kernel size
        self.set_attr(
            'kernel',
            to_integer_tuple(kernel, keep_scalar=True),
            validator=is_integers,
            min_val=1,
            ndims=allowed_dims,
            not_none=True
        )

        # Stride
        self.set_attr(
            'stride',
            to_integer_tuple(stride if stride is not None else self.stride,
                         keep_scalar=True),
            validator=is_integers,
            min_val=1,
            ndims=allowed_dims
        )

        # Padding
        self.set_attr(
            'padding',
            to_integer_tuple(padding if padding is not None else self.padding,
                         keep_scalar=True),
            validator=is_integers,
            min_val=0,
            ndims=allowed_dims + (self.ndim * 2,)
        )

        # Dilation
        self.set_attr(
            'dilation',
            to_integer_tuple(dilation if dilation is not None else self.dilation,
                         keep_scalar=True),
            validator=is_integers,
            min_val=1,
            ndims=allowed_dims
        )

        # Groups
        self.set_attr(
            'groups',
            groups if groups is not None else self.groups,
            validator=is_integer,
            min_val=1
        )

        # Auto padding
        if auto_pad is not None:
            auto_pad = auto_pad.upper()

        self.set_attr('auto_pad', auto_pad)

    # ======================================================
    # Validation Logic
    # ======================================================

    def validate(self) -> None:
        """
        Validate kernel-based operator configuration.
        """

        super().validate()

        kernel_tuple = to_integer_tuple(self.kernel, ndim=self.ndim)
        dilation_tuple = to_integer_tuple(self.dilation, ndim=self.ndim)

        # Kernel and dilation must be positive
        assert all(k > 0 for k in kernel_tuple), \
            f"Invalid kernel={self.kernel}"

        assert all(d > 0 for d in dilation_tuple), \
            f"Invalid dilation={self.dilation}"

        # auto_pad and padding conflict check
        if self.auto_pad is not None:
            assert self.padding in (None, 0), \
                "padding conflicts with auto_pad"

            assert self.auto_pad in self.AUTO_PADS, \
                f"Invalid auto_pad={self.auto_pad}"

        else:
            padding_tuple = to_integer_tuple(
                self.padding,
                ndim=self.ndim * 2
            )

            # When dilation == 1, enforce padding < kernel
            if self.dilation == 1:
                for i, pad_value in enumerate(padding_tuple):
                    kernel_dim = kernel_tuple[i % self.ndim]
                    assert pad_value < kernel_dim, \
                        f"Invalid padding={self.padding}"

        # Groups validation
        assert self.groups >= 1, \
            f"Invalid groups={self.groups}"

    # ======================================================
    # Formalized Attribute Export
    # ======================================================

    def formalized_attrs(self) -> Dict[str, Any]:
        """
        Return standardized attributes.

        All scalar values are expanded to full ndim format.

        Returns
        -------
        Dict[str, Any]
            Standardized attribute dictionary.
        """

        kernel_tuple = to_integer_tuple(self.kernel, ndim=self.ndim)
        stride_tuple = to_integer_tuple(self.stride, ndim=self.ndim)
        padding_tuple = to_integer_tuple(
            self.padding,
            ndim=self.ndim * 2,
            keep_scalar=True
        )
        dilation_tuple = to_integer_tuple(self.dilation, ndim=self.ndim)

        return {
            "kernel": kernel_tuple,
            "stride": stride_tuple,
            "padding": padding_tuple,
            "dilation": dilation_tuple,
            "groups": self.groups,
            "auto_pad": self.auto_pad,
        }