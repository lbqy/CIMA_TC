"""
Normalization operators (BatchNorm, InstanceNorm, LayerNorm).
Aligned with current IR core interface.
"""

from typing import Optional, Any, Tuple, Dict

from ..core import UnaryOp, is_integer, is_number


class BatchNormOp(UnaryOp):
    """Batch normalization."""

    op_id = 'batch_norm'
    attrs: Tuple[str, ...] = ('epsilon', 'scale', 'bias', 'input_mean', 'input_var')
    weights: Tuple[str, ...] = ('scale', 'bias', 'input_mean', 'input_var')
    unsigned_weights: Tuple[str, ...] = ('input_var',)
    ndim: Optional[int] = None
    epsilon: float = 1e-5
    scale: int = 1
    bias: int = 0

    def __init__(
        self,
        *,
        channel: Optional[int] = None,
        epsilon: Optional[float] = None,
        scale: Optional[Any] = None,
        bias: Optional[Any] = None,
        input_mean: Optional[Any] = None,
        input_var: Optional[Any] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.set_attr('channel', channel, validator=is_integer, min_val=1)
        self.set_attr(
            'epsilon',
            epsilon if epsilon is not None else self.epsilon,
            validator=is_number,
            lower_limit=0,
            upper_limit=1.0
        )
        # BN 参数不写入 IR，由 export_weights 单独导出；此处仅接受 None 或保留兼容旧 IR
        self.set_attr('scale', scale)
        self.set_attr('bias', bias)
        self.set_attr('input_mean', input_mean)
        self.set_attr('input_var', input_var)

    def weight_shapes(self, **kwargs: Any) -> Dict[str, tuple]:
        c = self.channel
        if c is None:
            raise ValueError("channel must be set for weight_shapes")
        return dict(scale=(c,), bias=(c,), input_mean=(c,), input_var=(c,))


class BatchNorm1dOp(BatchNormOp):
    op_id = 'batch_norm1d'
    ndim = 1


class BatchNorm2dOp(BatchNormOp):
    op_id = 'batch_norm2d'
    ndim = 2


class BatchNorm3dOp(BatchNormOp):
    op_id = 'batch_norm3d'
    ndim = 3


class InstanceNormOp(UnaryOp):
    """Instance normalization."""

    attrs: Tuple[str, ...] = ('epsilon',)
    weights: Tuple[str, ...] = ('scale', 'bias')
    ndim: Optional[int] = None
    epsilon: float = 1e-5
    scale: int = 1
    bias: int = 0

    def __init__(
        self,
        *,
        channel: Optional[int] = None,
        epsilon: Optional[float] = None,
        scale: Optional[Any] = None,
        bias: Optional[Any] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.set_attr('channel', channel, validator=is_integer, min_val=1)
        self.set_attr(
            'epsilon',
            epsilon if epsilon is not None else self.epsilon,
            validator=is_number,
            lower_limit=0,
            upper_limit=1.0
        )
        self.set_attr('scale', scale if scale is not None else self.scale)
        self.set_attr('bias', bias if bias is not None else self.bias)

    def weight_shapes(self, **kwargs: Any) -> Dict[str, tuple]:
        c = self.channel
        if c is None:
            raise ValueError("channel must be set for weight_shapes")
        return dict(scale=(c,), bias=(c,))


class InstanceNorm1dOp(InstanceNormOp):
    op_id = 'instance_norm1d'
    ndim = 1


class InstanceNorm2dOp(InstanceNormOp):
    op_id = 'instance_norm2d'
    ndim = 2


class InstanceNorm3dOp(InstanceNormOp):
    op_id = 'instance_norm3d'
    ndim = 3


class LayerNormOp(UnaryOp):
    """Layer normalization."""

    op_id = 'layer_norm'
    attrs: Tuple[str, ...] = ('axis', 'epsilon', 'scale', 'bias')
    weights: Tuple[str, ...] = ('scale', 'bias')
    ndim: Optional[int] = None
    epsilon: float = 1e-5
    scale: int = 1
    bias: int = 0

    def __init__(
        self,
        *,
        axis: Optional[int] = None,
        epsilon: Optional[float] = None,
        scale: Optional[Any] = None,
        bias: Optional[Any] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.set_attr('axis', axis, validator=is_integer, min_val=-1)
        self.set_attr(
            'epsilon',
            epsilon if epsilon is not None else self.epsilon,
            validator=is_number,
            lower_limit=0,
            upper_limit=1.0
        )
        # LayerNorm 的 scale/bias 不写入 IR，由 export_weights 单独导出
        self.set_attr('scale', scale)
        self.set_attr('bias', bias)
