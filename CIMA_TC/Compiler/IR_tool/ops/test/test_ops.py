"""
Pytest for common ops: make_op and attribute setting.
Run from repo root: pytest CIMA_TC/Compiler/IR_tool/ops/test_ops.py -v
"""

import pytest

from ...core import make_op
from ...core.type_utils import ValidationError

# ============================================================
# make_op by string
# ============================================================

def test_make_op_relu_by_string():
    """make_op('relu') returns ReluOp instance."""
    op = make_op("relu")
    assert op.op_id == "relu"
    assert op is not None


def test_make_op_conv2d_by_string():
    """make_op('conv2d') returns Conv2dOp (requires in_channel, out_channel, kernel via dict)."""
    op = make_op("conv2d", in_channel=3, out_channel=16, kernel=[3, 3])
    assert op.op_id == "conv2d"
    assert op.in_channel == 3
    assert op.out_channel == 16
    assert op.kernel == (3, 3)


def test_make_op_matmul_by_string():
    """make_op('matmul') with channel args."""
    op = make_op("matmul", in_channel=64, out_channel=128)
    assert op.op_id == "matmul"
    assert op.in_channel == 64
    assert op.out_channel == 128


def test_make_op_sigmoid_by_string():
    op = make_op("sigmoid")
    assert op.op_id == "sigmoid"


def test_make_op_constant_by_string():
    op = make_op("constant", value=1.0)
    assert op.op_id == "constant"
    assert op.value == 1.0


def test_make_op_identity_by_string():
    op = make_op("identity")
    assert op.op_id == "identity"


def test_make_op_slice_by_string():
    op = make_op("slice")
    assert op.op_id == "slice"


def test_make_op_max_pool2d_by_string():
    op = make_op("max_pool2d", kernel=[2, 2], stride=[2, 2])
    assert op.op_id == "max_pool2d"
    assert op.kernel == (2, 2)
    assert op.stride == (2, 2)


def test_make_op_reduce_mean_by_string():
    op = make_op("reduce_mean", axes=[1, 2], keepdims=True)
    assert op.op_id == "reduce_mean"


# ============================================================
# make_op by dict and attribute setting
# ============================================================

def test_make_op_conv2d_by_dict_with_attrs():
    """Create conv2d from dict with in_channel, out_channel, kernel, bias."""
    op = make_op({
        "op_id": "conv2d",
        "in_channel": 3,
        "out_channel": 16,
        "kernel": [3, 3],
        "bias": True,
    })
    assert op.op_id == "conv2d"
    assert op.in_channel == 3
    assert op.out_channel == 16
    assert op.kernel == (3, 3)
    assert op.bias is True


def test_make_op_leaky_relu_alpha():
    """LeakyRelu with explicit alpha."""
    op = make_op("leaky_relu", alpha=0.2)
    assert op.op_id == "leaky_relu"
    assert op.alpha == 0.2


def test_make_op_softmax_axis():
    """Softmax with axis."""
    op = make_op("softmax", axis=-1)
    assert op.op_id == "softmax"
    assert op.axis == -1


def test_make_op_batch_norm_attrs():
    """BatchNorm2d with channel, epsilon."""
    op = make_op("batch_norm2d", channel=32, epsilon=1e-5)
    assert op.op_id == "batch_norm2d"
    assert op.channel == 32
    assert op.epsilon == 1e-5


def test_make_op_concat_axis_channel_pos():
    """Concat with axis and channel_pos."""
    op = make_op("concat", axis=1, channel_pos="first")
    assert op.op_id == "concat"
    assert op.axis == 1
    assert op.channel_pos == "first"


def test_make_op_flatten_start_dim():
    """Flatten with start_dim."""
    op = make_op("flatten", start_dim=1)
    assert op.op_id == "flatten"
    assert op.start_dim == 1


def test_make_op_split_axis_sections():
    """Split with axis and split (number or sections)."""
    op = make_op("split", axis=1, split=4)
    assert op.op_id == "split"
    assert op.axis == 1
    assert op.split == 4
    assert op.num_outputs == 4


def test_make_op_constant_value_required():
    """Constant op requires value; missing value raises."""
    make_op("constant", value=0)
    with pytest.raises((ValidationError, ValueError, TypeError)):
        make_op("constant")


def test_make_op_unknown_op_id_raises():
    """Unknown op_id raises."""
    with pytest.raises((KeyError, ValueError, TypeError)):
        make_op("unknown_op_xyz")


# ============================================================
# weight_shapes and get_attrs
# ============================================================

def test_conv2d_weight_shapes():
    """Conv2d weight_shapes returns weight and bias shapes."""
    op = make_op("conv2d", in_channel=3, out_channel=16, kernel=[3, 3], bias=True)
    shapes = op.weight_shapes(channel_last=False)
    assert "weight" in shapes
    assert shapes["weight"] == (16, 3, 3, 3)
    assert shapes["bias"] == (16,)


def test_matmul_weight_shapes():
    """MatMul weight_shapes."""
    op = make_op("matmul", in_channel=64, out_channel=128)
    shapes = op.weight_shapes(channel_last=False)
    assert shapes["weight"] == (128, 64)
    assert shapes["bias"] == (128,)


def test_op_get_attrs():
    """get_attrs returns declared attrs."""
    op = make_op("leaky_relu", alpha=0.1)
    attrs = op.get_attrs()
    assert "alpha" in attrs
    assert attrs["alpha"] == 0.1
