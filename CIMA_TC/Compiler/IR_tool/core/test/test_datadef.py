"""
Comprehensive test suite for DataDef.

This test file validates:

1. Ref handling (string + Ref object)
2. Attribute validation
3. JSON serialization behavior
4. Shape construction logic
5. Shape history tracking
6. Error handling paths
"""

import pytest

from ..datadef import DataDef
from ..ref import Ref


# ============================================================
# Ref Handling
# ============================================================


def test_ref_string_auto_parse():
    d = DataDef(ref="encoder.layer:2")

    assert isinstance(d.ref, Ref)
    assert str(d.ref) == "encoder.layer:2"


def test_ref_object_input():
    r = Ref.parse("encoder.layer:1")
    d = DataDef(ref=r)

    assert d.ref is r


def test_ref_invalid_type():
    with pytest.raises(TypeError):
        DataDef(ref=123)


def test_set_ref_string():
    d = DataDef()
    d.set_ref("a.b:1")

    assert isinstance(d.ref, Ref)
    assert str(d.ref) == "a.b:1"


def test_set_ref_none():
    d = DataDef(ref="a.b")
    d.set_ref(None)

    assert d.ref is None


def test_get_ref_segments():
    d = DataDef(ref="a.b:2")

    segments = d.get_ref_segments()

    assert len(segments) == 2
    assert segments[1].index == 2


# ============================================================
# JSON Serialization
# ============================================================


def test_json_only_ref():
    """to_json_obj returns dict with ref when ref is set (other attrs may be None)."""
    d = DataDef(ref="a.b:1")

    json_obj = d.to_json_obj()

    assert isinstance(json_obj, dict)
    assert json_obj.get("ref") == "a.b:1"


def test_json_full_object():
    d = DataDef(ref="a.b:1", channel=3)

    json_obj = d.to_json_obj()

    assert isinstance(json_obj, dict)
    assert json_obj["ref"] == "a.b:1"
    assert json_obj["channel"] == 3


# ============================================================
# Basic Attribute Validation
# ============================================================


def test_batch_must_be_positive():
    with pytest.raises(Exception):
        DataDef(batch=0)


def test_channel_must_be_positive():
    with pytest.raises(Exception):
        DataDef(channel=-1)


def test_dims_conversion():
    d = DataDef(dims=3)

    assert d.dims == (3,)


def test_shape_conversion():
    d = DataDef(shape=[1, 2, 3])

    assert d.shape == (1, 2, 3)


# ============================================================
# make_shape Tests
# ============================================================


def test_make_shape_channel_first():
    d = DataDef(
        dims=(32, 32),
        channel=3,
        channel_last=False,
    )

    shape = d.make_shape()

    assert shape == (3, 32, 32)


def test_make_shape_channel_last():
    d = DataDef(
        dims=(32, 32),
        channel=3,
        channel_last=True,
    )

    shape = d.make_shape()

    assert shape == (32, 32, 3)


def test_make_shape_with_ndim_check():
    d = DataDef(
        dims=(32, 32),
        channel=3,
        ndim=2,
    )

    shape = d.make_shape()

    assert shape == (3, 32, 32)


def test_make_shape_ndim_mismatch():
    d = DataDef(
        dims=(32, 32),
        channel=3,
        ndim=3,
    )

    with pytest.raises(ValueError):
        d.make_shape()


def test_make_shape_infer_from_spatial():
    d = DataDef(
        height=64,
        width=64,
        channel=3,
    )

    shape = d.make_shape()

    assert shape == (3, 64, 64)


def test_make_shape_missing_dims():
    d = DataDef(channel=3)

    with pytest.raises(ValueError):
        d.make_shape()


def test_make_shape_missing_channel():
    d = DataDef(dims=(32, 32))

    with pytest.raises(ValueError):
        d.make_shape()


# ============================================================
# set_shape History Tracking
# ============================================================


def test_set_shape_initial():
    d = DataDef()

    d.set_shape((1, 2, 3))

    assert d.shape == (1, 2, 3)
    assert d.shapes is None


def test_set_shape_same_value():
    d = DataDef(shape=(1, 2, 3))

    d.set_shape((1, 2, 3))

    assert d.shapes is None


def test_set_shape_track_history():
    d = DataDef(shape=(1, 2, 3))

    d.set_shape((2, 3, 4))

    assert d.shapes == [(1, 2, 3)]
    assert d.shape == (2, 3, 4)


def test_set_shape_multiple_history():
    d = DataDef(shape=(1, 2, 3))

    d.set_shape((2, 3, 4))
    d.set_shape((3, 4, 5))

    assert d.shapes == [(1, 2, 3), (2, 3, 4)]
    assert d.shape == (3, 4, 5)


# ============================================================
# clone
# ============================================================

def test_datadef_clone():
    """clone copies ref and attributes; overrides can be passed."""
    d = DataDef(ref="a.b:1", channel=4, shape=(1, 4, 8))
    c = d.clone(channel=8)
    assert c.ref is not None and str(c.ref) == "a.b:1"
    assert c.channel == 8
    assert c.shape == (1, 4, 8)


# ============================================================
# JSON serialization with ref
# ============================================================

def test_json_ref_with_other_attr_is_dict():
    """When ref and other attrs are set, to_json_obj returns dict with ref as string."""
    d = DataDef(ref="a", channel=3)
    obj = d.to_json_obj()
    assert isinstance(obj, dict)
    assert obj["ref"] == "a"
    assert obj["channel"] == 3


# ============================================================
# Additional attribute validation
# ============================================================

def test_width_height_positive():
    """width and height must be >= 1 when set."""
    with pytest.raises(Exception):
        DataDef(width=0)
    with pytest.raises(Exception):
        DataDef(height=0)


def test_get_ref_segments_none():
    """get_ref_segments returns None when ref is None."""
    d = DataDef()
    assert d.get_ref_segments() is None


def test_datadef_dtype_passthrough():
    """dtype is stored without strict validation."""
    d = DataDef(ref="x", dtype="float32")
    assert d.dtype == "float32"