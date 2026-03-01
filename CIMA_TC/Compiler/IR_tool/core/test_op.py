"""
Comprehensive pytest suite for the operator registration module.

Covers:
- Root registry behavior
- Automatic subclass registration
- Intermediate abstract class behavior
- Factory creation (string / mapping / instance / kwargs)
- Validation
- Metadata handling
- Duplicate registration detection
- Registry isolation
- enum_op_ids correctness
"""

import pytest

from .op import (
    BaseOp,
    UnaryOp,
    BinaryOp,
    make_op,
    enum_op_ids,
)


# ============================================================
# Test Fixtures: Define Concrete Operators
# ============================================================

class Add(BinaryOp):
    op_id = "add"

    def execute(self, a, b):
        return a + b


class Sub(BinaryOp):
    op_id = "sub"

    def execute(self, a, b):
        return a - b


class Neg(UnaryOp):
    op_id = "neg"

    def execute(self, a):
        return -a


# ============================================================
# Registry Root Behavior
# ============================================================

def test_registry_initialized():
    assert hasattr(BaseOp, "_registry")
    assert isinstance(BaseOp._registry, dict)


def test_concrete_ops_registered():
    assert "add" in BaseOp._registry
    assert "sub" in BaseOp._registry
    assert "neg" in BaseOp._registry


def test_abstract_classes_not_registered():
    assert "unaryop" not in BaseOp._registry
    assert "binaryop" not in BaseOp._registry


# ============================================================
# enum_op_ids
# ============================================================

def test_enum_op_ids():
    ids = list(enum_op_ids())
    assert "add" in ids
    assert "sub" in ids
    assert "neg" in ids


# ============================================================
# Factory: String Input
# ============================================================

def test_make_op_from_string():
    op = make_op("add")
    assert isinstance(op, Add)
    assert op.execute(3, 4) == 7


def test_make_op_string_case_insensitive():
    op = make_op("ADD")
    assert isinstance(op, Add)


# ============================================================
# Factory: Mapping Input
# ============================================================

def test_make_op_from_mapping():
    op = make_op({"op_id": "sub"})
    assert isinstance(op, Sub)
    assert op.execute(5, 2) == 3


def test_make_op_mapping_with_extra_fields():
    class Mul(BinaryOp):
        op_id = "mul"

        def __init__(self, *, op_id=None, scale=1):
            super().__init__(op_id=op_id)
            self.scale = scale

        def execute(self, a, b):
            return (a * b) * self.scale

    op = make_op({"op_id": "mul", "scale": 10})
    assert op.execute(2, 3) == 60


def test_make_op_mapping_missing_key():
    with pytest.raises(ValueError):
        make_op({"wrong_key": "add"})


# ============================================================
# Factory: Existing Instance
# ============================================================

def test_make_op_existing_instance():
    original = Add()
    op = make_op(original)
    assert op is original


def test_make_op_existing_instance_with_kwargs_fails():
    original = Add()
    with pytest.raises(ValueError):
        make_op(original, op_id="add")


# ============================================================
# Factory: None Behavior
# ============================================================

def test_make_op_none_returns_none():
    assert make_op(None) is None


def test_make_op_none_with_kwargs():
    op = make_op(None, op_id="add")
    assert isinstance(op, Add)


# ============================================================
# Validation
# ============================================================

def test_validation_called():

    class BadOp(BaseOp):
        op_id = "bad"

        def validate(self):
            raise ValueError("Invalid op")

    with pytest.raises(ValueError):
        make_op("bad")


# ============================================================
# Metadata Tests
# ============================================================

def test_get_attrs():

    class AttrOp(UnaryOp):
        op_id = "attr_op"
        attrs = ("x", "y")

        def __init__(self, *, op_id=None, x=1, y=2):
            super().__init__(op_id=op_id)
            self.x = x
            self.y = y

    op = make_op({"op_id": "attr_op", "x": 10, "y": 20})
    attrs = op.get_attrs()

    assert attrs == {"x": 10, "y": 20}


def test_weight_shapes_default():
    op = make_op("add")
    assert op.weight_shapes() == {}


def test_weight_shapes_not_implemented():

    class WeightOp(BaseOp):
        op_id = "weight_op"
        weights = ("w",)

    op = make_op("weight_op")

    with pytest.raises(NotImplementedError):
        op.weight_shapes()


# ============================================================
# Duplicate Registration
# ============================================================

def test_duplicate_op_id_raises():
    with pytest.raises(ValueError):

        class DuplicateOp(BaseOp):
            op_id = "add"


# ============================================================
# Registry Isolation
# ============================================================

def test_registry_isolation():

    from .reg import RegistryMixin, RegistryEntry

    class AnotherRoot(RegistryMixin, RegistryEntry):
        __registry_key__ = "name"

    class Foo(AnotherRoot):
        name = "foo"

    assert "foo" in AnotherRoot._registry
    assert "foo" not in BaseOp._registry