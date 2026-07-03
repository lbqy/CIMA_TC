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

from ..op import (
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
    op_id = "test_add"

    def execute(self, a, b):
        return a + b


class Sub(BinaryOp):
    op_id = "test_sub"

    def execute(self, a, b):
        return a - b


class Neg(UnaryOp):
    op_id = "test_neg"

    def execute(self, a):
        return -a

# ============================================================
# Registry Root Behavior
# ============================================================

def test_registry_initialized():
    assert hasattr(BaseOp, "_registry")
    assert isinstance(BaseOp._registry, dict)


def test_concrete_ops_registered():
    assert "test_add" in BaseOp._registry
    assert "test_sub" in BaseOp._registry
    assert "test_neg" in BaseOp._registry

def test_abstract_classes_not_registered():
    assert "unaryop" not in BaseOp._registry
    assert "binaryop" not in BaseOp._registry


def test_registry_keys_lowercase():
    """Registered keys are stored in lowercase."""
    assert "test_add" in BaseOp._registry
    assert "TEST_ADD" not in BaseOp._registry
    assert BaseOp.lookup("TEST_ADD") is Add

# ============================================================
# enum_op_ids
# ============================================================

def test_enum_op_ids():
    ids = list(enum_op_ids())
    assert "test_add" in ids
    assert "test_sub" in ids
    assert "test_neg" in ids


def test_enum_op_ids_consistency_with_registry():
    """enum_op_ids yields exactly the registry keys."""
    ids = set(enum_op_ids())
    assert ids == set(BaseOp._registry.keys())


# ============================================================
# Factory: String Input
# ============================================================

def test_make_op_from_string():
    op = make_op("test_add")
    assert isinstance(op, Add)
    assert op.execute(3, 4) == 7


def test_make_op_string_case_insensitive():
    op = make_op("TEST_ADD")
    assert isinstance(op, Add)


# ============================================================
# Factory: Mapping Input
# ============================================================

def test_make_op_from_mapping():
    op = make_op({"op_id": "test_sub"})
    assert isinstance(op, Sub)
    assert op.execute(5, 2) == 3


def test_make_op_mapping_with_extra_fields():
    class Mul(BinaryOp):
        op_id = "test_mul"

        def __init__(self, *, op_id=None, scale=1):
            super().__init__(op_id=op_id)
            self.scale = scale

        def execute(self, a, b):
            return (a * b) * self.scale

    op = make_op({"op_id": "test_mul", "scale": 10})
    assert op.execute(2, 3) == 60


def test_make_op_mapping_missing_key():
    with pytest.raises(ValueError):
        make_op({"wrong_key": "test_add"})


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
        make_op(original, op_id="test_add")


# ============================================================
# Factory: None Behavior
# ============================================================

def test_make_op_none_returns_none():
    assert make_op(None) is None


def test_make_op_none_with_kwargs():
    op = make_op(None, op_id="test_add")
    assert isinstance(op, Add)


def test_make_op_none_no_kwargs_returns_none():
    """make_op(None) with no kwargs returns None (reg.create contract)."""
    assert make_op(None) is None


# ============================================================
# Validation
# ============================================================

def test_validation_called():

    class BadOp(BaseOp):
        op_id = "test_bad"

        def validate(self):
            raise ValueError("Invalid op")

    with pytest.raises(ValueError):
        make_op("test_bad")


def test_make_op_invalid_key_raises():
    """make_op with unregistered key raises KeyError."""
    with pytest.raises(KeyError):
        make_op("_nonexistent_op_xyz_")


# ============================================================
# num_inputs
# ============================================================

def test_num_inputs_unary():
    """UnaryOp has num_inputs = 1."""
    op = make_op("test_neg")
    assert op.num_inputs == 1


def test_num_inputs_binary():
    """BinaryOp has num_inputs = 2."""
    op = make_op("test_add")
    assert op.num_inputs == 2


# ============================================================
# Metadata Tests
# ============================================================

def test_get_attrs():

    class AttrOp(UnaryOp):
        op_id = "test_attr_op"
        attrs = ("x", "y")

        def __init__(self, *, op_id=None, x=1, y=2):
            super().__init__(op_id=op_id)
            self.x = x
            self.y = y

    op = make_op({"op_id": "test_attr_op", "x": 10, "y": 20})
    attrs = op.get_attrs()

    assert attrs == {"x": 10, "y": 20}


def test_weight_shapes_default():
    op = make_op("test_add")
    assert op.weight_shapes() == {}


def test_weight_shapes_not_implemented():

    class WeightOp(BaseOp):
        op_id = "test_weight_op"
        weights = ("w",)

    op = make_op("test_weight_op")

    with pytest.raises(NotImplementedError):
        op.weight_shapes()


# ============================================================
# Duplicate Registration
# ============================================================

def test_duplicate_op_id_raises():
    with pytest.raises(ValueError):

        class DuplicateOp(BaseOp):
            op_id = "test_add"


# ============================================================
# Registry Isolation
# ============================================================

def test_registry_isolation():

    from ..reg import RegistryMixin, RegistryEntry

    class AnotherRoot(RegistryMixin, RegistryEntry):
        __registry_key__ = "name"

    class Foo(AnotherRoot):
        name = "foo"

    assert "foo" in AnotherRoot._registry
    assert "foo" not in BaseOp._registry


def test_get_attrs_empty():
    """get_attrs returns empty dict when attrs is empty."""
    op = make_op("test_add")
    assert op.get_attrs() == {}


def test_make_op_from_dict_merge_kwargs():
    """Extra kwargs merge with dict source."""
    op = make_op({"op_id": "test_add"}, op_id="test_add")
    assert isinstance(op, Add)