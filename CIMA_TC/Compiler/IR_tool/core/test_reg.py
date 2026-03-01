"""
Comprehensive pytest suite for the registry framework.

Covers:
- Root registry initialization
- Automatic subclass registration
- Duplicate key detection
- Lookup behavior
- Factory creation (string / mapping / instance)
- Default key behavior
- Validation triggering
- Error handling
"""

import pytest

from .reg import (
    RegistryMixin,
    RegistryEntry,
)


# ============================================================
# Test Fixtures: Sample Registry
# ============================================================

class Operation(RegistryMixin, RegistryEntry):
    """
    Example registry root used for testing.
    """

    __registry_key__ = "op_code"
    __registry_default__ = "add"

    def __init__(self, op_code: str):
        self.op_code = op_code

    def execute(self, a: int, b: int) -> int:
        raise NotImplementedError

    def validate(self) -> None:
        if not self.op_code:
            raise ValueError("Invalid op_code")


class Add(Operation):
    op_code = "add"

    def execute(self, a: int, b: int) -> int:
        return a + b


class Sub(Operation):
    op_code = "sub"

    def execute(self, a: int, b: int) -> int:
        return a - b


# ============================================================
# Root Registry Tests
# ============================================================

def test_root_registry_initialized():
    """
    Ensure root class creates empty registry at definition time.
    """
    assert hasattr(Operation, "_registry")
    assert isinstance(Operation._registry, dict)


def test_plugins_auto_registered():
    """
    Ensure subclasses are automatically registered.
    """
    assert "add" in Operation._registry
    assert "sub" in Operation._registry
    assert Operation._registry["add"] is Add
    assert Operation._registry["sub"] is Sub


# ============================================================
# Lookup Tests
# ============================================================

def test_lookup_existing():
    """
    Lookup should return correct class.
    """
    assert Operation.lookup("add") is Add
    assert Operation.lookup("sub") is Sub


def test_lookup_case_insensitive():
    """
    Keys should be case-insensitive.
    """
    assert Operation.lookup("ADD") is Add


def test_lookup_missing():
    """
    Lookup of unknown key should return None.
    """
    assert Operation.lookup("unknown") is None


def test_get_existing():
    """
    get() should return class for valid key.
    """
    assert Operation.get("add") is Add


def test_get_missing_raises():
    """
    get() should raise KeyError for invalid key.
    """
    with pytest.raises(KeyError):
        Operation.get("invalid")


# ============================================================
# Factory Creation Tests
# ============================================================

def test_create_from_string():
    """
    create() with string should instantiate correct class.
    """
    op = Operation.create("add")
    assert isinstance(op, Add)
    assert op.execute(3, 4) == 7


def test_create_from_mapping():
    """
    create() with mapping should instantiate correctly.
    """
    op = Operation.create({"op_code": "sub"})
    assert isinstance(op, Sub)
    assert op.execute(5, 2) == 3


def test_create_from_existing_instance():
    """
    create() should return same instance if passed directly.
    """
    original = Add("add")
    instance = Operation.create(original)
    assert instance is original


def test_create_with_kwargs_override():
    """
    kwargs should override mapping values.
    """
    op = Operation.create({"op_code": "add"}, op_code="add")
    assert isinstance(op, Add)


def test_create_none_returns_none():
    """
    create(None) without kwargs should return None.
    """
    assert Operation.create(None) is None


# ============================================================
# Default Key Tests
# ============================================================

def test_default_key_lookup():
    """
    lookup(None) should use default key.
    """
    assert Operation.lookup(None) is Add


# ============================================================
# Validation Tests
# ============================================================

def test_validation_called():
    """
    validate() must be called after instantiation.
    """

    class BadOp(Operation):
        op_code = "bad"

        def validate(self):
            raise ValueError("Validation failed")

    with pytest.raises(ValueError):
        Operation.create("bad")


# ============================================================
# Duplicate Registration Tests
# ============================================================

def test_duplicate_key_raises():
    """
    Defining duplicate key should raise ValueError.
    """

    with pytest.raises(ValueError):

        class Duplicate(Operation):
            op_code = "add"


# ============================================================
# Multiple Registry Isolation Test
# ============================================================

def test_multiple_registries_are_isolated():
    """
    Separate registry roots must not share entries.
    """

    class Animal(RegistryMixin, RegistryEntry):
        __registry_key__ = "species"

        def __init__(self, species: str):
            self.species = species


    class Dog(Animal):
        species = "dog"


    assert "dog" in Animal._registry
    assert "dog" not in Operation._registry