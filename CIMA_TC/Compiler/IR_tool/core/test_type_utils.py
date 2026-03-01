import pytest
import warnings

from .type_utils import (
    is_scalar,
    is_boolean,
    is_integer,
    is_number,
    is_integers,
    is_numbers,
    CollectionConstraints,
    validate_collection,
    is_in_values,
    to_boolean,
    to_string_tokens,
    to_integer_tuple,
    to_typed_object,
    to_typed_dict,
    to_typed_list,
    to_variable_token,
    ValidationError,
    ConversionError,
)


# =========================================================
# Basic scalar validators
# =========================================================

def test_is_scalar():
    assert is_scalar(None)
    assert is_scalar(1)
    assert is_scalar(1.0)
    assert is_scalar("a")
    assert is_scalar(b"a")
    assert not is_scalar([1])


def test_is_boolean():
    assert is_boolean(True)
    assert is_boolean(False)
    assert not is_boolean(1)


def test_is_integer_strict_and_range():
    assert is_integer(5)
    assert not is_integer(True)
    assert is_integer(True, strict=False)
    assert is_integer(5, min_val=1, max_val=10)
    assert not is_integer(0, min_val=1)
    assert not is_integer(11, max_val=10)


def test_is_number_constraints():
    assert is_number(3.5)
    assert not is_number(True)
    assert is_number(True, strict=False)
    assert is_number(5, min_val=1, max_val=10)
    assert not is_number(5, lower_limit=5)
    assert not is_number(5, upper_limit=5)


# =========================================================
# is_integers / is_numbers (new API)
# =========================================================

def test_is_integers_basic():
    assert is_integers([1, 2, 3])
    assert not is_integers([1, 2.0])
    assert not is_integers(["a"])


def test_is_integers_with_element_constraints():
    assert is_integers([1, 2, 3], min_val=0)
    assert not is_integers([1, -1], min_val=0)
    assert not is_integers([True], strict=True)
    assert is_integers([True], strict=False)


def test_is_integers_collection_constraints():
    assert is_integers([1, 2], min_size=1, max_size=3)
    assert not is_integers([], min_size=1)
    assert not is_integers([1, 2, 3, 4], max_size=3)
    assert is_integers([1, 2], exact_size=2)
    assert not is_integers([1], exact_size=2)


def test_is_integers_allow_scalar_and_none():
    assert is_integers(5, allow_scalar=True)
    assert not is_integers(5, allow_scalar=False)
    assert is_integers(None, allow_none=True)
    assert not is_integers(None, allow_none=False)


def test_is_numbers_basic():
    assert is_numbers([1, 2.0])
    assert not is_numbers(["a"])


def test_is_numbers_element_constraints():
    assert is_numbers([1, 2.5], min_val=0)
    assert not is_numbers([1, -1], min_val=0)
    assert not is_numbers([True], strict=True)
    assert is_numbers([True], strict=False)


# =========================================================
# validate_collection (low-level)
# =========================================================

def test_validate_collection_direct():
    constraints = CollectionConstraints(min_size=1, max_size=3)
    assert validate_collection([1, 2], is_integer, constraints)
    assert not validate_collection([], is_integer, constraints)


# =========================================================
# is_in_values
# =========================================================

def test_is_in_values():
    assert is_in_values(1, {1, 2})
    assert not is_in_values(3, {1, 2})
    assert is_in_values(None, {1, 2}, allow_none=True)


# =========================================================
# to_boolean
# =========================================================

def test_to_boolean():
    assert to_boolean(True)
    assert to_boolean("yes")
    assert not to_boolean("no")
    assert not to_boolean("")
    with pytest.raises(ValueError):
        to_boolean("maybe")


# =========================================================
# to_string_tokens
# =========================================================

def test_to_string_tokens():
    assert to_string_tokens(5) == ("5",)
    assert to_string_tokens([1, 2]) == ("1", "2")
    assert to_string_tokens(5, keep_scalar=True) == "5"

    with pytest.raises(ConversionError):
        to_string_tokens({"a": 1})


# =========================================================
# to_integer_tuple
# =========================================================

def test_to_integer_tuple_basic():
    assert to_integer_tuple(3) == (3,)
    assert to_integer_tuple(3, dimensions=3) == (3, 3, 3)
    assert to_integer_tuple([1, 2]) == (1, 2)
    assert to_integer_tuple([1, 2], dimensions=4) == (1, 2, 1, 2)


def test_to_integer_tuple_invalid_expand():
    with pytest.raises(ConversionError):
        to_integer_tuple([1, 2], dimensions=3)


def test_to_integer_tuple_float_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = to_integer_tuple([1.7])
        assert result == (1,)
        assert len(w) == 1


def test_to_integer_tuple_invalid_element():
    with pytest.raises(ConversionError):
        to_integer_tuple(["a"])


# =========================================================
# to_typed_object / dict / list
# =========================================================

class Dummy:
    def __init__(self, x):
        self.x = x


def test_to_typed_object_scalar():
    obj = to_typed_object(5, Dummy)
    assert isinstance(obj, Dummy)
    assert obj.x == 5


def test_to_typed_object_mapping():
    obj = to_typed_object({"x": 10}, Dummy)
    assert obj.x == 10


def test_to_typed_object_sequence():
    obj = to_typed_object([20], Dummy)
    assert obj.x == 20


def test_to_typed_object_invalid_type():
    class StrictDummy:
        def __init__(self, x: int):
            if not isinstance(x, int):
                raise ValueError("x must be int")
            self.x = x

    with pytest.raises(ValidationError):
        to_typed_object("bad", StrictDummy)

def test_to_typed_object_conversion_error():
    class DummyNoArgs:
        def __init__(self):
            pass

    with pytest.raises(ConversionError):
        to_typed_object([1, 2], DummyNoArgs)
        
def test_to_typed_dict():
    result = to_typed_dict({"a": 1}, Dummy)
    assert isinstance(result["a"], Dummy)


def test_to_typed_list():
    result = to_typed_list([1, 2], Dummy)
    assert isinstance(result[0], Dummy)


# =========================================================
# to_variable_token
# =========================================================

def test_to_variable_token():
    assert to_variable_token("Hello World!") == "hello_world_"
    assert to_variable_token("123abc") == "_123abc"
    assert to_variable_token(None, allow_none=True) is None
    assert to_variable_token(None) == ""