import json
import pytest
from pathlib import Path
from datetime import datetime, date
from enum import Enum
from decimal import Decimal
from uuid import uuid4
from dataclasses import dataclass

from ..jsonable import (
    to_json_obj,
    dump_json,
    load_json,
    SerializationConfig,
    FilterPolicy,
    Jsonable,
    JSONSerializable,
    JSONDeserializable,
)


# ============================================================
# Test domain models
# ============================================================

class Role(Enum):
    ENGINEER = "engineer"
    MANAGER = "manager"


class Address(Jsonable):
    def __init__(self, city: str, country: str):
        self.city = city
        self.country = country


class Person(Jsonable):
    def __init__(self, name: str, age: int, role: Role, address: Address):
        self.name = name
        self.age = age
        self.role = role
        self.address = address
        self._private = "secret"


# ============================================================
# to_json_obj: scalars and builtins
# ============================================================

@pytest.mark.parametrize(
    "value",
    [1, 1.5, "x", True, False, None],
)
def test_scalar_roundtrip(value):
    assert to_json_obj(value) == value


def test_datetime_and_date():
    assert to_json_obj(date(2024, 1, 1)) == "2024-01-01"
    assert to_json_obj(datetime(2024, 1, 1, 12, 0)) == "2024-01-01T12:00:00"


def test_enum_decimal_uuid():
    assert to_json_obj(Role.ENGINEER) == "engineer"
    assert to_json_obj(Decimal("2.5")) == 2.5
    u = uuid4()
    assert to_json_obj(u) == str(u)


# ============================================================
# Containers
# ============================================================

def test_list_tuple_dict():
    obj = {"a": [1, (2, 3)]}
    assert to_json_obj(obj) == {"a": [1, [2, 3]]}


# ============================================================
# Jsonable
# ============================================================

def test_simple_jsonable():
    addr = Address("Tokyo", "Japan")
    assert to_json_obj(addr) == {"city": "Tokyo", "country": "Japan"}


def test_nested_jsonable_with_filter():
    addr = Address("Berlin", "Germany")
    p = Person("Alice", 30, Role.ENGINEER, addr)

    cfg = SerializationConfig(filter_policy=FilterPolicy.EXCLUDE_PRIVATE)
    data = to_json_obj(p, config=cfg)

    assert "name" in data
    assert "_private" not in data
    assert data["address"]["city"] == "Berlin"


# ============================================================
# Dataclass
# ============================================================

@dataclass
class Point:
    x: int
    y: int


def test_dataclass():
    assert to_json_obj(Point(1, 2)) == {"x": 1, "y": 2}


# ============================================================
# Depth & circular
# ============================================================

def test_max_depth():
    obj = {"a": {"b": {"c": 1}}}
    cfg = SerializationConfig(max_depth=1)

    with pytest.raises(ValueError):
        to_json_obj(obj, config=cfg)


def test_self_circular():
    a = {}
    a["self"] = a

    with pytest.raises(ValueError):
        to_json_obj(a)


def test_mutual_circular():
    a = {}
    b = {"a": a}
    a["b"] = b

    with pytest.raises(ValueError):
        to_json_obj(a)


# ============================================================
# Unsupported type
# ============================================================

def test_unsupported_type():
    class X:
        pass

    with pytest.raises(TypeError):
        to_json_obj(X())


# ============================================================
# dump_json
# ============================================================

def test_dump_returns_string():
    text = dump_json({"a": 1})
    assert isinstance(text, str)
    assert json.loads(text) == {"a": 1}


def test_dump_to_file(tmp_path: Path):
    path = tmp_path / "x.json"
    assert dump_json({"x": 1}, file=path) is None
    assert json.loads(path.read_text()) == {"x": 1}


# ============================================================
# load_json
# ============================================================

def test_load_from_string():
    assert load_json('{"a": 1}') == {"a": 1}


def test_load_from_file(tmp_path: Path):
    path = tmp_path / "y.json"
    path.write_text('{"x": 2}')
    assert load_json(file=path) == {"x": 2}


def test_load_missing_input():
    with pytest.raises(ValueError):
        load_json()


# ============================================================
# Jsonable.from_json_obj
# ============================================================

def test_from_json_obj():
    obj = {"city": "Rome", "country": "Italy"}
    addr = Address.from_json_obj(obj)

    assert isinstance(addr, Address)
    assert addr.city == "Rome"


def test_load_json_with_cls_jsonable():
    data = {
        "name": "Bob",
        "age": 40,
        "role": "manager",
        "address": {"city": "Paris", "country": "France"},
    }

    class PersonV2(Person):
        @classmethod
        def from_json_obj(cls, obj):
            obj["role"] = Role(obj["role"])
            obj["address"] = Address.from_json_obj(obj["address"])
            return cls(**obj)

    text = json.dumps(data)
    person = load_json(text, cls=PersonV2)

    assert isinstance(person, Person)
    assert person.role is Role.MANAGER
    assert person.address.city == "Paris"


# ============================================================
# JSONDeserializable protocol
# ============================================================

class Custom(JSONDeserializable):
    def __init__(self, x):
        self.x = x

    @classmethod
    def __json_decode__(cls, data):
        return cls(data["x"] * 2)


def test_json_deserializable():
    text = '{"x": 5}'
    obj = load_json(text, cls=Custom)
    assert isinstance(obj, Custom)
    assert obj.x == 10


# ============================================================
# Full round-trip
# ============================================================

def test_round_trip():
    addr = Address("Oslo", "Norway")
    p = Person("Clara", 28, Role.ENGINEER, addr)

    cfg = SerializationConfig(filter_policy=FilterPolicy.EXCLUDE_PRIVATE)
    text = dump_json(p, config=cfg)
    data = load_json(text)

    assert data["name"] == "Clara"
    assert data["address"]["country"] == "Norway"


# ============================================================
# SerializationConfig
# ============================================================

def test_config_copy():
    """SerializationConfig.copy returns new config with overrides."""
    cfg = SerializationConfig(exclude_none=True, max_depth=10)
    cfg2 = cfg.copy(max_depth=5)
    assert cfg2.max_depth == 5
    assert cfg2.exclude_none is True
    assert cfg.max_depth == 10


def test_config_exclude_none():
    """exclude_none=True omits None values from dict serialization."""
    obj = {"a": 1, "b": None, "c": "x"}
    cfg = SerializationConfig(exclude_none=True)
    result = to_json_obj(obj, config=cfg)
    assert "a" in result and "c" in result
    assert "b" not in result


def test_config_include_none():
    """exclude_none=False keeps None values."""
    obj = {"a": 1, "b": None}
    cfg = SerializationConfig(exclude_none=False)
    result = to_json_obj(obj, config=cfg)
    assert result["b"] is None


def test_config_exclude_empty():
    """exclude_empty=True omits empty dict/list from serialization."""
    obj = {"a": [], "b": {}, "c": 1}
    cfg = SerializationConfig(exclude_empty=True)
    result = to_json_obj(obj, config=cfg)
    assert result.get("c") == 1
    assert "a" not in result
    assert "b" not in result


# ============================================================
# Jsonable.set_attr and clone
# ============================================================

def test_jsonable_set_attr():
    """set_attr sets attribute; validator can be used."""
    addr = Address("London", "UK")
    addr.set_attr("city", "Paris")
    assert addr.city == "Paris"


def test_jsonable_set_attr_not_none_raises():
    """set_attr with not_none=True raises when value is None."""
    from ..type_utils import ValidationError
    addr = Address("x", "y")
    with pytest.raises(ValidationError):
        addr.set_attr("city", None, not_none=True)


def test_jsonable_clone():
    """clone creates new instance with overrides applied to serialized dict."""
    addr = Address("Berlin", "Germany")
    addr2 = addr.clone(city="Munich")
    assert addr2.city == "Munich"
    assert addr2.country == "Germany"
    assert addr.city == "Berlin"
