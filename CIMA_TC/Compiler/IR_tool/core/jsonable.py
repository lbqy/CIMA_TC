"""
Optimized and robust JSON / YAML serialization utility module.

Design goals:
- Preserve original public interfaces
- Improve robustness and error visibility
- Remove unsafe or misleading internal complexity
- Maintain predictable, auditable behavior
"""

from typing import (
    Mapping, Iterable, Callable, Any, Union, Optional, TextIO,
    Dict, List, Set, Tuple, Type, TypeVar, Generator, Protocol,
    runtime_checkable
)
from datetime import datetime, date, time
from contextlib import contextmanager
from pathlib import Path
import json
import warnings
from enum import Enum
from decimal import Decimal
from uuid import UUID
from dataclasses import is_dataclass, asdict
from functools import wraps, lru_cache
import threading
import time as time_module
from collections.abc import Iterable as ABCIterable
import hashlib
from .type_utils import is_scalar
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    yaml = None
    YAML_AVAILABLE = False


T = TypeVar("T", bound="Jsonable")

# ----------------------------------------------------------------------
# Filter policy
# ----------------------------------------------------------------------

class FilterPolicy(Enum):
    """Serialization attribute filter policy."""
    ALL = "all"
    EXCLUDE_PRIVATE = "exclude_private"
    EXCLUDE_PROTECTED = "exclude_protected"
    EXCLUDE_UNDERSCORES = "exclude_underscores"


# ----------------------------------------------------------------------
# Serialization configuration
# ----------------------------------------------------------------------

class SerializationConfig:
    """
    Serialization configuration container.

    This object is immutable in practice and copied during recursion.
    """

    __slots__ = (
        "filter_policy",
        "exclude_none",
        "exclude_empty",
        "max_depth",
        "indent",
        "ensure_ascii",
        "sort_keys",
        "default_flow_style",
        "max_ids",
        "use_cache",
        "strict",
    )

    def __init__(
        self,
        filter_policy: Union[FilterPolicy, bool, Callable[[str, Any], bool]] = FilterPolicy.EXCLUDE_PRIVATE,
        exclude_none: bool = False,
        exclude_empty: bool = False,
        max_depth: Optional[int] = 1_000_000,
        indent: Optional[int] = 2,
        ensure_ascii: bool = False,
        sort_keys: bool = False,
        default_flow_style: Optional[bool] = None,
        max_ids: int = 1_000_000,
        use_cache: bool = False,  # preserved for compatibility, not used
        strict: bool = True
    ):
        self.filter_policy = filter_policy
        self.exclude_none = exclude_none
        self.exclude_empty = exclude_empty
        self.max_depth = max_depth
        self.indent = indent
        self.ensure_ascii = ensure_ascii
        self.sort_keys = sort_keys
        self.default_flow_style = default_flow_style
        self.max_ids = max_ids
        self.use_cache = use_cache
        self.strict = strict

    def copy(self, **overrides: Any) -> "SerializationConfig":
        """Create a shallow copy with optional overrides."""
        data = {slot: getattr(self, slot) for slot in self.__slots__}
        data.update(overrides)
        return SerializationConfig(**data)


# ----------------------------------------------------------------------
# Protocols
# ----------------------------------------------------------------------

@runtime_checkable
class JSONSerializable(Protocol):
    """Protocol for custom JSON encoding."""

    def __json_encode__(self) -> Any:
        ...


@runtime_checkable
class JSONDeserializable(Protocol):
    """Protocol for custom JSON decoding."""

    @classmethod
    def __json_decode__(cls, data: Dict[str, Any]) -> Any:
        ...


# ----------------------------------------------------------------------
# Filter helpers
# ----------------------------------------------------------------------

@lru_cache(maxsize=128)
def _get_filter_func(
    policy: Union[FilterPolicy, bool, Callable[[str, Any], bool]]
) -> Callable[[str, Any], bool]:
    """Resolve filter policy to a callable."""
    if isinstance(policy, FilterPolicy):
        if policy == FilterPolicy.ALL:
            return lambda k, v: True
        if policy == FilterPolicy.EXCLUDE_PRIVATE:
            return lambda k, v: not k.startswith("_")
        if policy == FilterPolicy.EXCLUDE_PROTECTED:
            return lambda k, v: not (k.startswith("_") and not k.endswith("_"))
        if policy == FilterPolicy.EXCLUDE_UNDERSCORES:
            return lambda k, v: "_" not in k
        raise ValueError(f"Unsupported filter policy: {policy}")

    if policy is True:
        return lambda k, v: not k.startswith("_")
    if policy is False or policy is None:
        return lambda k, v: True
    if callable(policy):
        return policy

    raise TypeError(f"Invalid filter policy type: {type(policy)}")


# ----------------------------------------------------------------------
# Circular reference detection
# ----------------------------------------------------------------------

@contextmanager
def circular_check(
    ids: Optional[Set[str]],
    obj: Any,
    max_ids: int,
) -> Generator[Set[str], None, None]:
    """
    Detect circular references using object identity.

    Raises ValueError if a cycle is detected.
    """
    if ids is None:
        ids = set()

    key = f"{id(obj)}:{type(obj).__name__}"

    if len(ids) > max_ids:
        warnings.warn("Circular reference tracking limit exceeded")
        yield ids
        return

    if key in ids:
        raise ValueError(f"Circular reference detected for {type(obj).__name__}")

    ids.add(key)
    try:
        yield ids
    finally:
        ids.remove(key)


# ----------------------------------------------------------------------
# Core serialization
# ----------------------------------------------------------------------

def to_json_obj(
    obj: Any,
    *,
    config: Optional[SerializationConfig] = None,
    _ids: Optional[Set[str]] = None,
    **kwargs,
) -> Any:
    """
    Convert a Python object into a JSON-serializable structure.
    """

    if config is None:
        config = SerializationConfig(**kwargs)
    elif kwargs:
        config = config.copy(**kwargs)

    if obj is None:
        return None

    if is_scalar(obj):
        return obj

    if isinstance(obj, (datetime, date, time)):
        return obj.isoformat()

    if isinstance(obj, Decimal):
        return float(obj) if not obj.is_nan() else None

    if isinstance(obj, UUID):
        return str(obj)

    if isinstance(obj, Enum):
        return obj.value

    if is_dataclass(obj) and not isinstance(obj, type):
        return to_json_obj(asdict(obj), config=config, _ids=_ids)

    if isinstance(obj, JSONSerializable):
        encoded = obj.__json_encode__()
        if encoded is obj:
            raise ValueError("__json_encode__ must not return self")
        return to_json_obj(encoded, config=config, _ids=_ids)

    if config.max_depth is not None and config.max_depth <= 0:
        raise ValueError("Maximum serialization depth exceeded")

    next_config = config.copy(
        max_depth=config.max_depth - 1 if config.max_depth is not None else None
    )

    with circular_check(_ids, obj, config.max_ids) as ids:
        result = _process_complex_object(obj, next_config, ids)

    if (
        config.strict                    
        and isinstance(result, dict)     
        and not result                    
        and not isinstance(obj, dict)     
        and not isinstance(obj, Jsonable)  
    ):
        raise TypeError(
            f"Object of type {type(obj).__name__} has no serializable fields"
        )
    return result


def _process_complex_object(
    obj: Any,
    config: SerializationConfig,
    ids: Set[str],
) -> Any:
    """Process non-scalar objects."""
    filter_func = _get_filter_func(config.filter_policy)

    if isinstance(obj, Mapping):
        return _process_mapping(obj, config, ids, filter_func)

    if isinstance(obj, ABCIterable) and not isinstance(obj, (str, bytes, bytearray)):
        return _process_iterable(obj, config, ids)

    if isinstance(obj, Jsonable):
        return obj.to_json_obj(config=config, _ids=ids)

    if hasattr(obj, "__dict__"):
        return _process_object_dict(obj, config, ids, filter_func)

    if hasattr(obj, "__slots__"):
        return _process_object_slots(obj, config, ids, filter_func)

    raise TypeError(f"Object of type {type(obj).__name__} is not serializable")


def _process_mapping(
    obj: Mapping,
    config: SerializationConfig,
    ids: Set[str],
    filter_func: Callable[[str, Any], bool],
) -> Dict[str, Any]:
    """Serialize mapping objects."""
    result: Dict[str, Any] = {}

    for key, value in obj.items():
        key_str = str(key)

        if not filter_func(key_str, value):
            continue

        if config.exclude_none and value is None:
            continue

        if config.exclude_empty and isinstance(value, (Mapping, ABCIterable)) and not value:
            continue

        result[key_str] = to_json_obj(value, config=config, _ids=ids)

    return result


def _process_iterable(
    obj: ABCIterable,
    config: SerializationConfig,
    ids: Set[str],
) -> List[Any]:
    """Serialize iterable objects."""
    result: List[Any] = []

    for item in obj:
        if config.exclude_none and item is None:
            continue
        result.append(to_json_obj(item, config=config, _ids=ids))

    return result


def _process_object_dict(
    obj: Any,
    config: SerializationConfig,
    ids: Set[str],
    filter_func: Callable[[str, Any], bool],
) -> Dict[str, Any]:
    """Serialize objects using __dict__."""
    result: Dict[str, Any] = {}

    for key, value in obj.__dict__.items():
        if not filter_func(key, value):
            continue
        if config.exclude_none and value is None:
            continue
        result[key] = to_json_obj(value, config=config, _ids=ids)

    return result


def _process_object_slots(
    obj: Any,
    config: SerializationConfig,
    ids: Set[str],
    filter_func: Callable[[str, Any], bool],
) -> Dict[str, Any]:
    """Serialize objects using __slots__."""
    result: Dict[str, Any] = {}

    for slot in obj.__slots__:
        if not hasattr(obj, slot):
            continue
        value = getattr(obj, slot)
        if not filter_func(slot, value):
            continue
        result[slot] = to_json_obj(value, config=config, _ids=ids)

    return result


# ----------------------------------------------------------------------
# JSON Encoder
# ----------------------------------------------------------------------

class SafeJSONEncoder(json.JSONEncoder):
    """JSONEncoder delegating to to_json_obj."""

    def __init__(self, config: SerializationConfig, **kwargs):
        self.config = config
        super().__init__(**kwargs)

    def default(self, obj: Any) -> Any:
        return to_json_obj(obj, config=self.config)


# ----------------------------------------------------------------------
# Jsonable base class
# ----------------------------------------------------------------------

class Jsonable:
    """Base class for JSON-serializable objects."""

    _lock = threading.RLock()

    def to_json_obj(
        self,
        *,
        config: Optional[SerializationConfig] = None,
        _ids: Optional[Set[str]] = None,
        **kwargs,
    ) -> Any:
        if config is None:
            config = SerializationConfig(**kwargs)

        with self._lock:
            return to_json_obj(self.__dict__, config=config, _ids=_ids)

    @classmethod
    def from_json_obj(cls: Type[T], obj: Any, **kwargs) -> Optional[T]:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return cls(**obj)
        raise TypeError(f"Cannot deserialize {type(obj)} into {cls.__name__}")

    def clone(self: T, **overrides) -> T:
        data = self.to_json_obj()
        if isinstance(data, dict):
            data.update(overrides)
        return self.__class__.from_json_obj(data)


# ----------------------------------------------------------------------
# Dump / Load
# ----------------------------------------------------------------------

def dump_json(
    obj: Any,
    *,
    file: Optional[Union[str, Path, TextIO]] = None,
    config: Optional[SerializationConfig] = None,
    encoding: str = "utf-8",
    **kwargs,
) -> Optional[str]:
    """Serialize object to JSON or YAML."""

    if config is None:
        config = SerializationConfig(**kwargs)
    elif kwargs:
        config = config.copy(**kwargs)

    json_obj = to_json_obj(obj, config=config)

    if file is None:
        return _serialize_to_string(json_obj, config)

    _serialize_to_file(json_obj, file, config, encoding)
    return None


def load_json(
    data: Optional[str] = None,
    *,
    file: Optional[Union[str, Path, TextIO]] = None,
    encoding: str = "utf-8",
    cls: Optional[Type[Any]] = None,
) -> Any:
    """
    Load JSON or YAML data.

    If cls is provided, attempt to deserialize into the given class.
    """

    if data is None and file is None:
        raise ValueError("Either data or file must be provided")

    if file is not None:
        if isinstance(file, (str, Path)):
            data = Path(file).read_text(encoding=encoding)
        else:
            data = file.read()

    try:
        obj = json.loads(data)
    except json.JSONDecodeError:
        if YAML_AVAILABLE:
            obj = yaml.safe_load(data)
        else:
            raise

    if cls is None:
        return obj

    if isinstance(obj, dict):
        if hasattr(cls, "__json_decode__"):
            return cls.__json_decode__(obj)

        if issubclass(cls, Jsonable):
            return cls.from_json_obj(obj)

    raise TypeError(f"Cannot deserialize object into {cls}")



def _serialize_to_string(obj: Any, config: SerializationConfig) -> str:
    """Serialize object to string."""
    if YAML_AVAILABLE and config.default_flow_style is not None:
        return yaml.safe_dump(
            obj,
            sort_keys=config.sort_keys,
            allow_unicode=not config.ensure_ascii,
            default_flow_style=config.default_flow_style,
        )

    class _BoundJSONEncoder(SafeJSONEncoder):
        def __init__(self, **kwargs):
            super().__init__(config=config, **kwargs)

    return json.dumps(
        obj,
        indent=config.indent,
        ensure_ascii=config.ensure_ascii,
        sort_keys=config.sort_keys,
        cls=_BoundJSONEncoder,
        separators=(",", ":") if config.indent is None else None,
    )


def _serialize_to_file(
    obj: Any,
    file: Union[str, Path, TextIO],
    config: SerializationConfig,
    encoding: str,
) -> None:
    """Serialize object to file or stream."""
    if isinstance(file, (str, Path)):
        with open(file, "w", encoding=encoding) as f:
            f.write(_serialize_to_string(obj, config))
    else:
        file.write(_serialize_to_string(obj, config))

if __name__ == "__main__":
    from enum import Enum
    from datetime import date

    # ------------------------------------------------------------
    # Example domain models
    # ------------------------------------------------------------

    class Role(Enum):
        ENGINEER = "engineer"
        MANAGER = "manager"
        DESIGNER = "designer"

    class Address(Jsonable):
        def __init__(self, city: str, country: str, zipcode: str):
            self.city = city
            self.country = country
            self.zipcode = zipcode

    class Person(Jsonable):
        def __init__(
            self,
            name: str,
            age: int,
            role: Role,
            address: Address,
        ):
            self.name = name
            self.age = age
            self.role = role
            self.address = address

            # Private field to test filter policy
            self._internal_id = f"person:{name.lower()}"

    class Company(Jsonable):
        def __init__(
            self,
            name: str,
            founded: date,
            employees: list[Person],
            metadata: dict[str, Any],
        ):
            self.name = name
            self.founded = founded
            self.employees = employees
            self.metadata = metadata

    # ------------------------------------------------------------
    # Build example objects
    # ------------------------------------------------------------

    address_1 = Address("Tokyo", "Japan", "100-0001")
    address_2 = Address("Berlin", "Germany", "10115")

    alice = Person(
        name="Alice",
        age=32,
        role=Role.ENGINEER,
        address=address_1,
    )

    bob = Person(
        name="Bob",
        age=45,
        role=Role.MANAGER,
        address=address_2,
    )

    company = Company(
        name="ExampleTech",
        founded=date(2015, 6, 1),
        employees=[alice, bob],
        metadata={
            "public": True,
            "employee_count": 2,
            "offices": ["Tokyo", "Berlin"],
        },
    )

    # ------------------------------------------------------------
    # Serialization configuration
    # ------------------------------------------------------------

    config = SerializationConfig(
        filter_policy=FilterPolicy.EXCLUDE_PRIVATE,
        exclude_none=True,
        sort_keys=True,
        indent=2,
    )

    # ------------------------------------------------------------
    # Serialize to JSON string
    # ------------------------------------------------------------

    json_text = dump_json(company, config=config)
    print("=== Serialized JSON ===")
    print(json_text)

    # ------------------------------------------------------------
    # Deserialize back to Python object (raw dict)
    # ------------------------------------------------------------

    loaded_data = load_json(json_text)
    print("\n=== Loaded Data (dict) ===")
    print(loaded_data)

    # ------------------------------------------------------------
    # Clone example
    # ------------------------------------------------------------

    cloned_alice = alice.clone(age=33)
    print("\n=== Cloned Person ===")
    print(dump_json(cloned_alice, config=config))
