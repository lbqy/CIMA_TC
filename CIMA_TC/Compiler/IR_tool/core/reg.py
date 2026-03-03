"""
Generic registry framework with automatic subclass registration.

This module provides a clean and production-safe implementation of
a plugin-style registry system using a metaclass.
"""

from typing import (
    Any,
    Type,
    TypeVar,
    ClassVar,
    Optional,
    Dict,
    Iterator,
    Tuple,
    Mapping,
    Union,
)
from abc import ABCMeta


# ============================================================
# Type Variables
# ============================================================

T = TypeVar("T", bound="RegistryMixin")
R = TypeVar("R", bound="RegistryEntry")


# ============================================================
# Base Entry
# ============================================================

class RegistryEntry:
    """
    Base class for all registry entries.

    Subclasses may override `validate()` to implement custom
    validation logic.
    """

    def validate(self) -> None:
        """
        Validate the instance after creation.

        Raises:
            ValueError: If validation fails.
        """
        pass

    def __str__(self) -> str:
        return self.__class__.__name__


# ============================================================
# Metaclass
# ============================================================

class RegistryMeta(ABCMeta):
    """
    Metaclass responsible for automatic registration of subclasses.

    Behavior:
    - If a class defines `__registry_key__`, it becomes a root registry.
    - Concrete subclasses that define the registry key attribute
      are automatically registered.
    """

    def __init__(
        cls,
        name: str,
        bases: tuple,
        namespace: dict,
        **kwargs: Any,
    ) -> None:
        """
        Metaclass responsible for automatic registration of subclasses.

        Behavior:
        - If a class defines `__registry_key__`, it becomes a root registry.
        - Concrete subclasses that define the registry key attribute
        are automatically registered.
        - Classes marked with `__abstract__ = True` will NOT be registered.
        - Supports multi-level inheritance safely.
        """
        super().__init__(name, bases, namespace, **kwargs)

        # ----------------------------------------------------
        # 1. Root registry class
        # ----------------------------------------------------
        if "__registry_key__" in namespace:
            cls._registry_key_attribute: str = namespace["__registry_key__"]
            cls._default_key: Optional[str] = namespace.get("__registry_default__")
            cls._registry: Dict[str, Type[RegistryEntry]] = {}
            return
           
        # ----------------------------------------------------
        # 2. Concrete plugin class
        # ----------------------------------------------------
        # Identify direct parent registry
        parent_registry = None
        # for base in bases:
        for base in cls.__mro__[1:]:
            if hasattr(base, "_registry") and hasattr(base, "_registry_key_attribute"):
                parent_registry = base
                break

        if parent_registry is None:
            return

        if namespace.get("__abstract__", False):
            return

        key_attr = parent_registry._registry_key_attribute

        # Plugin must explicitly define key attribute
        if key_attr not in namespace:
            return

        key_value = namespace[key_attr]

        if not isinstance(key_value, str) or not key_value:
            raise ValueError(
                f"Registry key '{key_attr}' must be a non-empty string"
            )

        normalized = key_value.lower()

        if normalized in parent_registry._registry:
            existing = parent_registry._registry[normalized]
            raise ValueError(
                f"Registration conflict for key '{normalized}': "
                f"{cls} conflicts with {existing}"
            )

        parent_registry._registry[normalized] = cls


# ============================================================
# Registry Mixin
# ============================================================

class RegistryMixin(metaclass=RegistryMeta):
    """
    Mixin providing registry lookup and factory functionality.

    Classes inheriting this must declare:

        __registry_key__ = "attribute_name"

    Optionally:

        __registry_default__ = "default_key"
    """

    _registry: ClassVar[Dict[str, Type[RegistryEntry]]]
    _registry_key_attribute: ClassVar[str]
    _default_key: ClassVar[Optional[str]]

    # --------------------------------------------------------
    # Lookup Methods
    # --------------------------------------------------------

    @classmethod
    def lookup(cls, key: Optional[str]) -> Optional[Type[RegistryEntry]]:
        """
        Look up a registered class by key.

        Args:
            key: Registry key. If None, default key is used.

        Returns:
            Registered class or None if not found.
        """
        if key is None:
            key = cls._default_key

        if key is None:
            return None

        return cls._registry.get(key.lower())

    @classmethod
    def get(cls, key: str) -> Type[RegistryEntry]:
        """
        Retrieve a registered class.

        Args:
            key: Registry key.

        Returns:
            Registered class.

        Raises:
            KeyError: If key is not registered.
        """
        result = cls.lookup(key)
        if result is None:
            raise KeyError(f"No {cls.__name__} registered with key: {key}")
        return result

    @classmethod
    def all_registered(cls) -> Iterator[Tuple[str, Type[RegistryEntry]]]:
        """
        Iterate over all registered entries.

        Returns:
            Iterator of (key, class) tuples.
        """
        yield from cls._registry.items()

    @classmethod
    def is_registered(cls, key: str) -> bool:
        """
        Check if a key is registered.

        Args:
            key: Registry key.

        Returns:
            True if registered, otherwise False.
        """
        return key.lower() in cls._registry

    # --------------------------------------------------------
    # Factory Methods
    # --------------------------------------------------------

    @classmethod
    def create(
        cls: Type[T],
        source: Optional[Union[str, Mapping[str, Any], RegistryEntry]] = None,
        **kwargs: Any,
    ) -> Optional[R]:
        """
        Create an instance from various input types.

        Args:
            source:
                - None: returns None (unless kwargs provided)
                - str: registry key
                - Mapping: configuration dictionary
                - RegistryEntry: existing instance
            **kwargs: Additional constructor arguments

        Returns:
            Created instance or None.

        Raises:
            KeyError: If key not found.
            TypeError: If unsupported source type.
            ValueError: If validation fails.
        """

        if source is None and not kwargs:
            return None

        if isinstance(source, cls):
            if kwargs:
                raise ValueError("Cannot apply kwargs to existing instance")
            return source

        if isinstance(source, str):
            entry_cls = cls.get(source)
            kwargs[cls._registry_key_attribute] = source
            instance = entry_cls(**kwargs)

        elif isinstance(source, Mapping):
            key_attr = cls._registry_key_attribute
            key = source.get(key_attr)
            if not key:
                raise ValueError(
                    f"Mapping must contain key '{key_attr}'"
                )
            entry_cls = cls.get(key)
            merged = {**source, **kwargs}
            instance = entry_cls(**merged)

        elif source is None:
            return cls.create(kwargs)

        else:
            raise TypeError(
                f"Cannot create {cls.__name__} from {type(source).__name__}"
            )

        instance.validate()
        return instance


# ============================================================
# Example Usage (Safe Demo)
# ============================================================

# if __name__ == "__main__":

#     class Operation(RegistryMixin, RegistryEntry):
#         """
#         Example registry root class.
#         """

#         __registry_key__ = "op_code"
#         __registry_default__ = "add"

#         def __init__(self, op_code: str):
#             self.op_code = op_code

#         def execute(self, a: float, b: float) -> float:
#             raise NotImplementedError

#         def validate(self) -> None:
#             if not self.op_code:
#                 raise ValueError("Invalid op_code")


#     class Add(Operation):
#         op_code = "add"

#         def execute(self, a: float, b: float) -> float:
#             return a + b


#     class Sub(Operation):
#         op_code = "sub"

#         def execute(self, a: float, b: float) -> float:
#             return a - b


#     print("Registered operations:")
#     for key, cls_ in Operation.all_registered():
#         print(key, cls_)

#     op = Operation.create("add")
#     print(type(op), op)
#     print("3 + 4 =", op.execute(3, 4))