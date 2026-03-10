"""
Device tree and device registry for mapping ops to runtime devices.
Container (DeviceTree) and device types (BaseDevice, BaseRuntime) are separated:
- DeviceTree: only a container of nodes (DeviceTree or BaseDevice).
- BaseDevice: leaf device with kind, number, can_map; does not inherit DeviceTree.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple, Union

from .jsonable import Jsonable
from .reg import RegistryMixin, RegistryEntry
from .type_utils import (
    to_string_tokens,
    is_integer,
    ValidationError,
)
from .ref import NameSegment, get_ref, InvalidNameError, Ref


def _is_valid_name(name: str) -> bool:
    """Return True if name is a valid device name segment."""
    try:
        NameSegment.parse(name)
        return True
    except InvalidNameError:
        return False


def _node_from_value(val: Any) -> Union["DeviceTree", "BaseDevice"]:
    """
    Build a tree node from one value: DeviceTree (if dict with "devices"),
    otherwise BaseDevice via make_device.
    """
    if isinstance(val, DeviceTree):
        return val
    if isinstance(val, BaseDevice):
        return val
    if isinstance(val, Mapping) and "devices" in val:
        return DeviceTree(devices=val["devices"])
    dev = make_device(val)
    if dev is None:
        raise TypeError("cannot create device from None")
    return dev


# ============================================================
# BaseDevice (leaf): registry of device types, no tree
# ============================================================


class BaseDevice(Jsonable, RegistryMixin, RegistryEntry):
    """
    Base class for device entries; registry key is 'kind'.
    Subclasses are registered by their class attribute `kind`.
    Does not inherit DeviceTree: a BaseDevice is a leaf node in the tree.
    """

    __registry_key__: str = "kind"
    __registry_default__: Optional[str] = None
    __abstract__: bool = True

    kind: Optional[Union[str, Tuple[str, ...]]]
    number: Optional[int]

    def __init__(
        self,
        *,
        kind: Optional[Union[str, Tuple[str, ...]]] = None,
        number: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if kind is None:
            kind = self.kind
        self.set_attr(
            "kind",
            to_string_tokens(kind, keep_scalar=True),
            not_none=True,
        )
        if number is not None:
            self.set_attr("number", number, is_integer, min_val=0)
        else:
            self.number = number

    def can_map(self, op_id: str) -> bool:
        """Return True if this device can run the given op_id."""
        return False

    def validate(self) -> None:
        if not self.kind:
            raise ValidationError("kind cannot be empty")


class BaseRuntime(BaseDevice):
    """Runtime device with optional white/black list for op mapping."""

    kind: str = "runtime"
    white_list: Optional[Tuple[str, ...]] = None
    black_list: Tuple[str, ...] = ()

    def can_map(self, op_id: str) -> bool:
        if op_id in self.black_list:
            return False
        if self.white_list is not None:
            return op_id in self.white_list
        return True


def make_device(
    source: Optional[Union[str, Mapping[str, Any], BaseDevice]] = None,
    **kwargs: Any,
) -> Optional[BaseDevice]:
    """Create a device instance from a kind string, mapping, or existing instance."""
    return BaseDevice.create(source, **kwargs)


# ============================================================
# DeviceTree: container only, holds DeviceTree or BaseDevice
# ============================================================


class DeviceTree(Jsonable):
    """Container of named nodes; each node is either a DeviceTree or a BaseDevice."""

    devices: Optional[Dict[str, Union["DeviceTree", "BaseDevice"]]] = None

    def __init__(
        self,
        *,
        devices: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if devices is None:
            self.devices = None
            return
        self.devices = {}
        for k, v in devices.items():
            self.devices[k] = _node_from_value(v)

    def add_device(
        self,
        name: str,
        kind: Union[str, BaseDevice, "DeviceTree"],
        **kwargs: Any,
    ) -> None:
        if self.devices is None:
            self.devices = {}
        if not _is_valid_name(name):
            raise InvalidNameError(f"invalid device name={name!r}")
        if name in self.devices:
            raise ValueError(f"device name={name!r} already exists")
        if isinstance(kind, BaseDevice):
            self.devices[name] = kind.clone(**kwargs)
        elif isinstance(kind, DeviceTree):
            self.devices[name] = kind
        elif isinstance(kind, str):
            dev = make_device(kind, **kwargs)
            if dev is None:
                raise TypeError(f"cannot create device from kind={kind!r}")
            self.devices[name] = dev
        else:
            raise TypeError(f"invalid device kind={kind!r}")

    def get_device(
        self, ref: Union[str, Ref]
    ) -> Optional[Union["DeviceTree", BaseDevice]]:
        return get_ref(self, "devices", ref)

    def iter_devices(
        self,
        names: Optional[List[str]] = None,
        *,
        deep: bool = True,
    ) -> Iterator[Tuple[str, Union["DeviceTree", BaseDevice]]]:
        if names is None:
            names = []
        devices = self.devices or {}
        for name, dev in devices.items():
            names.append(name)
            yield ".".join(names), dev
            if deep and isinstance(dev, DeviceTree):
                yield from dev.iter_devices(names, deep=deep)
            names.pop()
