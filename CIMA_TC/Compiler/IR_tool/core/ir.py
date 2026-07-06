"""
Unified IR type combining computation graph (layers) and device tree (devices).
BaseIR can be loaded/saved from JSON and holds both the layer graph and device hierarchy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Optional, Type, TypeVar, Union

from .jsonable import Jsonable, SerializationConfig, load_json, dump_json, to_json_obj as _to_json_obj
from .type_utils import to_typed_object
from .layer import GraphLayer
from .device import DeviceTree


T = TypeVar("T", bound="BaseIR")


class BaseIR(GraphLayer, DeviceTree, Jsonable):
    """
    Root IR type: a computation graph (GraphLayer) plus a device tree (DeviceTree).

    - layers: computation graph (inputs, ops, outputs)
    - devices: device hierarchy (chip/core/PE or similar)
    - ir_version: version identifier for the IR format
    """

    ir_version: ClassVar[str] = "model_ir"

    @classmethod
    def load_ir(
        cls: Type[T],
        data: Optional[Union[str, bytes, dict, Any]] = None,
        *,
        file: Optional[Any] = None,
        **kwargs: Any,
    ) -> T:
        """
        Load IR from JSON data or file.

        Args:
            data: JSON string, bytes, or already-parsed dict. If None and file is None,
                returns an empty instance.
            file: Path or file-like object to load from. Mutually exclusive with data.
            **kwargs: Passed through to constructor when building from dict.

        Returns:
            Instance of cls (BaseIR or subclass).
        """
        if file is not None:
            if data is not None:
                raise ValueError("Cannot specify both data and file")
            return cls.load_ir(load_json(file=file), **kwargs)
        if data is None:
            return cls(**kwargs)
        if isinstance(data, (str, bytes)):
            return cls.load_ir(load_json(data), **kwargs)
        return to_typed_object(data, cls)

    @classmethod
    def make_ir(cls: Type[T], **kwargs: Any) -> T:
        """Build an IR instance from keyword arguments (layers, devices, ir_version, etc.)."""
        return cls(**kwargs)

    def __init__(
        self,
        *,
        ir_version: Optional[str] = None,
        layers: Optional[dict[str, Any]] = None,
        devices: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        # NOTE:
        # BaseIR uses multiple inheritance (GraphLayer + DeviceTree). Their __init__
        # chains do not line up via cooperative super() because GraphLayer->BaseLayer
        # does not route into DeviceTree. We therefore initialize both explicitly.
        #
        # Jsonable.__init__ is idempotent (no required constructor args), so calling
        # it multiple times via these initializers is safe.
        Jsonable.__init__(self, **kwargs)
        GraphLayer.__init__(self, layers=layers, **kwargs)
        DeviceTree.__init__(self, devices=devices, **kwargs)
        if ir_version is None:
            ir_version = self.ir_version
        self.set_attr("ir_version", ir_version, not_none=True)

        # Runtime-only parameter stores (NOT serialized by dump_json/save_ir).
        # Frontend can attach real weights / BN params here for later mapping passes.
        self.weight_store: dict[str, Any] = {}
        self.bn_store: dict[str, Any] = {}

    # Jsonable override: exclude runtime-only stores from serialization
    def to_json_obj(self, config: Any = None, _ids: Any = None) -> Any:  # type: ignore[override]
        """
        Serialize BaseIR while excluding runtime-only fields such as weight_store/bn_store.
        """
        data = {k: v for k, v in self.__dict__.items() if k not in ("weight_store", "bn_store")}
        return _to_json_obj(data, config=config, _ids=_ids)


def save_ir(ir: BaseIR, *, file: Optional[Any] = None, **kwargs: Any) -> Optional[str]:
    """
    Serialize IR to JSON/YAML string or file.

    Args:
        ir: BaseIR instance to serialize.
        file: If provided, write to this path or file-like object; otherwise return string.
        **kwargs: Passed to dump_json (e.g. config, encoding).

    Returns:
        Serialized string if file is None, else None.
    """
    if "config" not in kwargs and _is_yaml_file(file):
        kwargs["config"] = SerializationConfig(default_flow_style=False)
    return dump_json(ir, file=file, **kwargs)


def _is_yaml_file(file: Optional[Any]) -> bool:
    if not isinstance(file, (str, Path)):
        return False
    return Path(file).suffix.lower() in {".yaml", ".yml"}


# Backward compatibility: save_ir as alias for dump_json when used as (obj, file=...)
# Prefer save_ir(ir, file=path) for clarity.
load_ir = BaseIR.load_ir
make_ir = BaseIR.make_ir
