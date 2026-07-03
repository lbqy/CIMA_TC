from __future__ import annotations

from typing import Any, List, Optional, Tuple

from .jsonable import Jsonable
from .ref import Ref
from .type_utils import is_boolean, is_integer, is_integers, to_integer_tuple


class DataDef(Jsonable):
    """
    Tensor metadata used by IR inputs, outputs, and weights.

    DataDef intentionally stores only lightweight structural information:
    an optional producer reference, shape/channel metadata, dtype, and a small
    shape-history list used by rewrite passes.
    """

    ref: Optional[Ref]
    batch: Optional[int]
    channel: Optional[int]
    dims: Optional[Tuple[int, ...]]
    dtype: Optional[str]
    channel_last: Optional[bool]
    width: Optional[int]
    height: Optional[int]
    depth: Optional[int]
    ndim: Optional[int]
    shape: Optional[Tuple[int, ...]]
    shapes: Optional[List[Tuple[int, ...]]]

    def __init__(
        self,
        ref: Optional[Ref | str] = None,
        *,
        batch: Optional[int] = None,
        channel: Optional[int] = None,
        dims: Optional[Any] = None,
        dtype: Optional[str] = None,
        channel_last: Optional[bool] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        depth: Optional[int] = None,
        ndim: Optional[int] = None,
        shape: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.ref = self._coerce_ref(ref)
        self._set_scalar_attrs(
            batch=batch,
            channel=channel,
            dtype=dtype,
            channel_last=channel_last,
            width=width,
            height=height,
            depth=depth,
            ndim=ndim,
        )
        self._set_dims(dims)
        self._set_shape(shape)
        self.shapes = None

    @staticmethod
    def _coerce_ref(ref: Optional[Ref | str]) -> Optional[Ref]:
        if ref is None or isinstance(ref, Ref):
            return ref
        if isinstance(ref, str):
            return Ref.parse(ref)
        raise TypeError("ref must be None, str, or Ref")

    def _set_scalar_attrs(self, **values: Any) -> None:
        validators = {
            "batch": (is_integer, {"min_val": 1}),
            "channel": (is_integer, {"min_val": 1}),
            "dtype": (None, {}),
            "channel_last": (is_boolean, {}),
            "width": (is_integer, {"min_val": 1}),
            "height": (is_integer, {"min_val": 1}),
            "depth": (is_integer, {"min_val": 1}),
            "ndim": (is_integer, {"min_val": 1}),
        }
        for name, value in values.items():
            validator, kwargs = validators[name]
            self.set_attr(name, value, validator, **kwargs)

    def _set_dims(self, dims: Optional[Any]) -> None:
        if dims is None:
            self.dims = None
            return
        self.set_attr(
            "dims",
            to_integer_tuple(dims, keep_scalar=False),
            is_integers,
            min_val=1,
            min_dim=1,
        )

    def _set_shape(self, shape: Optional[Any]) -> None:
        if shape is None:
            self.shape = None
            return
        self.set_attr(
            "shape",
            to_integer_tuple(shape, keep_scalar=True),
            is_integers,
            min_val=0,
            min_dim=0,
        )

    def to_json_obj(self, **kwargs: Any) -> Any:
        """Serialize refs as strings while keeping the existing dict schema."""
        obj = super().to_json_obj(**kwargs)
        if isinstance(obj, dict) and self.ref is not None:
            obj["ref"] = str(self.ref)
        return obj

    def set_ref(self, ref: Optional[Ref | str]) -> None:
        """Set the producer reference from None, a string, or a Ref."""
        self.ref = self._coerce_ref(ref)

    def get_ref_segments(self) -> Optional[Tuple[str, ...]]:
        """Return structured ref segments, or None when no ref is set."""
        if self.ref is None:
            return None
        return tuple(self.ref)

    def make_shape(
        self,
        dims: Optional[Tuple[int, ...]] = None,
        channel_last: Optional[bool] = None,
    ) -> Tuple[int, ...]:
        """Construct a channel-first or channel-last tensor shape."""
        resolved_dims = self._resolve_dims(dims)
        self._validate_ndim(resolved_dims)

        if self.channel is None:
            raise ValueError("Unknown data channels (channel)")

        use_channel_last = self.channel_last if channel_last is None else channel_last
        if use_channel_last:
            return (*resolved_dims, self.channel)
        return (self.channel, *resolved_dims)

    def _resolve_dims(self, dims: Optional[Tuple[int, ...]]) -> Tuple[int, ...]:
        if dims is not None:
            return tuple(dims)
        if self.dims is not None:
            return self.dims
        inferred = self._infer_dims_from_spatial()
        if inferred is None:
            raise ValueError("Unknown data dimensions (dims)")
        return inferred

    def _validate_ndim(self, dims: Tuple[int, ...]) -> None:
        if self.ndim is not None and len(dims) != self.ndim:
            raise ValueError(f"Rank of dims {dims} != expected ndim {self.ndim}")

    def _infer_dims_from_spatial(self) -> Optional[Tuple[int, ...]]:
        dims = tuple(d for d in (self.depth, self.height, self.width) if d is not None)
        return dims or None

    def set_shape(self, shape: Optional[Tuple[int, ...]]) -> None:
        """Set the current shape and preserve previous distinct shapes."""
        new_shape = None if shape is None else tuple(shape)
        if self.shape is not None and self.shape != new_shape:
            if self.shapes is None:
                self.shapes = []
            self.shapes.append(self.shape)
        self.shape = new_shape


make_datadef = DataDef
