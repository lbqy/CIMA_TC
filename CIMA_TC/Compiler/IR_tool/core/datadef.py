from __future__ import annotations

from typing import Optional, Tuple, List, Any

from .jsonable import Jsonable
from .type_utils import (
    is_integer,
    is_integers,
    is_boolean,
    to_integer_tuple
)
from .ref import Ref, InvalidRefError


class DataDef(Jsonable):
    """
    Data definition descriptor.

    This class describes tensor metadata such as:

        - reference (Ref)
        - batch size
        - channel count
        - spatial dimensions
        - shape
        - dtype

    It is typically used in model graph or layer definition systems.
    """

    # --------------------------------------------------------
    # Type annotations (for static analysis & IDE support)
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Constructor
    # --------------------------------------------------------

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
        **kwargs,
    ):
        super().__init__(**kwargs)

        # -----------------------------
        # Ref handling (new version)
        # -----------------------------

        if isinstance(ref, str):
            ref = Ref.parse(ref)   
        elif ref is not None and not isinstance(ref, Ref):
            raise TypeError("ref must be None, str, or Ref")

        self.ref = ref

        # -----------------------------
        # Scalar attributes
        # -----------------------------

        self.set_attr("batch", batch, is_integer, min_val=1)
        self.set_attr("channel", channel, is_integer, min_val=1)
        self.set_attr("dtype", dtype)
        self.set_attr("channel_last", channel_last, is_boolean)
        self.set_attr("width", width, is_integer, min_val=1)
        self.set_attr("height", height, is_integer, min_val=1)
        self.set_attr("depth", depth, is_integer, min_val=1)
        self.set_attr("ndim", ndim, is_integer, min_val=1)

        # -----------------------------
        # Tuple attributes
        # -----------------------------

        dims_tuple = to_integer_tuple(dims, keep_scalar=True)
        self.set_attr(
            "dims",
            dims_tuple,
            is_integers,
            min_val=1,
            min_dim=1,
        )

        shape_tuple = to_integer_tuple(shape, keep_scalar=True)
        self.set_attr(
            "shape",
            shape_tuple,
            is_integers,
            min_val=0,
            min_dim=0,
        )

        self.shapes = None

    # --------------------------------------------------------
    # JSON serialization
    # --------------------------------------------------------

    def to_json_obj(self, **kwargs):
        """
        Custom JSON serialization.

        If only ref is defined, serialize as a string.
        Otherwise serialize as full object.
        """
        if self.ref and len(self.__dict__) == 1:
            return str(self.ref)

        obj = super().to_json_obj(**kwargs)

        if self.ref:
            obj["ref"] = str(self.ref)

        return obj

    # --------------------------------------------------------
    # Ref helpers
    # --------------------------------------------------------

    def set_ref(
        self,
        ref: Optional[Ref | str],
    ) -> None:
        """
        Set reference using Ref or string.
        """
        if ref is None:
            self.ref = None
        elif isinstance(ref, Ref):
            self.ref = ref
        elif isinstance(ref, str):
            self.ref = Ref.parse(ref)
        else:
            raise TypeError("ref must be None, str, or Ref")

    def get_ref_segments(self):
        """
        Return structured segments if ref exists.
        """
        if self.ref is None:
            return None
        return tuple(self.ref)

    # --------------------------------------------------------
    # Shape logic
    # --------------------------------------------------------

    def make_shape(
        self,
        dims: Optional[Tuple[int, ...]] = None,
        channel_last: Optional[bool] = None,
    ) -> Tuple[int, ...]:
        """
        Construct tensor shape from metadata.

        Returns
        -------
        Tuple[int, ...]
            Final tensor shape.
        """

        # Resolve dims
        if dims is None:
            if self.dims is not None:
                dims = self.dims
            else:
                dims = self._infer_dims_from_spatial()

        if dims is None:
            raise ValueError("Unknown data dimensions (dims)")

        if self.ndim is not None and len(dims) != self.ndim:
            raise ValueError(
                f"Rank of dims {dims} != expected ndim {self.ndim}"
            )

        channel = self.channel
        if channel is None:
            raise ValueError("Unknown data channels (channel)")

        if channel_last is None:
            channel_last = self.channel_last

        if channel_last:
            return (*dims, channel)
        else:
            return (channel, *dims)

    def _infer_dims_from_spatial(self) -> Optional[Tuple[int, ...]]:
        """
        Infer dims from depth/height/width.
        """

        dims = (self.depth, self.height, self.width)
        dims = tuple(d for d in dims if d is not None)

        return dims if dims else None

    # --------------------------------------------------------
    # Shape tracking
    # --------------------------------------------------------

    def set_shape(self, shape: Optional[Tuple[int, ...]]) -> None:
        """
        Track historical shapes.
        """

        shape = None if shape is None else tuple(shape)

        if self.shape is None or self.shape == shape:
            pass
        elif not self.shapes:
            self.shapes = [self.shape]
        else:
            self.shapes.append(self.shape)

        self.shape = shape


# Backward compatibility alias
make_datadef = DataDef