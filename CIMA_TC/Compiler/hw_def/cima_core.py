from __future__ import annotations

"""
Core-level hardware abstraction for the CIMA NPU.

- One core contains 4 PEs, 1 DMAC and 1 MFOP.
- Each PE has 16 logical 576x128 RRAM XBs (crossbars).
- Uses BaseRuntime / DeviceTree from IR_tool.core.device to build a device tree.

Example
-------
from CIMA_TC.Compiler.hw_def.cima_core import (
    CIMAPEDevice,
    CIMADMACDevice,
    CIMAMFOPDevice,
    build_cima_core,
)

core0 = build_cima_core(core_id=0)
for name, dev in core0.iter_devices():
    print(name, dev.kind)
"""

from typing import Any, Dict, Optional, Tuple

from CIMA_TC.Compiler.IR_tool.core import BaseRuntime, DeviceTree
from CIMA_TC.Compiler.IR_tool.core.type_utils import is_integer, is_boolean


class _CIMABaseUnit(BaseRuntime):
    """
    Base class for CIMA compute units: PE / DMAC / MFOP.

    Only records hardware capability limits on attributes; actual mapping
    and scheduling policies are decided by the backend.
    """

    __abstract__: bool = True

    threads: Optional[int] = None
    in_bits: Optional[int] = None
    out_bits: Optional[int] = None
    weight_bits: Optional[int] = None

    def __init__(
        self,
        *,
        threads: Optional[int] = None,
        in_bits: Optional[int] = None,
        out_bits: Optional[int] = None,
        weight_bits: Optional[int] = None,
        enable: Optional[bool] = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # Lightweight capability / status flags used by later schedulers
        # to check mapping constraints.
        self.set_attr("threads", threads or self.threads, is_integer, min_val=1)
        self.set_attr("in_bits", in_bits or self.in_bits, is_integer, min_val=1)
        self.set_attr("out_bits", out_bits or self.out_bits, is_integer, min_val=1)
        self.set_attr("weight_bits", weight_bits or self.weight_bits, is_integer, min_val=1)
        self.set_attr("enable", enable, is_boolean)


class CIMAPEDevice(_CIMABaseUnit):
    """
    CIMA PE compute unit.

    - 16 logical 576x128 RRAM XBs per PE, 2 threads.
    - 4-bit input / output / weights.
    - Mainly used for conv2d / linear-like ops.
    """

    kind: str = "cima_pe"

    # Default capabilities (more profile fields can be added if needed).
    threads: Optional[int] = 2
    in_bits: Optional[int] = 4
    out_bits: Optional[int] = 4
    weight_bits: Optional[int] = 4

    # RRAM XB configuration: 16 logical 576x128 crossbars per PE.
    xb_rows: int = 576
    xb_cols: int = 128
    xb_count: int = 16

    def __init__(
        self,
        *,
        pe_index: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # Expose XB geometry so placement/mapping can reason about tiling.
        self.set_attr("xb_rows", self.xb_rows, is_integer, min_val=1)
        self.set_attr("xb_cols", self.xb_cols, is_integer, min_val=1)
        self.set_attr("xb_count", self.xb_count, is_integer, min_val=1)
        if pe_index is not None:
            self.set_attr("pe_index", pe_index, is_integer, min_val=0)

    # Coarse-grained whitelist; detailed shape constraints are validated
    # by the placement / scheduling logic.
    white_list = ("conv2d", "linear")


class CIMADMACDevice(_CIMABaseUnit):
    """
    CIMA DMAC digital compute unit.

    - Can switch between 256x64 and 512x32 array shapes.
    - 8-bit input / output / weights, single thread.
    - Typically used for first/last layers or other high-precision conv/fc.
    """

    kind: str = "cima_dmac"

    threads: Optional[int] = 1
    in_bits: Optional[int] = 8
    out_bits: Optional[int] = 8
    weight_bits: Optional[int] = 8

    # Supported array shapes (rows, cols): 256x64 and 512x32.
    array_shapes: Tuple[Tuple[int, int], Tuple[int, int]] = ((256, 64), (512, 32))

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.set_attr("array_shapes", self.array_shapes)

    white_list = ("conv2d", "linear")


class CIMAMFOPDevice(_CIMABaseUnit):
    """
    CIMA MFOP up/down-sampling unit.

    - Supports up_sample / maxpool / avgpool.
    - 4-bit input, 2 threads.
    - Handles scale-changing ops (pooling / upsampling).
    """

    kind: str = "cima_mfop"

    threads: Optional[int] = 2
    in_bits: Optional[int] = 4
    out_bits: Optional[int] = 4
    # MFOP has no persistent weights; use a dummy positive value to satisfy
    # validation (min_val=1) while keeping semantics clear.
    weight_bits: Optional[int] = 4

    white_list = (
        "max_pool2d",
        "avg_pool2d",
        "adaptive_avg_pool2d",
        "resize",
        "upsample",
    )


class CIMACoreRoleDevice(BaseRuntime):
    """
    Lightweight device used to tag non-compute cores (HOSTI / DDRI / ROUTER).
    It is not a real compute resource but helps schedulers distinguish roles.
    """

    kind: str = "cima_core_role"

    def __init__(
        self,
        *,
        role: str,
        core_id: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.set_attr("role", role)
        self.set_attr("core_id", core_id, is_integer, min_val=0)


def build_cima_core(core_id: int, *, enabled: bool = True) -> DeviceTree:
    """
    Build the hardware abstraction of a single CIMA core,
    containing 4 PEs, 1 DMAC and 1 MFOP compute unit.

    Returned DeviceTree JSON-like structure:
    {
      "devices": {
        "pe":   { "kind": "cima_pe",   "number": 0, "threads": 2, ... },
        "dmac": { "kind": "cima_dmac", "number": 0, "threads": 1, ... },
        "mfop": { "kind": "cima_mfop", "number": 0, "threads": 2, ... }
      }
    }
    """

    devices: Dict[str, Any] = {}
    # 4 PEs per core (e.g. attached to 4 NoC directions).
    for pe_idx in range(4):
        devices[f"pe{pe_idx}"] = CIMAPEDevice(number=core_id, enable=enabled, pe_index=pe_idx)
    # One DMAC and one MFOP per core.
    devices["dmac"] = CIMADMACDevice(number=core_id, enable=enabled)
    devices["mfop"] = CIMAMFOPDevice(number=core_id, enable=enabled)

    return DeviceTree(devices=devices)


__all__ = [
    "CIMAPEDevice",
    "CIMADMACDevice",
    "CIMAMFOPDevice",
    "CIMACoreRoleDevice",
    "build_cima_core",
]

