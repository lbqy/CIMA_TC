from __future__ import annotations

"""
Chip-level 4x9 mesh abstraction for the CIMA NPU.

This module builds a DeviceTree containing 36 cores laid out as a 4x9 mesh.
Each core node is named using the convention ``Core{y}_{x}`` (0-based row/col).

Roles (from hw_description.md):
- Core0_4: HOSTI  (PCIe host interface)
- Core3_4: DDRI   (external DDR interface)
- Core0_3, Core0_5, Core3_3, Core3_5: routing-only cores
- All other cores are compute cores with PE/DMAC/MFOP, modeled via build_cima_core().

NoC adjacency is intentionally not modeled here; only the presence and role
of each core is captured at this level.
"""

from typing import Any, Dict, Tuple

from CIMA_TC.Compiler.IR_tool.core import DeviceTree

from .cima_core import build_cima_core, CIMACoreRoleDevice


def _core_role(y: int, x: int) -> str:
    """
    Return logical role of a core at (y, x): 'hosti', 'ddri', 'router', or 'compute'.
    """
    if (y, x) == (0, 4):
        return "HOSTI"
    if (y, x) == (3, 4):
        return "DDRI"
    if (y, x) in {(0, 3), (0, 5), (3, 3), (3, 5)}:
        return "ROUTER"
    return "COMPUTE"


def build_cima_chip_4x9() -> DeviceTree:
    """
    Build a 4x9 mesh CIMA chip as a DeviceTree.

    Top-level structure (JSON-like):
    {
      "devices": {
        "Core0_0": { ... core device tree ... },
        "Core0_1": { ... },
        ...
        "Core3_8": { ... }
      }
    }

    - Compute cores: populated via build_cima_core(core_id).
    - HOSTI / DDRI / router cores: present as empty DeviceTree containers
      with a synthetic 'role' child device for tagging.
    """
    devices: Dict[str, Any] = {}

    for y in range(4):
        for x in range(9):
            name = f"Core{y}_{x}"
            role = _core_role(y, x)
            core_id = y * 9 + x

            if role == "COMPUTE":
                devices[name] = build_cima_core(core_id=core_id)
            else:
                # For non-compute cores we keep a lightweight container with a
                # single 'role' leaf device so that schedulers can distinguish
                # HOSTI / DDRI / ROUTER from compute cores.
                role_dev = CIMACoreRoleDevice(role=role, core_id=core_id)
                devices[name] = DeviceTree(devices={"role": role_dev})

    return DeviceTree(devices=devices)


__all__ = [
    "build_cima_chip_4x9",
]

