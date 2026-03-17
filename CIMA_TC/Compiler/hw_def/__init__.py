"""
Hardware definition helpers for the CIMA NPU.

This package provides:
- Core-level abstractions: PE / DMAC / MFOP units per core.
- Chip-level 4x9 mesh topology builders.
"""

from .cima_core import (
    CIMAPEDevice,
    CIMADMACDevice,
    CIMAMFOPDevice,
    CIMACoreRoleDevice,
    build_cima_core,
)
from .cima_chip import build_cima_chip_4x9

__all__ = [
    "CIMAPEDevice",
    "CIMADMACDevice",
    "CIMAMFOPDevice",
    "CIMACoreRoleDevice",
    "build_cima_core",
    "build_cima_chip_4x9",
]

