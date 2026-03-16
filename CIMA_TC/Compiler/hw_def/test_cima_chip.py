"""
Simple test helper to build and visualize the CIMA 4x9 chip DeviceTree.

Run from repo root:
    python -m CIMA_TC.Compiler.hw_def.test_cima_chip

It will print a JSON summary to stdout and optionally write a full JSON file.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from CIMA_TC.Compiler.IR_tool.core.jsonable import dump_json, SerializationConfig
from CIMA_TC.Compiler.hw_def.cima_chip import build_cima_chip_4x9


def main() -> None:
    chip = build_cima_chip_4x9()

    # Pretty-print a lightweight JSON string for quick inspection.
    text = dump_json(
        chip,
        config=SerializationConfig(indent=2, default_flow_style=False),
    )
    assert isinstance(text, str)
    print(text)

    # Also write a full JSON file next to this script for deeper inspection.
    out_path = Path(_SCRIPT_DIR) / "cima_chip_4x9.yaml"
    dump_json(
        chip,
        file=out_path,
        config=SerializationConfig(indent=2, default_flow_style=False),
    )
    print(f"\nFull chip JSON written to: {out_path}")


if __name__ == "__main__":
    main()

