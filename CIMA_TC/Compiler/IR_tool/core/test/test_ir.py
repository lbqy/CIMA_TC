"""
Pytest for core/ir.py (BaseIR).

Covers:
- load_ir from None / dict / json string / file
- make_ir convenience wrapper
- save_ir wrapper around dump_json
- BaseIR initializes both layers (GraphLayer) and devices (DeviceTree)
"""

import json

import pytest

from ..ir import BaseIR, save_ir
from ..device import DeviceTree, BaseRuntime


def _sample_layers_dict():
    # Minimal valid graph: input -> output
    return {
        "in": {"type": "input"},
        "out": {"type": "output", "inputs": ["in"]},
    }


def _sample_devices_dict():
    # chip0.core0.pe0 is a runtime leaf
    return {
        "chip0": {
            "devices": {
                "core0": {
                    "devices": {
                        "pe0": {"kind": "runtime", "number": 0},
                    }
                }
            }
        }
    }


def test_make_ir_builds_layers_and_devices():
    ir = BaseIR.make_ir(layers=_sample_layers_dict(), devices=_sample_devices_dict())
    assert ir.ir_version == "model_ir"

    # layers (GraphLayer)
    assert ir.layers is not None
    assert "in" in ir.layers and "out" in ir.layers

    # devices (DeviceTree)
    assert ir.devices is not None
    assert "chip0" in ir.devices
    node = ir.get_device("chip0.core0.pe0")
    assert node is not None
    assert isinstance(node, BaseRuntime)
    assert node.number == 0


def test_load_ir_none_returns_empty():
    ir = BaseIR.load_ir()
    assert isinstance(ir, BaseIR)
    assert ir.ir_version == "model_ir"


def test_load_ir_from_dict():
    data = {
        "type": "graph",
        "layers": _sample_layers_dict(),
        "devices": _sample_devices_dict(),
        "ir_version": "model_ir",
    }
    ir = BaseIR.load_ir(data)
    assert ir.layers is not None and "out" in ir.layers
    assert isinstance(ir.get_device("chip0.core0.pe0"), BaseRuntime)


def test_load_ir_from_json_string():
    data = {
        "type": "graph",
        "layers": _sample_layers_dict(),
        "devices": _sample_devices_dict(),
    }
    text = json.dumps(data)
    ir = BaseIR.load_ir(text)
    assert ir.layers is not None and "in" in ir.layers
    assert isinstance(ir.get_device("chip0.core0.pe0"), BaseRuntime)


def test_save_ir_returns_string_when_no_file():
    ir = BaseIR.make_ir(layers=_sample_layers_dict(), devices=_sample_devices_dict())
    text = save_ir(ir)
    assert isinstance(text, str)
    obj = json.loads(text)
    assert obj["type"] == "graph"
    assert "layers" in obj


def test_save_and_load_ir_file_roundtrip(tmp_path):
    ir = BaseIR.make_ir(layers=_sample_layers_dict(), devices=_sample_devices_dict())
    path = tmp_path / "m.json"
    assert save_ir(ir, file=path) is None

    loaded = BaseIR.load_ir(file=path)
    assert loaded.layers is not None and "out" in loaded.layers
    assert isinstance(loaded.get_device("chip0.core0.pe0"), BaseRuntime)


def test_load_ir_rejects_data_and_file_together(tmp_path):
    with pytest.raises(ValueError, match="both data and file"):
        BaseIR.load_ir("{}", file=tmp_path / "x.json")

