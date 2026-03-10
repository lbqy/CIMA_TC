"""
Pytest for device module: DeviceTree (container), BaseDevice, BaseRuntime, make_device.
Run from repo root: pytest CIMA_TC/Compiler/IR_tool/core/test/test_device.py -v
"""

from CIMA_TC.Compiler.IR_tool.core.device import BaseDevice, DeviceTree
import pytest

from CIMA_TC.Compiler.IR_tool.core.jsonable import dump_json

from ..ref import InvalidNameError, Ref
from ..device import DeviceTree, BaseDevice, BaseRuntime, make_device
from typing import Tuple


# ============================================================
# make_device
# ============================================================


def test_make_device_from_string():
    """make_device('runtime') returns BaseRuntime (BaseDevice) instance."""
    d = make_device("runtime")
    assert d is not None
    assert isinstance(d, BaseRuntime)
    assert isinstance(d, BaseDevice)
    assert d.kind == "runtime"


def test_make_device_from_dict():
    """make_device(dict) creates device with given attributes."""
    d = make_device({"kind": "runtime", "number": 0})
    assert d is not None
    assert isinstance(d, BaseRuntime)
    assert d.kind == "runtime"
    assert d.number == 0


def test_make_device_from_kwargs():
    """make_device(None, kind='runtime', number=1) works."""
    d = make_device(None, kind="runtime", number=1)
    assert d is not None
    assert d.kind == "runtime"
    assert d.number == 1


def test_make_device_from_existing_instance():
    """make_device(existing BaseDevice) returns the same instance."""
    orig = make_device("runtime", number=0)
    same = make_device(orig)
    assert same is orig


def test_make_device_unknown_kind_raises():
    """make_device with unregistered kind raises KeyError."""
    with pytest.raises(KeyError):
        make_device("_nonexistent_kind_xyz_")


def test_make_device_none_no_kwargs_returns_none():
    """make_device(None) with no kwargs returns None."""
    assert make_device(None) is None


# ============================================================
# DeviceTree: init and add_device
# ============================================================


def test_device_tree_empty():
    """DeviceTree() has devices=None."""
    t = DeviceTree()
    assert t.devices is None


def test_device_tree_with_devices_dict_string_kind():
    """DeviceTree(devices={'r0': 'runtime'}) builds devices from string kinds."""
    t = DeviceTree(devices={"r0": "runtime"})
    assert t.devices is not None
    assert "r0" in t.devices
    assert isinstance(t.devices["r0"], BaseRuntime)
    assert isinstance(t.devices["r0"], BaseDevice)
    assert t.devices["r0"].kind == "runtime"


def test_device_tree_with_devices_dict_mapping():
    """DeviceTree(devices={'r0': {'kind': 'runtime', 'number': 0}}) builds from dict."""
    t = DeviceTree(devices={"r0": {"kind": "runtime", "number": 0}})
    assert t.devices["r0"].number == 0


def test_device_tree_with_nested_devices_dict():
    """DeviceTree(devices={'a': {'devices': {'b': 'runtime'}}}) builds nested tree."""
    t = DeviceTree(devices={"a": {"devices": {"b": "runtime"}}})
    assert t.devices is not None
    assert "a" in t.devices
    assert isinstance(t.devices["a"], DeviceTree)
    assert t.devices["a"].devices is not None
    assert "b" in t.devices["a"].devices
    assert isinstance(t.devices["a"].devices["b"], BaseRuntime)


def test_device_tree_add_device_by_string():
    """add_device(name, kind_str) adds device created via make_device."""
    t = DeviceTree()
    t.add_device("cpu", "runtime")
    assert t.devices is not None
    assert "cpu" in t.devices
    assert isinstance(t.devices["cpu"], BaseRuntime)


def test_device_tree_add_device_by_instance():
    """add_device(name, BaseDevice_instance) adds clone of device."""
    t = DeviceTree()
    dev = make_device("runtime", number=2)
    t.add_device("r2", dev)
    assert t.devices["r2"] is not dev
    assert t.devices["r2"].kind == dev.kind
    assert t.devices["r2"].number == 2


def test_device_tree_add_device_duplicate_name_raises():
    """add_device with existing name raises ValueError."""
    t = DeviceTree(devices={"a": "runtime"})
    with pytest.raises(ValueError, match="already exists"):
        t.add_device("a", "runtime")


def test_device_tree_add_device_invalid_name_raises():
    """add_device with invalid name (e.g. starts with digit) raises InvalidNameError."""
    t = DeviceTree()
    with pytest.raises(InvalidNameError):
        t.add_device("1cpu", "runtime")


def test_device_tree_add_device_invalid_kind_type_raises():
    """add_device with kind not str, BaseDevice or DeviceTree raises TypeError."""
    t = DeviceTree()
    with pytest.raises(TypeError, match="invalid device kind"):
        t.add_device("x", 123)


# ============================================================
# DeviceTree: get_device
# ============================================================


def test_device_tree_get_device_flat():
    """get_device(ref) returns device by name."""
    t = DeviceTree(devices={"r0": "runtime"})
    dev = t.get_device("r0")
    assert dev is not None
    assert dev is t.devices["r0"]


def test_device_tree_get_device_missing_returns_none():
    """get_device(missing_ref) returns None."""
    t = DeviceTree(devices={"r0": "runtime"})
    assert t.get_device("nonexistent") is None


def test_device_tree_get_device_nested():
    """get_device('a.b') returns nested device when tree has nested structure."""
    inner = DeviceTree(devices={"b": "runtime"})
    t = DeviceTree()
    t.add_device("a", inner)
    dev = t.get_device("a.b")
    assert dev is not None
    assert isinstance(dev, BaseRuntime)


# ============================================================
# DeviceTree: iter_devices
# ============================================================


def test_device_tree_iter_devices_flat():
    """iter_devices yields (path, node) for each node; nodes can be BaseDevice or DeviceTree."""
    t = DeviceTree(devices={"r0": "runtime", "r1": "runtime"})
    pairs = list[Tuple[str, DeviceTree | BaseDevice]](t.iter_devices())
    assert len(pairs) == 2
    paths = {p[0] for p in pairs}
    assert paths == {"r0", "r1"}
    assert all(isinstance(p[1], BaseDevice) for p in pairs)


def test_device_tree_iter_devices_empty():
    """iter_devices on empty tree yields nothing."""
    t = DeviceTree()
    assert list[Tuple[str, DeviceTree | BaseDevice]](t.iter_devices()) == []


def test_device_tree_iter_devices_deep_false():
    """iter_devices(deep=False) does not recurse into nested DeviceTree."""
    inner = DeviceTree(devices={"b": "runtime"})
    t = DeviceTree()
    t.add_device("a", inner)
    pairs = list[Tuple[str, DeviceTree | BaseDevice]](t.iter_devices(deep=False))
    assert len(pairs) == 1
    assert pairs[0][0] == "a"
    assert pairs[0][1] is inner


def test_device_tree_iter_devices_deep_true():
    """iter_devices(deep=True) yields nested devices with dotted path."""
    inner = DeviceTree(devices={"b": "runtime"})
    t = DeviceTree()
    t.add_device("a", inner)
    pairs = list[Tuple[str, DeviceTree | BaseDevice]](t.iter_devices(deep=True))
    assert len(pairs) == 2
    paths = {p[0] for p in pairs}
    assert "a" in paths
    assert "a.b" in paths


# ============================================================
# BaseDevice / BaseRuntime: attributes and can_map
# ============================================================


def test_base_runtime_can_map_default_all_true():
    """BaseRuntime with no white_list allows all op_ids."""
    d = make_device("runtime")
    assert d.can_map("relu") is True
    assert d.can_map("conv2d") is True


def test_base_runtime_can_map_black_list():
    """BaseRuntime.black_list excludes op_ids."""
    d = make_device("runtime", number=0)
    d.black_list = ("conv2d",)
    assert d.can_map("relu") is True
    assert d.can_map("conv2d") is False


def test_base_runtime_can_map_white_list():
    """BaseRuntime.white_list restricts to listed op_ids."""
    d = make_device("runtime", number=0)
    d.white_list = ("relu", "add")
    assert d.can_map("relu") is True
    assert d.can_map("add") is True
    assert d.can_map("conv2d") is False


def test_device_has_can_map():
    """BaseDevice (and BaseRuntime) have can_map method."""
    d = make_device("runtime")
    assert hasattr(d, "can_map")


def test_device_is_not_device_tree():
    """BaseDevice does not inherit DeviceTree (leaf only)."""
    d = make_device("runtime")
    assert not isinstance(d, DeviceTree)


# ============================================================
# Validation and serialization
# ============================================================


def test_device_tree_to_json_obj():
    """DeviceTree with devices can be serialized to JSON-suitable dict."""
    t = DeviceTree(devices={"r0": "runtime"})
    obj = t.to_json_obj()
    assert isinstance(obj, dict)
    assert "devices" in obj
    assert "r0" in obj["devices"]


def test_make_device_number_validation():
    """make_device with number < 0 raises ValidationError."""
    from ..type_utils import ValidationError
    with pytest.raises(ValidationError):
        make_device("runtime", number=-1)


def test_device_clone():
    """BaseDevice can be cloned with overrides."""
    d = make_device("runtime", number=0)
    c = d.clone(number=1)
    assert c is not d
    assert c.kind == d.kind
    assert c.number == 1


# ============================================================
# Chip -> Core -> PE hardware hierarchy example
# ============================================================


def _build_chip_core_pe_hierarchy() -> DeviceTree:
    """
    Build device tree: chip -> core -> PE (Processing Element).
    Root has chip0, chip1; each chip has cores; each core has PEs (BaseRuntime).
    """
    pe0 = make_device("runtime", number=0)
    pe1 = make_device("runtime", number=1)
    pe2 = make_device("runtime", number=2)

    core0 = DeviceTree(devices={"pe0": pe0, "pe1": pe1})
    core1 = DeviceTree(devices={"pe0": pe0.clone(), "pe1": pe1.clone()})
    core2 = DeviceTree(devices={
        "pe0": make_device("runtime", number=0),
        "pe1": make_device("runtime", number=1),
        "pe2": make_device("runtime", number=2),
    })

    chip0 = DeviceTree()
    chip0.add_device("core0", core0)
    chip0.add_device("core1", core1)

    chip1 = DeviceTree()
    chip1.add_device("core0", core2)

    root = DeviceTree()
    root.add_device("chip0", chip0)
    root.add_device("chip1", chip1)
    return root


def test_chip_core_pe_build():
    """Chip-core-PE hierarchy builds without error and has expected structure."""
    tree = _build_chip_core_pe_hierarchy()
    assert tree.devices is not None
    assert "chip0" in tree.devices
    assert "chip1" in tree.devices
    chip0 = tree.devices["chip0"]
    assert isinstance(chip0, DeviceTree)
    assert chip0.devices is not None
    assert "core0" in chip0.devices
    assert "core1" in chip0.devices
    core0 = chip0.devices["core0"]
    assert isinstance(core0, DeviceTree)
    assert "pe0" in core0.devices
    assert "pe1" in core0.devices
    assert isinstance(core0.devices["pe0"], BaseRuntime)
    assert isinstance(core0.devices["pe1"], BaseRuntime)


def test_chip_core_pe_get_device_paths():
    """get_device resolves chip0, chip0.core0, chip0.core0.pe0, chip1.core0.pe2."""
    tree = _build_chip_core_pe_hierarchy()
    node = tree.get_device("chip0")
    assert node is not None
    assert isinstance(node, DeviceTree)
    assert "core0" in (node.devices or {})

    node = tree.get_device("chip0.core0")
    assert node is not None
    assert isinstance(node, DeviceTree)
    assert "pe0" in (node.devices or {})

    node = tree.get_device("chip0.core0.pe0")
    assert node is not None
    assert isinstance(node, BaseRuntime)
    assert node.kind == "runtime"
    assert node.number == 0

    node = tree.get_device("chip1.core0.pe2")
    assert node is not None
    assert isinstance(node, BaseRuntime)
    assert node.number == 2

    assert tree.get_device("chip0.missing") is None


def test_chip_core_pe_iter_devices_deep():
    """iter_devices(deep=True) yields all nodes including chip0, chip0.core0, chip0.core0.pe0, etc."""
    tree = _build_chip_core_pe_hierarchy()
    paths = [path for path, _ in tree.iter_devices(deep=True)]
    assert "chip0" in paths
    assert "chip1" in paths
    assert "chip0.core0" in paths
    assert "chip0.core1" in paths
    assert "chip0.core0.pe0" in paths
    assert "chip0.core0.pe1" in paths
    assert "chip1.core0.pe0" in paths
    assert "chip1.core0.pe2" in paths
    # Total: chip0, chip0.core0, chip0.core0.pe0, chip0.core0.pe1, chip0.core1, chip0.core1.pe0, chip0.core1.pe1,
    #        chip1, chip1.core0, chip1.core0.pe0, chip1.core0.pe1, chip1.core0.pe2 = 12
    assert len(paths) == 12


def test_chip_core_pe_leaf_can_map():
    """PE at chip0.core0.pe0 can_map(op_id) works (BaseRuntime allows by default)."""
    tree = _build_chip_core_pe_hierarchy()
    pe = tree.get_device("chip0.core0.pe0")
    assert pe is not None
    assert isinstance(pe, BaseRuntime)
    assert pe.can_map("relu") is True
    assert pe.can_map("conv2d") is True


def test_chip_core_pe_to_json_obj():
    """Chip-core-PE tree serializes to JSON with devices hierarchy."""
    tree = _build_chip_core_pe_hierarchy()
    obj = tree.to_json_obj()
    assert "devices" in obj
    assert "chip0" in obj["devices"]
    assert "chip1" in obj["devices"]
