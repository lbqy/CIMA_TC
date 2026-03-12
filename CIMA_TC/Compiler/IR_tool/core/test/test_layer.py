import pytest
from typing import Any, Tuple

from ..layer import (
    BaseLayer,
    OpLayer,
    GraphLayer,
    BlockLayer,
    InputLayer,
    OutputLayer,
    make_layer,
)
from ..datadef import DataDef
from ..type_utils import ValidationError
from ..op import BaseOp, UnaryOp, BinaryOp
from ..ref import InvalidRefError, InvalidNameError
from ...ops import Conv2dOp, ReluOp, AddOp  # noqa: F401 - register conv2d

# ============================================================
# Helpers
# ============================================================

def input_layer(output_refs=None):
    """output_refs: list of output ref names, e.g. ['out']"""
    if output_refs is None:
        output_refs = ["out"]
    return make_layer({"type": "input", "outputs": output_refs})


def output_layer(input_refs=None):
    """input_refs: list of input ref names, e.g. ['inp']"""
    if input_refs is None:
        input_refs = ["inp"]
    return make_layer({"type": "output", "inputs": input_refs})


# ============================================================
# Basic layer creation
# ============================================================

def test_basic_irlayer_creation():
    layer = make_layer({
        "type": "op",
        "op": "relu",
        "inputs": ["x"],
        "outputs": ["y"],
    })
    layer.validate()


# ============================================================
# OpLayer validation
# ============================================================

def test_oplayer_invalid_input_number():
    """OpLayer requires input count to match op.num_inputs; empty inputs trigger validate on create."""
    with pytest.raises(ValidationError):
        make_layer({
            "type": "op",
            "op": "relu",
            "inputs": [],
            "outputs": ["y"],
        })


# ============================================================
# GraphLayer
# ============================================================

def test_graphlayer_build_by_layer_refs():
    """Build graph with inputs as layer names; outputs need not be given (default to layer name)."""
    g = make_layer({
        "type": "graph",
        "layers": {
            "in": {"type": "input"},
            "mid": {"type": "op", "op": "relu", "inputs": ["in"]},
            "out": {"type": "output", "inputs": ["mid"]},
        },
    })
    g.validate()
    assert str(g.layers["in"].outputs[0].ref) == "in"
    assert str(g.layers["mid"].inputs[0].ref) == "in"
    assert str(g.layers["mid"].outputs[0].ref) == "mid"
    assert str(g.layers["out"].inputs[0].ref) == "mid"


def test_graphlayer_topological_order():
    """topological_order returns layer names so that producers come before consumers."""
    # Graph: in1, in2 -> a, in1 -> b -> c, a and c -> d -> out (diamond + multi-input)
    g = make_layer({
        "type": "graph",
        "layers": {
            "in1": {"type": "input"},
            "in2": {"type": "input"},
            "a": {"type": "op", "op": "add", "inputs": ["in1", "in2"]},
            "b": {"type": "op", "op": "relu", "inputs": ["in1"]},
            "c": {"type": "op", "op": "relu", "inputs": ["b"]},
            "d": {"type": "op", "op": "add", "inputs": ["a", "c"]},
            "out": {"type": "output", "inputs": ["d"]},
        },
    })
    order = g.topological_order()
    names = {"in1", "in2", "a", "b", "c", "d", "out"}
    assert set(order) == names, "order must contain every layer exactly once"
    assert len(order) == len(names)

    # Build edges (producer -> consumer) from get_all_inputs; ref_name is the producer
    all_inputs = g.get_all_inputs()
    edges = []
    for consumer, refs in all_inputs.items():
        if refs is None:
            continue
        for ref_str in refs:
            producer = ref_str.split(":")[0]
            if producer in names:
                edges.append((producer, consumer))

    # Topological order: for every edge (u, v), u must appear before v
    pos = {name: i for i, name in enumerate(order)}
    for u, v in edges:
        assert pos[u] < pos[v], f"edge ({u!r}, {v!r}): {u!r} must come before {v!r} in topological order"

    # Explicit ordering constraints for this graph
    assert order.index("in1") < order.index("a")
    assert order.index("in2") < order.index("a")
    assert order.index("in1") < order.index("b")
    assert order.index("b") < order.index("c")
    assert order.index("a") < order.index("d")
    assert order.index("c") < order.index("d")
    assert order.index("d") < order.index("out")


def test_graphlayer_get_all_inputs():
    """get_all_inputs returns dict[layer_name] -> list of input ref strings; InputLayer -> None."""
    g = make_layer({
        "type": "graph",
        "layers": {
            "in": {"type": "input"},
            "mid": {"type": "op", "op": "relu", "inputs": ["in"]},
            "out": {"type": "output", "inputs": ["mid"]},
        },
    })
    inputs = g.get_all_inputs()
    assert inputs["in"] is None
    assert inputs["mid"] == ["in"]
    assert inputs["out"] == ["mid"]


def test_graphlayer_get_all_outputs():
    """get_all_outputs returns dict[layer_name] -> list of consumers (inferred from inputs); OutputLayer -> None."""
    g = make_layer({
        "type": "graph",
        "layers": {
            "in": {"type": "input"},
            "mid": {"type": "op", "op": "relu", "inputs": ["in"]},
            "out": {"type": "output", "inputs": ["mid"]},
        },
    })
    outputs = g.get_all_outputs()
    assert outputs["in"] == ["mid"]
    assert outputs["mid"] == ["out"]
    assert outputs["out"] is None


def test_graphlayer_get_all_inputs_outputs_with_branches():
    """get_all_inputs/outputs with ref_name:index (split branches)."""
    g = make_layer({
        "type": "graph",
        "layers": {
            "in": {"type": "input"},
            "split": {"type": "op", "op": "relu", "inputs": ["in"]},
            "out": {"type": "output", "inputs": ["split:0", "split:1"]},
        },
    })
    assert g.get_all_inputs()["out"] == ["split:0", "split:1"]
    assert g.get_all_outputs()["split"] == ["out"]


def test_graphlayer_outputs_serialized():
    """Outputs (including graph-filled defaults) are included in JSON."""
    g = make_layer({
        "type": "graph",
        "layers": {
            "in": {"type": "input"},
            "mid": {"type": "op", "op": "relu", "inputs": ["in"]},
            "out": {"type": "output", "inputs": ["mid"]},
        },
    })
    obj = g.to_json_obj()
    assert "outputs" in obj["layers"]["in"]
    assert "outputs" in obj["layers"]["mid"]


def test_graphlayer_explicit_outputs_are_serialized():
    """Explicit outputs should be preserved in JSON."""
    g = make_layer({
        "type": "graph",
        "layers": {
            "in": {"type": "input"},
            "mid": {"type": "op", "op": "relu", "inputs": ["in"], "outputs": ["custom_out"]},
            "out": {"type": "output", "inputs": ["mid"]},
        },
    })
    obj = g.to_json_obj()
    assert "outputs" in obj["layers"]["mid"]


def test_graphlayer_build_dict_with_explicit_layer_params():
    """Graph built from dict can have layers with explicit params (e.g. op as dict with in_channel)."""
    g = make_layer({
        "type": "graph",
        "layers": {
            "in": {"type": "input"},
            "conv": {
                "type": "op",
                "op": {"op_id": "conv2d", "in_channel": 3, "out_channel": 16, "kernel": [3, 3]},
                "inputs": ["in"],
            },
            "out": {"type": "output", "inputs": ["conv"]},
        },
    })
    g.validate()
    assert g.layers["conv"].op.in_channel == 3
    assert g.layers["conv"].op.out_channel == 16


def test_graphlayer_add_and_validate():
    """Graph: in (input), out (output) consumes in; outputs default to layer name."""
    g = make_layer({
        "type": "graph",
        "layers": {
            "in": {"type": "input"},
            "out": {"type": "output", "inputs": ["in"]},
        },
    })
    g.validate()
    assert str(g.layers["in"].outputs[0].ref) == "in"
    assert str(g.layers["out"].inputs[0].ref) == "in"


def test_graphlayer_missing_input():
    """Graph must contain at least one InputLayer (here only op and output, no input)."""
    with pytest.raises(ValidationError, match="InputLayer"):
        make_layer({
            "type": "graph",
            "layers": {
                "mid": {"type": "op", "op": "relu", "inputs": ["mid"]},
                "out": {"type": "output", "inputs": ["mid"]},
            },
        })


def test_graphlayer_missing_output():
    """Graph must contain at least one OutputLayer."""
    with pytest.raises(ValidationError, match="OutputLayer"):
        make_layer({
            "type": "graph",
            "layers": {
                "in": {"type": "input"},
            },
        })


def test_graphlayer_connection_mismatch_raises():
    """When a layer inputs ref is not a layer name in the graph, validation fails."""
    with pytest.raises(ValidationError, match="not a layer|consumed but not produced|ref.*is not a layer"):
        make_layer({
            "type": "graph",
            "layers": {
                "in": {"type": "input"},
                "out": {"type": "output", "inputs": ["nonexistent"]},
            },
        })


def test_graphlayer_input_ref_with_index():
    """Inputs ref can be ref_name:index (layer name + branch); connection is defined by inputs only, outputs need not be declared."""
    # split has no explicit outputs; consumer uses "split:0" and "split:1" to reference branches
    g = make_layer({
        "type": "graph",
        "layers": {
            "in": {"type": "input"},
            "split": {
                "type": "op",
                "op": "relu",
                "inputs": ["in"],
            },
            "out": {"type": "output", "inputs": ["split:0", "split:1"]},
        },
    })
    g.validate()
    assert g.layers["out"].inputs[0].ref is not None
    assert str(g.layers["out"].inputs[0].ref) == "split:0"
    assert str(g.layers["out"].inputs[1].ref) == "split:1"


def test_graphlayer_input_ref_nonexistent_layer_raises():
    """When ref is ref_name:index but ref_name is not a layer in the graph, validation fails."""
    with pytest.raises(ValidationError, match="not a layer in this graph"):
        make_layer({
            "type": "graph",
            "layers": {
                "in": {"type": "input"},
                "out": {"type": "output", "inputs": ["nonexistent:0"]},
            },
        })


# ============================================================
# Multi IO
# ============================================================

def test_graphlayer_multi_io():
    """Multiple inputs/outputs; inputs are layer names (in1->out1, in2->out2)."""
    g = make_layer({
        "type": "graph",
        "layers": {
            "in1": {"type": "input"},
            "in2": {"type": "input"},
            "out1": {"type": "output", "inputs": ["in1"]},
            "out2": {"type": "output", "inputs": ["in2"]},
        },
    })
    g.validate()


# ============================================================
# Nested graph
# ============================================================

def test_nested_graph():
    """Nested graph: outer in, sub (graph with in2->out2), outer out consumes sub."""
    g = make_layer({
        "type": "graph",
        "layers": {
            "in": {"type": "input"},
            "sub": {
                "type": "graph",
                "layers": {
                    "in2": {"type": "input"},
                    "out2": {"type": "output", "inputs": ["in2"]},
                },
            },
            "out": {"type": "output", "inputs": ["sub"]},
        },
    })
    g.validate()


# ============================================================
# BlockLayer
# ============================================================

def test_blocklayer_repeat():
    b = make_layer({
        "type": "block",
        "repeat": 3,
        "layers": {
            "in": {"type": "input"},
            "out": {"type": "output", "inputs": ["in"]},
        },
    })
    b.validate()


def test_blocklayer_invalid_repeat():
    """BlockLayer.repeat must be >= 1."""
    with pytest.raises(ValidationError):
        make_layer({
            "type": "block",
            "repeat": 0,
            "layers": {
                "in": {"type": "input"},
                "out": {"type": "output", "inputs": ["in"]},
            },
        })

# ============================================================
# OpLayer attribute access
# ============================================================

def test_oplayer_attribute_access():
    layer = make_layer({
        "type": "op",
        "op": "relu",
        "inputs": ["x"],
        "outputs": ["y"],
    })
    assert layer.type == "op"
    assert isinstance(layer.op, ReluOp)
    assert isinstance(layer.inputs, list)
    assert len(layer.inputs) == 1
    assert isinstance(layer.inputs[0], DataDef)
    assert str(layer.inputs[0].ref) == "x"
    assert isinstance(layer.outputs, list)
    assert len(layer.outputs) == 1
    assert str(layer.outputs[0].ref) == "y"
    assert layer.weights is None
    assert layer.has_subgraph() is False
    assert list[Tuple[str, BaseLayer]](layer.iter_sublayers()) == []


def test_oplayer_supports_op_dict_parameters():
    """OpLayer should support op specified as dict with parameters (e.g. conv2d)."""
    layer = make_layer({
        "type": "op",
        "op": {
            "op_id": "conv2d",
            "in_channel": 3,
            "out_channel": 16,
            "kernel": [3, 3],
        },
        "inputs": ["x"],
        "outputs": ["y"],
    })
    layer.validate()
    assert layer.op.op_id == "conv2d"
    assert layer.op.in_channel == 3
    assert layer.op.out_channel == 16


# ============================================================
# GraphLayer attribute access
# ============================================================

def test_graphlayer_attribute_access():
    g = make_layer({
        "type": "graph",
        "layers": {
            "in": {"type": "input"},
            "out": {"type": "output", "inputs": ["in"]},
        },
    })
    assert g.type == "graph"
    assert isinstance(g.layers, dict)
    assert "in" in g.layers
    assert "out" in g.layers

    # subgraph capability
    assert g.has_subgraph() is True
    assert len(list[Tuple[str, BaseLayer]](g.iter_sublayers())) == 2

    # child layer types
    assert isinstance(g.layers["in"], InputLayer)
    assert isinstance(g.layers["out"], OutputLayer)


# ============================================================
# BlockLayer attribute access
# ============================================================

def test_blocklayer_attribute_access():
    b = make_layer({
        "type": "block",
        "repeat": 2,
        "layers": {
            "in": {"type": "input"},
            "out": {"type": "output", "inputs": ["in"]},
        },
    })

    assert b.type == "block"
    assert b.repeat == 2
    assert b.is_single() is False

    # inherited graph behavior
    assert isinstance(b.layers, dict)
    assert b.has_subgraph() is True


# ============================================================
# IO layer restrictions
# ============================================================

def test_inputlayer_access():
    n = input_layer()
    assert n.type == "input"
    assert n.inputs is None
    assert isinstance(n.outputs, list)
    assert len(n.outputs) == 1
    assert str(n.outputs[0].ref) == "out"


def test_outputlayer_access():
    n = output_layer()
    assert n.type == "output"
    assert isinstance(n.inputs, list)
    assert len(n.inputs) == 1
    assert n.outputs is None


# ============================================================
# make_layer default type
# ============================================================

def test_make_layer_default_type_op():
    """When type is omitted, default is 'op'."""
    layer = make_layer({
        "op": "relu",
        "inputs": ["x"],
        "outputs": ["y"],
    })
    assert layer.type == "op"
    assert isinstance(layer, OpLayer)


# ============================================================
# GraphLayer add_layer / get_layer / require_layer
# ============================================================

def test_graphlayer_add_layer_by_kwargs():
    """Add a layer via add_layer(name, None, **kwargs). Start with in, out consumes in; then add mid (consumes in, produces mid)."""
    g = make_layer({
        "type": "graph",
        "layers": {
            "in": {"type": "input"},
            "out": {"type": "output", "inputs": ["in"]},
        },
    })
    g.add_layer("mid", type="op", op="relu", inputs=["in"], outputs=None)
    assert "mid" in g.layers
    assert g.layers["mid"].type == "op"
    g.validate()


def test_graphlayer_add_layer_with_explicit_op_params():
    """add_layer accepts explicit op params; use set_layer_inputs to rewire so graph is in->conv->out."""
    g = make_layer({
        "type": "graph",
        "layers": {
            "in": {"type": "input"},
            "out": {"type": "output", "inputs": ["in"]},
        },
    })
    g.add_layer(
        "conv",
        type="op",
        op={"op_id": "conv2d", "in_channel": 3, "out_channel": 16, "kernel": [3, 3]},
        inputs=["in"],
    )
    g.set_layer_inputs("out", ["conv"])
    assert g.layers["conv"].op.op_id == "conv2d"
    assert g.layers["conv"].op.in_channel == 3
    assert g.layers["conv"].op.out_channel == 16
    assert str(g.layers["out"].inputs[0].ref) == "conv"
    g.validate()


def test_graphlayer_add_layer_duplicate_raises():
    """add_layer with duplicate name must raise ValueError."""
    g = make_layer({
        "type": "graph",
        "layers": {
            "in": {"type": "input"},
            "out": {"type": "output", "inputs": ["in"]},
        },
    })
    with pytest.raises(ValueError, match="already exists"):
        g.add_layer("in", type="input")


def test_graphlayer_add_layer_invalid_name_raises():
    """Invalid layer name (NameSegment parse failure) must raise."""
    g = make_layer({
        "type": "graph",
        "layers": {
            "in": {"type": "input"},
            "out": {"type": "output", "inputs": ["in"]},
        },
    })
    with pytest.raises(InvalidNameError):
        g.add_layer("invalid-name!", type="op", op="relu", inputs=["in"])


def test_graphlayer_get_layer_by_ref():
    """get_layer(ref) resolves sublayer. Build graph with inputs = layer names; outputs default."""
    g = make_layer({
        "type": "graph",
        "layers": {
            "in": {"type": "input"},
            "mid": {"type": "op", "op": "relu", "inputs": ["in"]},
            "out": {"type": "output", "inputs": ["mid"]},
        },
    })
    assert g.get_layer("mid") is g.layers["mid"]
    assert g.get_layer("in").type == "input"


def test_graphlayer_get_layer_missing_returns_none():
    """get_layer returns None for missing ref (get_ref behavior)."""
    g = make_layer({
        "type": "graph",
        "layers": {
            "in": {"type": "input"},
            "out": {"type": "output", "inputs": ["in"]},
        },
    })
    assert g.get_layer("nonexistent") is None


def test_graphlayer_require_layer_missing_raises():
    """require_layer must raise RefResolutionError for missing ref."""
    from ..ref import RefResolutionError
    g = make_layer({
        "type": "graph",
        "layers": {
            "in": {"type": "input"},
            "out": {"type": "output", "inputs": ["in"]},
        },
    })
    with pytest.raises(RefResolutionError):
        g.require_layer("nonexistent")


# ============================================================
# InputLayer / OutputLayer validation
# ============================================================

def test_inputlayer_has_inputs_raises():
    """InputLayer must not have inputs."""
    with pytest.raises(ValidationError):
        make_layer({
            "type": "input",
            "inputs": ["x"],
            "outputs": ["out"],
        })


def test_inputlayer_no_outputs_raises():
    """InputLayer must have at least one output."""
    with pytest.raises(ValidationError):
        make_layer({
            "type": "input",
            "outputs": [],
        })


def test_outputlayer_no_inputs_raises():
    """OutputLayer must have at least one input."""
    with pytest.raises(ValidationError):
        make_layer({
            "type": "output",
            "inputs": [],
        })


def test_outputlayer_has_outputs_raises():
    """OutputLayer must not have outputs."""
    with pytest.raises(ValidationError):
        make_layer({
            "type": "output",
            "inputs": ["inp"],
            "outputs": ["out"],
        })


# ============================================================
# iter_inputs / iter_outputs / iter_weights
# ============================================================

def test_irlayer_iter_inputs_outputs():
    """iter_inputs / iter_outputs yield (ref name, DataDef) pairs."""
    layer = make_layer({
        "type": "op",
        "op": "relu",
        "inputs": ["x"],
        "outputs": ["y"],
    })
    names_in = [n for n, _ in layer.iter_inputs()]
    names_out = [n for n, _ in layer.iter_outputs()]
    assert names_in == ["x"]
    assert names_out == ["y"]


def test_irlayer_iter_weights_empty():
    """iter_weights yields nothing when there are no weights."""
    layer = make_layer({
        "type": "op",
        "op": "relu",
        "inputs": ["x"],
        "outputs": ["y"],
    })
    assert list[Tuple[str, DataDef]](layer.iter_weights()) == []


# ============================================================
# Clone
# ============================================================

def test_layer_clone():
    """clone(**overrides) returns a new instance with overridden attributes."""
    layer = make_layer({
        "type": "op",
        "op": "relu",
        "inputs": ["x"],
        "outputs": ["y"],
    })
    cloned = layer.clone()
    assert cloned is not layer
    assert cloned.type == layer.type
    assert cloned.op.op_id == layer.op.op_id

    cloned2 = layer.clone(inputs=[{"ref": "other"}])
    assert cloned2.inputs[0].ref is not None
    assert str(cloned2.inputs[0].ref) == "other"


# ============================================================
# BlockLayer is_single
# ============================================================

def test_blocklayer_is_single_true():
    """is_single() is True when repeat==1."""
    b = make_layer({
        "type": "block",
        "repeat": 1,
        "layers": {
            "in": {"type": "input"},
            "out": {"type": "output", "inputs": ["in"]},
        },
    })
    assert b.is_single() is True


# ============================================================
# OpLayer binary op (two inputs)
# ============================================================

def test_oplayer_binary_op_two_inputs():
    """BinaryOp subclass (e.g. add) requires exactly two inputs."""
    layer = make_layer({
        "type": "op",
        "op": "add",
        "inputs": ["a", "b"],
        "outputs": ["y"],
    })
    layer.validate()
    assert len(layer.inputs) == 2
    assert layer.op.num_inputs == 2


def test_oplayer_binary_op_wrong_input_count_raises():
    """Validation fails when BinaryOp does not have exactly two inputs."""
    with pytest.raises(ValidationError):
        make_layer({
            "type": "op",
            "op": "add",
            "inputs": ["a"],
            "outputs": ["y"],
        })
