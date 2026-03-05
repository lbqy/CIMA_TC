import pytest

from .layer import (
    IRLayer,
    OpLayer,
    GraphLayer,
    BlockLayer,
    InputLayer,
    OutputLayer,
    make_layer,
)
from .type_utils import ValidationError
from .op import BaseOp, UnaryOp

# ============================================================
# Helpers
# ============================================================

def dd():
    """Minimal valid datadef payload"""
    return {}


def input_layer():
    return make_layer({
        "type": "input",
        "outputs": {"out": dd()}
    })


def output_layer():
    return make_layer({
        "type": "output",
        "inputs": {"inp": dd()}
    })

class ReluOp(UnaryOp):
    op_id = "relu"

# ============================================================
# Basic layer creation
# ============================================================

def test_basic_irlayer_creation():

    layer = make_layer({
        "type": "op",
        "op": "relu",
        "inputs": {"x": dd()},
        "outputs": {"y": dd()}
    })

    layer.validate()


# ============================================================
# OpLayer validation
# ============================================================

def test_oplayer_invalid_input_number():

    with pytest.raises(ValidationError):
        layer = make_layer({
            "type": "op",
            "op": "relu",
            "inputs": {},   # empty
            "outputs": {"y": dd()}
    })

    # with pytest.raises(ValidationError):
        layer.validate()


# ============================================================
# GraphLayer
# ============================================================

def test_graphlayer_add_and_validate():

    g = make_layer({
        "type": "graph",
        "layers": {
            "in": {
                "type": "input",
                "outputs": {"out": dd()}
            },
            "out": {
                "type": "output",
                "inputs": {"inp": dd()}
            }
        }
    })

    g.validate()


def test_graphlayer_missing_input():

    with pytest.raises(ValidationError):
        g = make_layer({
            "type": "graph",
            "layers": {
                "out": {
                    "type": "output",
                    "inputs": {"inp": dd()}
                }
            }
        })

    # with pytest.raises(ValidationError):
    #     g.validate()


def test_graphlayer_missing_output():

    with pytest.raises(ValidationError):
        g = make_layer({
            "type": "graph",
            "layers": {
                "in": {
                    "type": "input",
                    "outputs": {"out": dd()}
                }
            }
        })

    # with pytest.raises(ValidationError):
    #     g.validate()


# ============================================================
# Multi IO
# ============================================================

def test_graphlayer_multi_io():

    g = make_layer({
        "type": "graph",
        "layers": {
            "in1": {
                "type": "input",
                "outputs": {"o1": dd()}
            },
            "in2": {
                "type": "input",
                "outputs": {"o2": dd()}
            },
            "out1": {
                "type": "output",
                "inputs": {"i1": dd()}
            },
            "out2": {
                "type": "output",
                "inputs": {"i2": dd()}
            }
        }
    })

    g.validate()


# ============================================================
# Nested graph
# ============================================================

def test_nested_graph():

    g = make_layer({
        "type": "graph",
        "layers": {
            "in": {
                "type": "input",
                "outputs": {"o": dd()}
            },
            "sub": {
                "type": "graph",
                "layers": {
                    "in2": {
                        "type": "input",
                        "outputs": {"o2": dd()}
                    },
                    "out2": {
                        "type": "output",
                        "inputs": {"i2": dd()}
                    }
                }
            },
            "out": {
                "type": "output",
                "inputs": {"i": dd()}
            }
        }
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
            "in": {
                "type": "input",
                "outputs": {"o": dd()}
            },
            "out": {
                "type": "output",
                "inputs": {"i": dd()}
            }
        }
    })

    b.validate()


def test_blocklayer_invalid_repeat():

    with pytest.raises(ValidationError):
        b = make_layer({
            "type": "block",
            "repeat": 0,
            "layers": {
                "in": {
                    "type": "input",
                    "outputs": {"o": dd()}
                },
                "out": {
                    "type": "output",
                    "inputs": {"i": dd()}
                }
            }
        })

    # with pytest.raises(ValidationError):
    #     b.validate()

# ============================================================
# OpLayer attribute access
# ============================================================

def test_oplayer_attribute_access():

    layer = make_layer({
        "type": "op",
        "op": "relu",
        "inputs": {"x": dd()},
        "outputs": {"y": dd()}
    })
    
    # ---- basic attributes ----
    assert layer.type == "op"
    assert isinstance(layer.op, ReluOp) 

    # ---- inputs / outputs ----
    assert isinstance(layer.inputs, dict)
    assert "x" in layer.inputs
    assert isinstance(layer.inputs["x"], type(layer.inputs["x"]))

    assert isinstance(layer.outputs, dict)
    assert "y" in layer.outputs

    # ---- weights default ----
    assert layer.weights is None

    # ---- subgraph capability ----
    assert layer.has_subgraph() is False
    assert list(layer.iter_sublayers()) == []


# ============================================================
# GraphLayer attribute access
# ============================================================

def test_graphlayer_attribute_access():

    g = make_layer({
        "type": "graph",
        "layers": {
            "in": {
                "type": "input",
                "outputs": {"o": dd()}
            },
            "out": {
                "type": "output",
                "inputs": {"i": dd()}
            }
        }
    })

    assert g.type == "graph"
    assert isinstance(g.layers, dict)
    assert "in" in g.layers
    assert "out" in g.layers

    # subgraph capability
    assert g.has_subgraph() is True
    assert len(list(g.iter_sublayers())) == 2

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
            "in": {
                "type": "input",
                "outputs": {"o": dd()}
            },
            "out": {
                "type": "output",
                "inputs": {"i": dd()}
            }
        }
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
    assert isinstance(n.outputs, dict)


def test_outputlayer_access():

    n = output_layer()

    assert n.type == "output"
    assert isinstance(n.inputs, dict)
    assert n.outputs is None