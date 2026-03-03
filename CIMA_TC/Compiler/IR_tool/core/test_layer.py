import pytest

from .layer import (
    IRNode,
    OpNode,
    GraphNode,
    BlockNode,
    InputNode,
    OutputNode,
    make_node,
)
from .type_utils import ValidationError
from .op import BaseOp, UnaryOp

# ============================================================
# Helpers
# ============================================================

def dd():
    """Minimal valid datadef payload"""
    return {}


def input_node():
    return make_node({
        "type": "input",
        "outputs": {"out": dd()}
    })


def output_node():
    return make_node({
        "type": "output",
        "inputs": {"inp": dd()}
    })

class ReluOp(UnaryOp):
    op_id = "relu"

# ============================================================
# Basic node creation
# ============================================================

def test_basic_irnode_creation():

    node = make_node({
        "type": "op",
        "op": "relu",
        "inputs": {"x": dd()},
        "outputs": {"y": dd()}
    })

    node.validate()


# ============================================================
# OpNode validation
# ============================================================

def test_opnode_invalid_input_number():

    with pytest.raises(ValidationError):
        node = make_node({
            "type": "op",
            "op": "relu",
            "inputs": {},   # empty
            "outputs": {"y": dd()}
    })

    # with pytest.raises(ValidationError):
        node.validate()


# ============================================================
# GraphNode
# ============================================================

def test_graphnode_add_and_validate():

    g = make_node({
        "type": "graph",
        "nodes": {
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


def test_graphnode_missing_input():

    with pytest.raises(ValidationError):
        g = make_node({
            "type": "graph",
            "nodes": {
                "out": {
                    "type": "output",
                    "inputs": {"inp": dd()}
                }
            }
        })

    # with pytest.raises(ValidationError):
    #     g.validate()


def test_graphnode_missing_output():

    with pytest.raises(ValidationError):
        g = make_node({
            "type": "graph",
            "nodes": {
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

def test_graphnode_multi_io():

    g = make_node({
        "type": "graph",
        "nodes": {
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

    g = make_node({
        "type": "graph",
        "nodes": {
            "in": {
                "type": "input",
                "outputs": {"o": dd()}
            },
            "sub": {
                "type": "graph",
                "nodes": {
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
# BlockNode
# ============================================================

def test_blocknode_repeat():

    b = make_node({
        "type": "block",
        "repeat": 3,
        "nodes": {
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


def test_blocknode_invalid_repeat():

    with pytest.raises(ValidationError):
        b = make_node({
            "type": "block",
            "repeat": 0,
            "nodes": {
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
# OpNode attribute access
# ============================================================

def test_opnode_attribute_access():

    node = make_node({
        "type": "op",
        "op": "relu",
        "inputs": {"x": dd()},
        "outputs": {"y": dd()}
    })

    # ---- basic attributes ----
    assert node.type == "op"
    assert isinstance(node.op, ReluOp) 

    # ---- inputs / outputs ----
    assert isinstance(node.inputs, dict)
    assert "x" in node.inputs
    assert isinstance(node.inputs["x"], type(node.inputs["x"]))

    assert isinstance(node.outputs, dict)
    assert "y" in node.outputs

    # ---- weights default ----
    assert node.weights is None

    # ---- subgraph capability ----
    assert node.has_subgraph() is False
    assert list(node.iter_subnodes()) == []


# ============================================================
# GraphNode attribute access
# ============================================================

def test_graphnode_attribute_access():

    g = make_node({
        "type": "graph",
        "nodes": {
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
    assert isinstance(g.nodes, dict)
    assert "in" in g.nodes
    assert "out" in g.nodes

    # subgraph capability
    assert g.has_subgraph() is True
    assert len(list(g.iter_subnodes())) == 2

    # child node types
    assert isinstance(g.nodes["in"], InputNode)
    assert isinstance(g.nodes["out"], OutputNode)


# ============================================================
# BlockNode attribute access
# ============================================================

def test_blocknode_attribute_access():

    b = make_node({
        "type": "block",
        "repeat": 2,
        "nodes": {
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
    assert isinstance(b.nodes, dict)
    assert b.has_subgraph() is True


# ============================================================
# IO node restrictions
# ============================================================

def test_inputnode_access():

    n = input_node()

    assert n.type == "input"
    assert n.inputs is None
    assert isinstance(n.outputs, dict)


def test_outputnode_access():

    n = output_node()

    assert n.type == "output"
    assert isinstance(n.inputs, dict)
    assert n.outputs is None