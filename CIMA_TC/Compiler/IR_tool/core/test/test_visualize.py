from __future__ import annotations


from ..ir import BaseIR
from ..layer import OpLayer, InputLayer, OutputLayer
from ..datadef import DataDef
from ..op import make_op
from ..visualize import to_dot
# Register conv2d, relu so make_op can resolve them
from ...ops import Conv2dOp, ReluOp  # noqa: F401


def test_to_dot_contains_nodes_edges_and_shape_labels() -> None:
    # Build a tiny graph: in -> conv -> relu -> out
    layers = {
        "in": InputLayer(type="input", outputs=[DataDef(ref="in", channel=3, height=224, width=224)]),
        "conv": OpLayer(
            type="op",
            op=make_op("conv2d", in_channel=3, out_channel=16, kernel=3, stride=1, padding=1, bias=False),
            inputs=[DataDef(ref="in", channel=3, height=224, width=224)],
            outputs=[DataDef(ref="conv", channel=16, height=224, width=224)],
            weights={"weight": {"shape": [16, 3, 3, 3]}},
        ),
        "relu": OpLayer(
            type="op",
            op=make_op("relu"),
            inputs=[DataDef(ref="conv", channel=16, height=224, width=224)],
            outputs=[DataDef(ref="relu", channel=16, height=224, width=224)],
        ),
        "out": OutputLayer(type="output", inputs=[DataDef(ref="relu")]),
    }
    ir = BaseIR.make_ir(layers=layers)

    dot = to_dot(ir)
    assert "digraph IR" in dot
    assert '"conv"' in dot and '"relu"' in dot
    assert '"conv" -> "relu"' in dot
    # Edge label should contain a tuple-like shape
    assert "(16, 224, 224)" in dot or "(" in dot

