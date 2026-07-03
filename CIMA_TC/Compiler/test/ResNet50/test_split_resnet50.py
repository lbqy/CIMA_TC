"""
Test: run XB-based split pass for ResNet50 and verify exports.

Run from repo root:
    python -m CIMA_TC.Compiler.test.ResNet50.test_split_resnet50
"""
from __future__ import annotations

from typing import Any

import os

from ...frontend import ConvertONNX, ConversionConfig
from ...mapper.xb_split import XBConfig
from ...mapper.split_pass import export_split_model
from ...IR_tool.core.ir import BaseIR
from ...IR_tool.core.visualize import to_dot, render_ir
from ...backend.to_training_code.gen_code import gen_pytorch_model_script, gen_train_script


def _shape_tuple(x) -> tuple:
    try:
        return tuple[int, ...](int(i) for i in x)
    except Exception:
        return tuple[Any, ...]()


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))

    onnx_path = os.path.join(script_dir, "resnet50.onnx")
    ir_out = os.path.join(script_dir, "ResNet50_ir.yaml")

    split_ir_out = os.path.join(script_dir, "ResNet50_split_ir.yaml")
    split_weights_pt = os.path.join(script_dir, "ResNet50_split_weights.pt")

    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    # 1) Frontend: ONNX -> IR (weights are attached to the IR object in memory)
    cfg = ConversionConfig(
        onnx_file=onnx_path,
        ir_file=ir_out,
        fix_layer_name=True,
        store_intermediate_model=False,
    )
    converter = ConvertONNX(cfg)
    print(f"Converting ONNX -> IR: {onnx_path}")
    ir = converter.convert()
    converter.dump(ir_file=ir_out, as_yaml=True)

    assert os.path.isfile(ir_out), f"Missing IR output: {ir_out}"

    # 2) Run split pass using weights attached on the IR object
    xb = XBConfig(rows=576, cols=128, max_xbs=4)
    print("Running XB-based split mapping ...")
    export_split_model(
        ir,
        xb,
        # Default split_bn=True: keep BN adjacent to split conv/fc.
        ir_file=split_ir_out,
        weight_file=split_weights_pt,
    )

    assert os.path.isfile(split_ir_out), f"Missing split IR output: {split_ir_out}"
    assert os.path.isfile(split_weights_pt), f"Missing split weights output: {split_weights_pt}"

    # 3) Verify: split IR contains at least one Concat_0_* or *_0_*
    split_ir = BaseIR.load_ir(file=split_ir_out)
    layer_names = list[str]((split_ir.layers or {}).keys())
    has_split = any(n.startswith("Concat_0_") or "_0_" in n for n in layer_names)
    assert has_split, "Split IR does not seem to contain split layers (Concat_0_* or *_0_*)"

    # 3.5) Visualize: export DOT + PDF
    dot_out = os.path.join(script_dir, "ResNet50_split_graph.dot")
    pdf_out = os.path.join(script_dir, "ResNet50_split_graph.pdf")
    with open(dot_out, "w", encoding="utf-8") as f:
        f.write(to_dot(split_ir, rankdir="TB"))
    try:
        render_ir(split_ir, pdf_out, format="pdf", rankdir="TB")
    except Exception as e:
        raise RuntimeError(
            "Failed to render PDF via Graphviz. "
            "You can still use the exported DOT file. "
            "To enable PDF rendering, install Graphviz and python-graphviz."
        ) from e
    assert os.path.isfile(pdf_out), f"Missing split graph PDF output: {pdf_out}"

    # 3.6) Export a PyTorch model script (nn.Module) from split IR
    model_py = os.path.join(script_dir, "resnet50_from_split_ir_model.py")
    gen_pytorch_model_script(ir_path=split_ir_out, out_py=model_py, class_name="ResNet50SplitIR")
    assert os.path.isfile(model_py), f"Missing generated model script: {model_py}"

    # 3.7) Export a PyTorch training script from split IR + weights
    train_py = os.path.join(script_dir, "train_resnet50_from_split_ir.py")
    gen_train_script(
        ir_path=split_ir_out,
        weights_path=split_weights_pt,
        out_py=train_py,
        example_shape="1,3,224,224",
        lr=1e-3,
        steps=1,
    )
    assert os.path.isfile(train_py), f"Missing generated training script: {train_py}"
    print(f"Generated training script: {train_py}")

    # 4) Verify: every op layer with weights has matching tensor in split weights
    import torch

    split_weights: dict[str, torch.Tensor] = torch.load(split_weights_pt, map_location="cpu")
    missing = []
    bad_shape = []
    for lname, layer in (split_ir.layers or {}).items():
        w = getattr(layer, "weights", None)
        if not w or "weight" not in w:
            continue
        spec_shape = list[Any](getattr(w["weight"], "shape", []) or [])
        if not spec_shape:
            continue
        key = f"{lname}.weight"
        if key not in split_weights:
            missing.append(key)
            continue
        tensor = split_weights[key]
        if _shape_tuple(getattr(tensor, "shape", ())) != _shape_tuple(spec_shape):
            bad_shape.append((key, tuple[Any, ...](spec_shape), tuple(getattr(tensor, "shape", ())))
            )

    assert not missing, f"Missing split weight tensors: {missing[:10]} (total {len(missing)})"
    assert not bad_shape, f"Split weight shape mismatches: {bad_shape[:5]} (total {len(bad_shape)})"

    print("ResNet50 split export verification OK")


if __name__ == "__main__":
    main()

