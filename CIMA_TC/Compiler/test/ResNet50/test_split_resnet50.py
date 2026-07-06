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


def _max_mean_abs(a: Any, b: Any) -> tuple[float, float]:
    import numpy as np

    diff = np.abs(a - b)
    return float(diff.max()), float(diff.mean())


def _assert_outputs_close(name: str, expected: Any, actual: Any, *, atol: float = 1e-4, rtol: float = 1e-4) -> None:
    import numpy as np

    if expected.shape != actual.shape:
        raise AssertionError(f"{name} shape mismatch: expected {expected.shape}, got {actual.shape}")
    try:
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    except AssertionError as exc:
        max_abs, mean_abs = _max_mean_abs(expected, actual)
        raise AssertionError(f"{name} mismatch: max_abs={max_abs:.6g}, mean_abs={mean_abs:.6g}") from exc


def _make_inference_onnx(src: str, dst: str) -> None:
    """Write an inference-mode ONNX copy by pruning BatchNormalization training outputs."""
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("onnx package is required for split equivalence verification") from exc

    model = onnx.load(src)
    changed = False
    for node in model.graph.node:
        if node.op_type == "BatchNormalization" and len(node.output) > 1:
            del node.output[1:]
            changed = True
    if changed:
        onnx.save(model, dst)
    else:
        onnx.save(model, dst)


def _run_onnx(onnx_path: str, x_np: Any) -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("onnxruntime is required for split equivalence verification") from exc

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: x_np})[0]


def _random_input_for_onnx(onnx_path: str, *, seed: int = 0) -> Any:
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    shape = []
    for dim in session.get_inputs()[0].shape:
        shape.append(1 if not isinstance(dim, int) else dim)
    rng = np.random.default_rng(seed)
    return rng.standard_normal(tuple(shape), dtype=np.float32)


def _load_generated_split_model(model_py: str, weights_pt: str) -> Any:
    import importlib.util
    import torch

    spec = importlib.util.spec_from_file_location("resnet50_split_generated_equivalence", model_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import generated model script: {model_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.ResNet50SplitIR()
    weights = torch.load(weights_pt, map_location="cpu")
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing or unexpected:
        raise AssertionError(
            f"Generated split model state_dict mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    return model


def _verify_split_equivalence(
    *,
    original_onnx: str,
    generated_model_py: str,
    split_weights_pt: str,
    reference_onnx: str,
    split_onnx: str,
) -> None:
    """Compare original inference ONNX, split ONNX, and generated PyTorch outputs."""
    import torch

    _make_inference_onnx(original_onnx, reference_onnx)
    x_np = _random_input_for_onnx(reference_onnx, seed=0)

    y_original = _run_onnx(reference_onnx, x_np)

    model = _load_generated_split_model(generated_model_py, split_weights_pt)
    model.eval()
    with torch.no_grad():
        y_torch = model(torch.from_numpy(x_np)).detach().cpu().numpy()

    torch.onnx.export(
        model,
        torch.from_numpy(x_np),
        split_onnx,
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
        do_constant_folding=True,
    )
    y_split_onnx = _run_onnx(split_onnx, x_np)

    _assert_outputs_close("original ONNX vs generated split PyTorch", y_original, y_torch)
    _assert_outputs_close("generated split PyTorch vs split ONNX", y_torch, y_split_onnx)
    _assert_outputs_close("original ONNX vs split ONNX", y_original, y_split_onnx)

    pt_max, pt_mean = _max_mean_abs(y_original, y_torch)
    onnx_max, onnx_mean = _max_mean_abs(y_original, y_split_onnx)
    print(
        "Split equivalence OK "
        f"(PyTorch max={pt_max:.3g}, mean={pt_mean:.3g}; "
        f"ONNX max={onnx_max:.3g}, mean={onnx_mean:.3g})"
    )


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

    # 3) Verify: split IR uses topology-renamed layers and contains split/merge ops.
    split_ir = BaseIR.load_ir(file=split_ir_out)
    layers = split_ir.layers or {}
    layer_names = list[str](layers.keys())
    def has_legacy_grid_suffix(name: str) -> bool:
        parts = name.rsplit("_", 2)
        return len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit()

    legacy_split_names = [
        n
        for n in layer_names
        if n.startswith(("Concat_0_", "Split_0_"))
        or has_legacy_grid_suffix(n)
        or (n.startswith("Add_") and len(n.split("_")) > 2)
    ]
    assert not legacy_split_names, f"Split IR still contains legacy split names: {legacy_split_names[:10]}"
    split_or_merge_ops = [
        n
        for n, layer in layers.items()
        if getattr(getattr(layer, "op", None), "op_id", None) in ("split", "concat", "add")
    ]
    assert split_or_merge_ops, "Split IR does not contain split/merge ops"

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

    reference_onnx = os.path.join(script_dir, "ResNet50_inference.onnx")
    split_onnx = os.path.join(script_dir, "ResNet50_split.onnx")
    _verify_split_equivalence(
        original_onnx=onnx_path,
        generated_model_py=model_py,
        split_weights_pt=split_weights_pt,
        reference_onnx=reference_onnx,
        split_onnx=split_onnx,
    )

    print("ResNet50 split export verification OK")


if __name__ == "__main__":
    main()

