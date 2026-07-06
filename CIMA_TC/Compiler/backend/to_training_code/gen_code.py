from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _safe_name(name: str) -> str:
    return name.replace(".", "_").replace("-", "_")


def _weight_shape(layer: Any, name: str = "weight") -> Optional[Tuple[int, ...]]:
    weights = getattr(layer, "weights", None) or {}
    spec = weights.get(name)
    shape = getattr(spec, "shape", None)
    if not shape:
        return None
    try:
        return tuple(int(x) for x in shape)
    except Exception:
        return None


def _has_weight(layer: Any, name: str) -> bool:
    return name in (getattr(layer, "weights", None) or {})


def gen_pytorch_model_script(
    *,
    ir_path: str | Path,
    out_py: str | Path,
    class_name: str = "IRModel",
) -> None:
    """
    Generate a PyTorch model definition script (format similar to ResNet50_example.py):
    - explicit submodules in __init__ (self.Conv_0 = nn.Conv2d(...))
    - forward uses named tensors x_<layer>
    - relies on PyTorch state_dict loading (keys match layer names)
    """
    from CIMA_TC.Compiler.IR_tool.core.ir import BaseIR

    ir = BaseIR.load_ir(file=str(ir_path))
    layers = ir.layers or {}
    topo = ir.topological_order()

    # Map a ref like "A" or "A:1" -> variable name
    var_map: Dict[str, str] = {}
    # graph_input outputs
    g_in = layers.get("graph_input")
    g_outs = getattr(g_in, "outputs", None) or []
    for i in range(len(g_outs)):
        var_map[f"graph_input:{i}"] = f"x_graph_input_{i}"

    def var_for_ref(ref: str) -> str:
        seg = ref.split(".", 1)[0]
        if ":" in seg:
            name, idx = seg.split(":", 1)
            key = f"{name}:{idx}"
        else:
            key = seg
        if key in var_map:
            return var_map[key]
        # default for single-output layers
        if ":" not in key:
            return f"x_{_safe_name(key)}"
        return f"x_{_safe_name(key.replace(':', '_'))}"

    init_lines: List[str] = []
    fwd_lines: List[str] = []

    # Pre-create var_map for op outputs so split branches can be referenced.
    for name in topo:
        layer = layers.get(name)
        if layer is None or getattr(layer, "type", None) != "op":
            continue
        outs = getattr(layer, "outputs", None) or []
        if len(outs) > 1:
            for j in range(len(outs)):
                var_map[f"{name}:{j}"] = f"x_{_safe_name(name)}_{j}"
        else:
            var_map[name] = f"x_{_safe_name(name)}"

    for name in topo:
        layer = layers.get(name)
        if layer is None:
            continue
        ltype = getattr(layer, "type", None)
        if ltype != "op":
            continue
        op = getattr(layer, "op", None)
        op_id = getattr(op, "op_id", None)
        if not op_id:
            continue

        # Inputs
        ins = getattr(layer, "inputs", None) or []
        in_vars = [var_for_ref(str(dd.ref)) for dd in ins if dd.ref is not None]

        # Outputs
        out_vars: List[str] = []
        outs = getattr(layer, "outputs", None) or []
        if len(outs) > 1:
            out_vars = [var_map[f"{name}:{j}"] for j in range(len(outs))]
        else:
            out_vars = [var_map[name]]

        if op_id == "conv2d":
            groups = int(getattr(op, "groups", getattr(op, "group", 1)))
            w_shape = _weight_shape(layer)
            if w_shape and len(w_shape) >= 4:
                out_ch = int(w_shape[0])
                in_ch = int(w_shape[1]) * groups
                k_tuple = tuple(int(v) for v in w_shape[2:4])
                k = k_tuple[0] if k_tuple[0] == k_tuple[1] else k_tuple
            else:
                in_ch = int(getattr(op, "in_channel"))
                out_ch = int(getattr(op, "out_channel"))
                k = int(getattr(op, "kernel", 3))
            stride = int(getattr(op, "stride", 1))
            pad = int(getattr(op, "padding", 0))
            dil = int(getattr(op, "dilation", 1))
            bias = _has_weight(layer, "bias") or bool(getattr(op, "bias", False))
            init_lines.append(
                f"self.{name} = nn.Conv2d({in_ch}, {out_ch}, {k!r}, {stride}, {pad}, dilation={dil}, groups={groups}, bias={bias})"
            )
            fwd_lines.append(f"{out_vars[0]} = self.{name}({in_vars[0]})")

        elif op_id in ("batch_norm", "batch_norm1d", "batch_norm2d", "batch_norm3d"):
            ch_shape = _weight_shape(layer) or _weight_shape(layer, "bias") or _weight_shape(layer, "running_mean")
            ch = int(ch_shape[0]) if ch_shape else int(getattr(op, "channel"))
            eps = float(getattr(op, "epsilon", 1e-5))
            init_lines.append(f"self.{name} = nn.BatchNorm2d({ch}, eps={eps})")
            fwd_lines.append(f"{out_vars[0]} = self.{name}({in_vars[0]})")

        elif op_id == "linear":
            w_shape = _weight_shape(layer)
            if w_shape and len(w_shape) >= 2:
                out_ch = int(w_shape[0])
                in_ch = int(w_shape[1])
            else:
                in_ch = int(getattr(op, "in_channel"))
                out_ch = int(getattr(op, "out_channel"))
            bias = _has_weight(layer, "bias") or bool(getattr(op, "bias", False))
            init_lines.append(f"self.{name} = nn.Linear({in_ch}, {out_ch}, bias={bias})")
            fwd_lines.append(f"{out_vars[0]} = self.{name}({in_vars[0]})")

        elif op_id == "relu":
            init_lines.append(f"self.{name} = nn.ReLU()")
            fwd_lines.append(f"{out_vars[0]} = self.{name}({in_vars[0]})")

        elif op_id == "max_pool2d":
            k = int(getattr(op, "kernel", 2))
            stride = int(getattr(op, "stride", k))
            pad = int(getattr(op, "padding", 0))
            init_lines.append(f"self.{name} = nn.MaxPool2d({k}, {stride}, {pad})")
            fwd_lines.append(f"{out_vars[0]} = self.{name}({in_vars[0]})")

        elif op_id == "avg_pool2d":
            k = int(getattr(op, "kernel", 2))
            stride = int(getattr(op, "stride", k))
            pad = int(getattr(op, "padding", 0))
            init_lines.append(f"self.{name} = nn.AvgPool2d({k}, {stride}, {pad})")
            fwd_lines.append(f"{out_vars[0]} = self.{name}({in_vars[0]})")

        elif op_id == "global_avg_pool2d":
            # keep functional, no params
            fwd_lines.append(f"{out_vars[0]} = F.adaptive_avg_pool2d({in_vars[0]}, output_size=(1, 1))")

        elif op_id == "flatten":
            start_dim = int(getattr(op, "start_dim", 1))
            fwd_lines.append(f"{out_vars[0]} = torch.flatten({in_vars[0]}, start_dim={start_dim})")

        elif op_id == "split":
            axis = int(getattr(op, "axis", 1))
            split = getattr(op, "split", None)
            if split is None:
                raise ValueError(f"split layer {name} missing split attr")
            sizes = list(split) if isinstance(split, (list, tuple)) else [int(split)]
            fwd_lines.append(
                f"{', '.join(out_vars)} = torch.split({in_vars[0]}, {sizes}, dim={axis})"
            )

        elif op_id == "concat":
            axis = int(getattr(op, "axis", 1))
            fwd_lines.append(f"{out_vars[0]} = torch.cat([{', '.join(in_vars)}], dim={axis})")

        elif op_id == "add":
            fwd_lines.append(f"{out_vars[0]} = torch.add({in_vars[0]}, {in_vars[1]})")

        elif op_id == "mul":
            fwd_lines.append(f"{out_vars[0]} = {in_vars[0]} * {in_vars[1]}")

        elif op_id == "sigmoid":
            fwd_lines.append(f"{out_vars[0]} = torch.sigmoid({in_vars[0]})")

        elif op_id == "silu":
            fwd_lines.append(f"{out_vars[0]} = F.silu({in_vars[0]})")

        else:
            raise NotImplementedError(f"gen_pytorch_model_script: unsupported op_id={op_id!r} layer={name!r}")

    # Output layer
    out_layer = layers.get("graph_output")
    out_ins = getattr(out_layer, "inputs", None) or []
    out_vars = [var_for_ref(str(dd.ref)) for dd in out_ins if dd.ref is not None]
    ret_stmt = f"return {out_vars[0]}" if len(out_vars) == 1 else f"return ({', '.join(out_vars)})"

    # Forward signature (assume 1 input tensor for now; ResNet50 has one)
    fwd_args = ", ".join(var_map[f"graph_input:{i}"] for i in range(len(g_outs)))
    if not fwd_args:
        fwd_args = "x_graph_input_0"

    code = (
        "# **************************************** #\n"
        "# This file is automatically generated !!! #\n"
        "#        Please do not modify it !!!      #\n"
        "# **************************************** #\n\n"
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n\n"
        f"class {class_name}(nn.Module):\n\n"
        "    def __init__(self, **kwargs):\n"
        f"        super({class_name}, self).__init__()\n\n"
        + "\n".join(f"        {s}" for s in init_lines)
        + "\n\n"
        f"    def forward(self, {fwd_args}):\n"
        + "\n".join(f"        {s}" for s in fwd_lines)
        + "\n"
        f"        {ret_stmt}\n"
    )

    Path(out_py).write_text(code, encoding="utf-8")


def gen_train_script(
    *,
    ir_path: str | Path,
    weights_path: str | Path,
    out_py: str | Path,
    example_shape: str = "1,3,224,224",
    lr: float = 1e-3,
    steps: int = 1,
) -> None:
    """
    Generate a minimal runnable training script:
    - load IR
    - build torch model
    - load exported weights dict
    - run a few dummy training steps on random input
    """
    ir_path = str(ir_path)
    weights_path = str(weights_path)
    out_py = str(out_py)

    out_dir = Path(out_py).resolve().parent

    def script_local_path(path_value: str) -> str:
        path = Path(path_value)
        try:
            return str(path.resolve().relative_to(out_dir))
        except ValueError:
            return str(path)

    ir_ref = script_local_path(ir_path)
    weights_ref = script_local_path(weights_path)

    code = f'''"""
Auto-generated training script from IR.

IR: {ir_ref}
Weights: {weights_ref}
"""

from __future__ import annotations

from pathlib import Path

import torch

import CIMA_TC.Compiler.IR_tool.ops  # Registers built-in op classes for IR loading.
from CIMA_TC.Compiler.IR_tool.core.ir import BaseIR
from CIMA_TC.Compiler.backend.to_training_code.ir_to_torch import (
    build_torch_model_from_ir,
    load_weights_file,
    load_weights_into_model,
)


def _parse_shape(s: str):
    parts = [int(x) for x in s.split(",") if x.strip()]
    if len(parts) != 4:
        raise ValueError("example_shape must be 'N,C,H,W'")
    return parts


_SCRIPT_DIR = Path(__file__).resolve().parent


def main():
    ir = BaseIR.load_ir(file=str(_SCRIPT_DIR / {ir_ref!r}))
    built = build_torch_model_from_ir(ir)
    model = built.model
    weights = load_weights_file(str(_SCRIPT_DIR / {weights_ref!r}))
    load_weights_into_model(model, weights, module_name_map=built.module_name_map, strict=False)

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr={lr})

    n, c, h, w = _parse_shape({example_shape!r})
    x = torch.randn(n, c, h, w)

    for i in range({steps}):
        y = model(x)
        # dummy scalar loss
        loss = (y.float() ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"step={{i}} loss={{loss.item():.6f}}")


if __name__ == "__main__":
    main()
'''

    Path(out_py).write_text(code, encoding="utf-8")


__all__ = ["gen_pytorch_model_script", "gen_train_script"]

