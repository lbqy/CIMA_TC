from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:  # pragma: no cover
    raise RuntimeError("ir_to_torch requires torch. Install with: pip install torch") from e

from ...IR_tool.core import BaseIR, DataDef


def _as_tuple2(x: Any) -> Tuple[int, int]:
    if isinstance(x, (tuple, list)) and len(x) == 2:
        return int(x[0]), int(x[1])
    v = int(x)
    return v, v


def _safe_module_key(name: str) -> str:
    # Avoid '.' which is a hierarchy separator in state_dict keys.
    return name.replace(".", "_").replace("-", "_")


def _parse_ref_name_and_index(ref: str) -> Tuple[str, int]:
    # Only handle first segment: "Layer" or "Layer:idx"
    seg = ref.split(".", 1)[0]
    if ":" not in seg:
        return seg, 0
    n, i = seg.split(":", 1)
    try:
        return n, int(i)
    except Exception:
        return n, 0


def _get_layer_outputs_dd(ir: BaseIR, layer_name: str, out_idx: int) -> Optional[DataDef]:
    layers = getattr(ir, "layers", None) or {}
    if layer_name not in layers:
        return None
    outs = getattr(layers[layer_name], "outputs", None) or []
    if 0 <= out_idx < len(outs):
        return outs[out_idx]
    return None


def _dd_shape_tuple(dd: Optional[DataDef]) -> Optional[Tuple[int, ...]]:
    if dd is None:
        return None
    shape = getattr(dd, "shape", None)
    if shape:
        try:
            return tuple(int(x) for x in shape)
        except Exception:
            return None
    # For NCHW-ish tensors in this project, we only track (c,h,w); batch is dynamic.
    c = getattr(dd, "channel", None)
    h = getattr(dd, "height", None)
    w = getattr(dd, "width", None)
    if c is None:
        return None
    c = int(c)
    if h is None or w is None:
        return (c,)
    return (c, int(h), int(w))


def _layer_weight_shape(layer: Any, name: str = "weight") -> Optional[Tuple[int, ...]]:
    weights = getattr(layer, "weights", None) or {}
    spec = weights.get(name)
    shape = getattr(spec, "shape", None)
    if not shape:
        return None
    try:
        return tuple(int(x) for x in shape)
    except Exception:
        return None


def _layer_has_weight(layer: Any, name: str) -> bool:
    return name in (getattr(layer, "weights", None) or {})


@dataclass
class BuildResult:
    model: nn.Module
    # map IR layer name -> safe module key (if a nn.Module was created for this layer)
    module_name_map: Dict[str, str]


class IRModule(nn.Module):
    """
    Execute BaseIR.layers in topological order.

    - op layers with parameters are implemented as nn.Modules stored in self.mods (ModuleDict).
    - pure functional ops use torch ops in forward.
    """

    def __init__(self, ir: BaseIR):
        super().__init__()
        self.ir = ir
        self.mods = nn.ModuleDict()
        self._name_map: Dict[str, str] = {}

        layers = getattr(ir, "layers", None) or {}
        for name in ir.topological_order():
            layer = layers.get(name)
            if layer is None:
                continue
            if getattr(layer, "type", None) != "op":
                continue
            op = getattr(layer, "op", None)
            op_id = getattr(op, "op_id", None)
            if op_id is None:
                continue

            key = _safe_module_key(name)

            if op_id == "conv2d":
                groups = int(getattr(op, "groups", getattr(op, "group", 1)))
                w_shape = _layer_weight_shape(layer)
                if w_shape and len(w_shape) >= 4:
                    out_ch = int(w_shape[0])
                    in_ch = int(w_shape[1]) * groups
                    k = tuple(int(v) for v in w_shape[2:4])
                else:
                    in_ch = int(getattr(op, "in_channel"))
                    out_ch = int(getattr(op, "out_channel"))
                    k = _as_tuple2(getattr(op, "kernel"))
                stride = _as_tuple2(getattr(op, "stride", 1))
                padding = _as_tuple2(getattr(op, "padding", 0))
                dilation = _as_tuple2(getattr(op, "dilation", 1))
                bias = _layer_has_weight(layer, "bias") or bool(getattr(op, "bias", False))
                self.mods[key] = nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=k,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                )
                self._name_map[name] = key

            elif op_id == "linear":
                w_shape = _layer_weight_shape(layer)
                if w_shape and len(w_shape) >= 2:
                    out_ch = int(w_shape[0])
                    in_ch = int(w_shape[1])
                else:
                    in_ch = int(getattr(op, "in_channel"))
                    out_ch = int(getattr(op, "out_channel"))
                bias = _layer_has_weight(layer, "bias") or bool(getattr(op, "bias", False))
                self.mods[key] = nn.Linear(in_features=in_ch, out_features=out_ch, bias=bias)
                self._name_map[name] = key

            elif op_id in ("batch_norm", "batch_norm1d", "batch_norm2d", "batch_norm3d"):
                ch_shape = (
                    _layer_weight_shape(layer)
                    or _layer_weight_shape(layer, "bias")
                    or _layer_weight_shape(layer, "running_mean")
                )
                ch = int(ch_shape[0]) if ch_shape else int(getattr(op, "channel"))
                eps = float(getattr(op, "epsilon", 1e-5))
                # Use BatchNorm2d as default (IR is primarily CNN)
                self.mods[key] = nn.BatchNorm2d(num_features=ch, eps=eps, affine=True, track_running_stats=True)
                self._name_map[name] = key

            # Other ops are functional; no module created.

    @property
    def module_name_map(self) -> Dict[str, str]:
        return dict(self._name_map)

    def forward(self, *inputs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        ir = self.ir
        layers = getattr(ir, "layers", None) or {}
        cache: Dict[str, Any] = {}

        # graph_input provides multiple outputs (one per provided input tensor)
        cache["graph_input"] = list(inputs)

        for name in ir.topological_order():
            layer = layers.get(name)
            if layer is None:
                continue
            ltype = getattr(layer, "type", None)

            if ltype == "input":
                # already handled as graph_input
                continue

            if ltype == "output":
                ins = getattr(layer, "inputs", None) or []
                outs: List[torch.Tensor] = []
                for dd in ins:
                    if dd.ref is None:
                        continue
                    prod, idx = _parse_ref_name_and_index(str(dd.ref))
                    val = cache.get(prod)
                    if isinstance(val, list):
                        outs.append(val[idx])
                    else:
                        outs.append(val)
                if len(outs) == 1:
                    return outs[0]
                return tuple(outs)

            if ltype != "op":
                continue

            op = getattr(layer, "op", None)
            op_id = getattr(op, "op_id", None)
            ins = getattr(layer, "inputs", None) or []

            def _get_input(dd: DataDef) -> torch.Tensor:
                if dd.ref is None:
                    raise ValueError(f"Layer {name} has input without ref")
                prod, idx = _parse_ref_name_and_index(str(dd.ref))
                val = cache.get(prod)
                if isinstance(val, list):
                    return val[idx]
                return val

            xs = [_get_input(dd) for dd in ins] if ins else []

            if op_id in ("conv2d", "linear") or op_id in ("batch_norm", "batch_norm1d", "batch_norm2d", "batch_norm3d"):
                mod_key = self._name_map[name]
                m = self.mods[mod_key]
                y = m(xs[0])
                cache[name] = y

            elif op_id == "relu":
                cache[name] = F.relu(xs[0])
            elif op_id == "sigmoid":
                cache[name] = torch.sigmoid(xs[0])
            elif op_id == "silu":
                cache[name] = F.silu(xs[0])
            elif op_id == "max_pool2d":
                k = _as_tuple2(getattr(op, "kernel", 2))
                stride = _as_tuple2(getattr(op, "stride", k))
                padding = _as_tuple2(getattr(op, "padding", 0))
                cache[name] = F.max_pool2d(xs[0], kernel_size=k, stride=stride, padding=padding)
            elif op_id == "avg_pool2d":
                k = _as_tuple2(getattr(op, "kernel", 2))
                stride = _as_tuple2(getattr(op, "stride", k))
                padding = _as_tuple2(getattr(op, "padding", 0))
                cache[name] = F.avg_pool2d(xs[0], kernel_size=k, stride=stride, padding=padding)
            elif op_id == "global_avg_pool2d":
                cache[name] = F.adaptive_avg_pool2d(xs[0], output_size=(1, 1))
            elif op_id == "global_max_pool2d":
                cache[name] = F.adaptive_max_pool2d(xs[0], output_size=(1, 1))
            elif op_id == "split":
                axis = int(getattr(op, "axis", 1))
                split = getattr(op, "split", None)
                if split is None:
                    raise ValueError(f"split layer {name} missing split attr")
                sizes = list(split) if isinstance(split, (list, tuple)) else [int(split)]
                cache[name] = list(torch.split(xs[0], sizes, dim=axis))
            elif op_id == "mul":
                cache[name] = xs[0] * xs[1]
            elif op_id == "add":
                cache[name] = xs[0] + xs[1]
            elif op_id == "concat":
                axis = int(getattr(op, "axis", 1))
                cache[name] = torch.cat(xs, dim=axis)
            elif op_id == "flatten":
                start_dim = int(getattr(op, "start_dim", 1))
                cache[name] = torch.flatten(xs[0], start_dim=start_dim)
            elif op_id == "reshape":
                # reshape op typically has a target shape attr or comes from constant; here best-effort
                shape = getattr(op, "shape", None)
                if shape is None:
                    raise NotImplementedError("reshape without explicit shape is not supported yet")
                cache[name] = xs[0].reshape(*[int(s) for s in shape])
            else:
                raise NotImplementedError(f"Unsupported op_id in IRModule: {op_id!r} (layer {name})")

        raise RuntimeError("No graph_output layer found in IR")


def build_torch_model_from_ir(ir: BaseIR) -> BuildResult:
    m = IRModule(ir)
    return BuildResult(model=m, module_name_map=m.module_name_map)


def load_weights_into_model(
    model: nn.Module,
    weights: Dict[str, Any],
    *,
    module_name_map: Optional[Dict[str, str]] = None,
    strict: bool = True,
) -> None:
    """
    Load exported weights dict (keys like 'LayerName.weight', 'LayerName.running_mean') into IRModule.

    Handles common transpose mismatch for Linear:
      - if weight matches param shape -> copy
      - else if transposed matches -> transpose then copy
    """
    if module_name_map is None and hasattr(model, "module_name_map"):
        module_name_map = getattr(model, "module_name_map")  # type: ignore
    module_name_map = module_name_map or {}

    # Helpers
    def get_module(layer_name: str) -> Optional[nn.Module]:
        if not hasattr(model, "mods"):
            return None
        mods = getattr(model, "mods")
        key = module_name_map.get(layer_name, _safe_module_key(layer_name))
        if key not in mods:
            return None
        return mods[key]

    missing: List[str] = []

    def handle_shape_mismatch(message: str) -> None:
        if strict:
            raise ValueError(message)

    for k, v in weights.items():
        if "." not in k:
            continue
        layer_name, suffix = k.rsplit(".", 1)
        mod = get_module(layer_name)
        if mod is None:
            if strict:
                missing.append(k)
            continue

        t = v
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(v)

        if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear) or isinstance(mod, nn.BatchNorm2d):
            if suffix == "weight":
                if hasattr(mod, "weight") and mod.weight is not None:
                    dst = mod.weight.data
                    if t.shape == dst.shape:
                        dst.copy_(t.to(dst.dtype))
                    elif isinstance(mod, nn.Linear) and t.t().shape == dst.shape:
                        dst.copy_(t.t().to(dst.dtype))
                    else:
                        handle_shape_mismatch(f"Weight shape mismatch for {k}: got {tuple(t.shape)}, expected {tuple(dst.shape)}")
            elif suffix == "bias":
                if hasattr(mod, "bias") and mod.bias is not None:
                    dst = mod.bias.data
                    if t.shape == dst.shape:
                        dst.copy_(t.to(dst.dtype))
                    else:
                        handle_shape_mismatch(f"Bias shape mismatch for {k}: got {tuple(t.shape)}, expected {tuple(dst.shape)}")
            elif suffix in ("running_mean", "running_var"):
                if hasattr(mod, suffix):
                    buf = getattr(mod, suffix)
                    if t.shape == buf.shape:
                        buf.copy_(t.to(buf.dtype))
                    else:
                        handle_shape_mismatch(f"{suffix} shape mismatch for {k}: got {tuple(t.shape)}, expected {tuple(buf.shape)}")
            else:
                # ignore unknown
                continue

    if missing and strict:
        raise KeyError(f"Some weight keys did not map to any module: {missing[:20]} (total {len(missing)})")


def load_weights_file(path: str) -> Dict[str, Any]:
    """
    Load exported weights from .pt or .npz.
    For .npz, keys are assumed to have '.' replaced with '_' (export_weights behavior).
    """
    if path.endswith(".pt") or path.endswith(".pth"):
        return torch.load(path, map_location="cpu")
    if path.endswith(".npz"):
        import numpy as np

        z = np.load(path)
        out: Dict[str, Any] = {}
        for k in z.files:
            # best-effort: cannot reliably restore dots, so keep '_' key as-is
            out[k] = z[k]
        return out
    raise ValueError(f"Unsupported weights file: {path}")

