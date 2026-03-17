"""
Frontend-shared IR rewrite passes.

These passes operate on BaseIR.layers (GraphLayer) and can be reused by multiple frontends
(ONNX, PyTorch->ONNX, PyTorch FX).

Notes:
- Passes optionally accept a `name_map` (e.g. ONNX parser.name_map: onnx_node_name -> ir_layer_name)
  and will update it when layer renaming happens, so later weight attachment stays consistent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from ...IR_tool.core import BaseIR, BaseLayer, DataDef, OpLayer, make_op


BN_OP_IDS: Tuple[str, ...] = ("batch_norm", "batch_norm1d", "batch_norm2d", "batch_norm3d")


def _layer_op_id(layers: Dict[str, BaseLayer], name: str) -> str:
    l = layers.get(name)
    op = getattr(l, "op", None)
    return getattr(op, "op_id", "") if op else ""


def _rewrite_refs(layers: Dict[str, BaseLayer], old: str, new: str) -> None:
    """
    Replace all DataDef refs that point to `old` (as first segment) with `new`.
    Preserves branch index in the first segment.
    """
    for _, layer in layers.items():
        for lst_name in ("inputs", "outputs"):
            lst = getattr(layer, lst_name, None) or []
            for dd in lst:
                if dd.ref is None or not dd.ref.segments:
                    continue
                seg0 = dd.ref.segments[0]
                if seg0.name != old:
                    continue
                new_first = f"{new}:{seg0.index}" if seg0.index is not None else new
                rest = ".".join(str(s) for s in dd.ref.segments[1:])
                dd.set_ref(f"{new_first}.{rest}" if rest else new_first)


def _build_consumers(layers: Dict[str, BaseLayer]) -> Dict[str, List[str]]:
    consumers: Dict[str, List[str]] = {n: [] for n in layers}
    for cname, layer in layers.items():
        ins = getattr(layer, "inputs", None) or []
        for dd in ins:
            if dd.ref is None or not dd.ref.segments:
                continue
            prod = dd.ref.segments[0].name
            if prod in consumers:
                consumers[prod].append(cname)
    return consumers


def _nearest_conv_or_fc(layers: Dict[str, BaseLayer], start: str) -> Optional[str]:
    """
    Walk upstream following the first input ref to find nearest conv2d/linear.
    """
    seen: Set[str] = set()
    cur = start
    while cur and cur not in seen and cur in layers:
        seen.add(cur)
        if _layer_op_id(layers, cur) in ("conv2d", "linear"):
            return cur
        ins = getattr(layers[cur], "inputs", None) or []
        if not ins or ins[0].ref is None or not ins[0].ref.segments:
            break
        cur = ins[0].ref.segments[0].name
    return None


def fuse_sigmoid_mul_to_silu(
    ir: BaseIR,
    *,
    name_map: Optional[Dict[str, str]] = None,
) -> None:
    """
    Fuse Sigmoid + Mul pattern into a Silu op in the IR graph.

    Pattern:
        sigmoid = Sigmoid(x)
        y = Mul(x, sigmoid)
    Rewritten as:
        y = Silu(x)

    Naming:
        - if x producer is BN: Silu_{nearest_conv_or_fc} (fallback Silu_{bn_name})
        - else: Silu_{producer_name}
    """
    if not getattr(ir, "layers", None):
        return
    layers = ir.layers or {}
    consumers = _build_consumers(layers)

    def rewrite_consumers(from_name: str, to_name: str) -> None:
        _rewrite_refs(layers, from_name, to_name)

    to_remove: Set[str] = set()
    to_add: Dict[str, OpLayer] = {}

    for sig_name, sig_layer in list(layers.items()):
        if _layer_op_id(layers, sig_name) != "sigmoid":
            continue
        sig_ins = getattr(sig_layer, "inputs", None) or []
        if len(sig_ins) != 1 or sig_ins[0].ref is None or not sig_ins[0].ref.segments:
            continue
        x_prod = sig_ins[0].ref.segments[0].name
        if x_prod not in layers:
            continue

        # sigmoid must feed only this mul to avoid breaking shared subgraphs
        if len(consumers.get(sig_name, [])) != 1:
            continue

        # Find mul consuming both x_prod and sig_name
        mul_name: Optional[str] = None
        for cand in consumers.get(sig_name, []):
            if _layer_op_id(layers, cand) != "mul":
                continue
            ins = getattr(layers[cand], "inputs", None) or []
            if len(ins) != 2:
                continue
            refs = []
            for dd in ins:
                if dd.ref is None or not dd.ref.segments:
                    refs.append(None)
                else:
                    refs.append(dd.ref.segments[0].name)
            if (x_prod in refs) and (sig_name in refs):
                mul_name = cand
                break
        if mul_name is None:
            continue

        mul_outs = getattr(layers[mul_name], "outputs", None) or []
        if not mul_outs:
            continue

        # Naming
        if _layer_op_id(layers, x_prod) in BN_OP_IDS:
            base = _nearest_conv_or_fc(layers, x_prod) or x_prod
            silu_name = f"Silu_{base}"
        else:
            silu_name = f"Silu_{x_prod}"
        if silu_name in layers or silu_name in to_add:
            idx = 0
            while f"{silu_name}_{idx}" in layers or f"{silu_name}_{idx}" in to_add:
                idx += 1
            silu_name = f"{silu_name}_{idx}"

        # Input metadata: reuse x_prod first output if present
        prod_outs = getattr(layers[x_prod], "outputs", None) or []
        if prod_outs and isinstance(prod_outs[0], DataDef):
            pdd = prod_outs[0]
            silu_in = DataDef(
                ref=x_prod,
                channel=getattr(pdd, "channel", None),
                height=getattr(pdd, "height", None),
                width=getattr(pdd, "width", None),
                channel_last=getattr(pdd, "channel_last", True),
            )
        else:
            silu_in = DataDef(ref=x_prod)

        silu_layer = OpLayer(
            type="op",
            op=make_op("silu"),
            inputs=[silu_in],
            outputs=[o.clone() if hasattr(o, "clone") else o for o in mul_outs],  # type: ignore
        )
        to_add[silu_name] = silu_layer

        rewrite_consumers(mul_name, silu_name)
        to_remove.add(sig_name)
        to_remove.add(mul_name)

    if not to_add and not to_remove:
        return
    for n, l in to_add.items():
        layers[n] = l
    for n in to_remove:
        layers.pop(n, None)

    topo = ir.topological_order()
    if topo and len(topo) == len(layers):
        ir.layers = {n: layers[n] for n in topo}  # type: ignore[index]


def rename_batchnorm_layers(
    ir: BaseIR,
    *,
    name_map: Optional[Dict[str, str]] = None,
) -> None:
    """
    Rename BN layers to '{nearest_conv_or_fc}_bn' (nearest upstream conv2d/linear).

    If name_map is provided (onnx_node_name -> ir_layer_name), it will be updated so
    later weight attachment uses renamed layer names.
    """
    if not getattr(ir, "layers", None):
        return
    layers = ir.layers or {}

    inv_name_map: Dict[str, str] = {}
    if name_map:
        inv_name_map = {v: k for k, v in name_map.items()}

    def is_bn(n: str) -> bool:
        return _layer_op_id(layers, n) in BN_OP_IDS

    def unique_bn_name(base: str) -> str:
        cand = f"{base}_bn"
        if cand not in layers:
            return cand
        i = 0
        while f"{cand}_{i}" in layers:
            i += 1
        return f"{cand}_{i}"

    topo = ir.topological_order()
    for bn_name in list(topo):
        if bn_name not in layers or not is_bn(bn_name):
            continue
        ins = getattr(layers[bn_name], "inputs", None) or []
        if not ins or ins[0].ref is None or not ins[0].ref.segments:
            continue
        src = ins[0].ref.segments[0].name
        base = _nearest_conv_or_fc(layers, src) or src
        new_name = unique_bn_name(base)
        if new_name == bn_name:
            continue

        layer_obj = layers.pop(bn_name)
        layers[new_name] = layer_obj
        _rewrite_refs(layers, bn_name, new_name)

        if name_map:
            orig_node = inv_name_map.get(bn_name)
            if orig_node is not None:
                name_map[orig_node] = new_name
                inv_name_map[new_name] = orig_node

    topo2 = ir.topological_order()
    if topo2 and len(topo2) == len(layers):
        ir.layers = {n: layers[n] for n in topo2}  # type: ignore[index]


__all__ = ["fuse_sigmoid_mul_to_silu", "rename_batchnorm_layers"]

