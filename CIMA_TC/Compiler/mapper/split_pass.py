from __future__ import annotations

"""
XB-based split mapping pass.

Given an IR and conv/linear weights, split layers that exceed a single PE thread's
XB capacity along row (input-channel) and/or column (output-channel) dimensions:
- Row split: Split input -> sub-convs -> Add tree.
- Column split: sub-convs -> Concat.
- 2D split: Split -> grid of sub-convs -> Add per column -> Concat.
BN can be adaptively split per column chunk when split_bn=True.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..IR_tool.core import (
    BaseIR,
    BaseLayer,
    OpLayer,
    DataDef,
    make_op,
)
from ..IR_tool.core.ir import save_ir
from ..IR_tool.core.ref import Ref
from ..frontend.utils.weight_export import export_weights
from .xb_split import XBConfig, analyze_ir_for_xb_splits


@dataclass
class SplitResult:
    ir: BaseIR
    weights: Dict[str, Any]
    bn_params: Dict[str, Any]


def _clone_ir(ir: BaseIR) -> BaseIR:
    """Deep clone IR via JSON round-trip (structure only)."""
    text = save_ir(ir)
    assert isinstance(text, str)
    return BaseIR.load_ir(text)


def _split_ranges(total: int, num_splits: int) -> List[Tuple[int, int]]:
    """Split [0, total) into num_splits nearly equal ranges."""
    base = total // num_splits
    rem = total % num_splits
    ranges: List[Tuple[int, int]] = []
    start = 0
    for i in range(num_splits):
        size = base + (1 if i < rem else 0)
        end = start + size
        ranges.append((start, end))
        start = end
    return ranges


def _slice_conv_weight_col(w: Any, out_c_slices: List[Tuple[int, int]]) -> List[Any]:
    """Slice conv weight [out_c, in_c, kh, kw] along dim=0."""
    import numpy as np

    arr = np.asarray(w)
    return [arr[s:e, ...].copy() for (s, e) in out_c_slices]


def _slice_fc_weight_col(w: Any, out_c_slices: List[Tuple[int, int]]) -> List[Any]:
    """Slice fc weight [out_c, in_c] along dim=0."""
    import numpy as np

    arr = np.asarray(w)
    return [arr[s:e, :].copy() for (s, e) in out_c_slices]


def _slice_conv_weight_row(
    w: Any, in_c_ranges: List[Tuple[int, int]]
) -> List[Any]:
    """Slice conv weight [out_c, in_c, kh, kw] along dim=1 (in_c)."""
    import numpy as np

    arr = np.asarray(w)
    return [arr[:, s:e, :, :].copy() for (s, e) in in_c_ranges]


def _slice_fc_weight_row(w: Any, in_c_ranges: List[Tuple[int, int]]) -> List[Any]:
    """Slice fc weight [out_c, in_c] along dim=1 (in_c)."""
    import numpy as np

    arr = np.asarray(w)
    return [arr[:, s:e].copy() for (s, e) in in_c_ranges]


def _output_shape_from_ref(
    layers: Dict[str, BaseLayer],
    ref: str,
) -> Tuple[int, int, int, bool]:
    """Resolve ref (e.g. 'LayerName' or 'LayerName:0') to producer output shape; return (channel, height, width, channel_last)."""
    segs = ref.split(":")
    layer_name = segs[0]
    out_idx = int(segs[1]) if len(segs) > 1 else 0
    if layer_name not in layers:
        return 1, 1, 1, True
    layer = layers[layer_name]
    outs = getattr(layer, "outputs", None) or []
    if out_idx >= len(outs):
        return 1, 1, 1, True
    dd = outs[out_idx]
    ch = getattr(dd, "channel", None) or 1
    h = getattr(dd, "height", None) or 1
    w = getattr(dd, "width", None) or 1
    cl = getattr(dd, "channel_last", True)
    return ch, h, w, cl


def _data_def_with_shape(
    ref: str,
    layers: Dict[str, BaseLayer],
    *,
    channel: Optional[int] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    channel_last: Optional[bool] = None,
) -> DataDef:
    """Build DataDef(ref=..., channel, height, width, channel_last); infer from producer output if not given."""
    if channel is None or height is None or width is None:
        ch, h, w, cl = _output_shape_from_ref(layers, ref)
        channel = channel if channel is not None else ch
        height = height if height is not None else h
        width = width if width is not None else w
        channel_last = channel_last if channel_last is not None else cl
    else:
        channel_last = channel_last if channel_last is not None else True
    return DataDef(
        ref=ref,
        channel=channel,
        height=height,
        width=width,
        channel_last=channel_last,
    )


def _rewire_consumers_to(
    layers: Dict[str, BaseLayer],
    from_name: str,
    to_name: str,
    *,
    output_index: Optional[int] = None,
) -> None:
    """Rewrite all layers that consume `from_name` (optionally branch output_index) to consume `to_name` (same branch)."""
    for other_name, other_layer in layers.items():
        if other_name in (from_name, to_name):
            continue
        ins = getattr(other_layer, "inputs", None)
        if not ins:
            continue
        for dd in ins:
            if dd.ref is None or not dd.ref.segments:
                continue
            seg = dd.ref.segments[0]
            if seg.name != from_name:
                continue
            # Build new ref: replace first segment name with to_name, keep index if any.
            new_first = f"{to_name}:{seg.index}" if seg.index is not None else to_name
            rest = ".".join(str(s) for s in dd.ref.segments[1:])
            new_ref_str = f"{new_first}.{rest}" if rest else new_first
            dd.set_ref(Ref.parse(new_ref_str))


def _get_next_bn_consumer(
    layers: Dict[str, BaseLayer], topo: List[str], producer_name: str
) -> Optional[str]:
    """If the single immediate consumer of producer_name is a BN layer, return its name; else None."""
    consumers = []
    for n in topo:
        if n == producer_name:
            continue
        layer = layers.get(n)
        if not layer or not getattr(layer, "inputs", None):
            continue
        for dd in layer.inputs:
            if dd.ref is not None and dd.ref.segments and dd.ref.segments[0].name == producer_name:
                consumers.append(n)
                break
    if len(consumers) != 1:
        return None
    op = getattr(layers[consumers[0]], "op", None)
    if op is None:
        return None
    op_id = getattr(op, "op_id", "")
    if op_id not in ("batch_norm", "batch_norm1d", "batch_norm2d", "batch_norm3d"):
        return None
    return consumers[0]


def _bn_op_id_and_epsilon(bn_layer: BaseLayer) -> tuple[str, float]:
    op = getattr(bn_layer, "op", None)
    op_id = getattr(op, "op_id", "") if op else ""
    if op_id in ("batch_norm", "batch_norm1d", "batch_norm2d", "batch_norm3d"):
        eps = float(getattr(op, "epsilon", 1e-5))
        return op_id, eps
    # Fallback to 2d BN for CNN-like graphs
    return "batch_norm2d", 1e-5


def _get_bn_param_array(
    *,
    weight_store: Dict[str, Any],
    bn_store: Dict[str, Any],
    key: str,
) -> Optional[Any]:
    """
    BN 参数在当前前端的存储策略是：
    - weight/bias 通常放在 weight_store
    - running_mean/running_var 放在 bn_store
    为了兼容，两个 store 都查。
    """
    if key in bn_store:
        return bn_store[key]
    if key in weight_store:
        return weight_store[key]
    return None


OP_NAME_PREFIXES: Dict[str, str] = {
    "conv1d": "Conv",
    "conv2d": "Conv",
    "conv3d": "Conv",
    "linear": "Gemm",
    "batch_norm": "BatchNorm",
    "batch_norm1d": "BatchNorm",
    "batch_norm2d": "BatchNorm",
    "batch_norm3d": "BatchNorm",
    "relu": "Relu",
    "max_pool2d": "MaxPool",
    "avg_pool2d": "AveragePool",
    "global_avg_pool2d": "GlobalAveragePool",
    "flatten": "Flatten",
    "split": "Split",
    "concat": "Concat",
    "add": "Add",
    "mul": "Mul",
    "sigmoid": "Sigmoid",
    "silu": "Silu",
}


def _op_name_prefix(layer: BaseLayer) -> str:
    op = getattr(layer, "op", None)
    op_id = getattr(op, "op_id", "") if op is not None else ""
    if op_id in OP_NAME_PREFIXES:
        return OP_NAME_PREFIXES[op_id]
    if op_id:
        return "".join(part[:1].upper() + part[1:] for part in op_id.split("_") if part)
    layer_type = getattr(layer, "type", "layer") or "layer"
    return layer_type[:1].upper() + layer_type[1:]


def _rewrite_ref_name(ref: Ref, name_map: Dict[str, str]) -> str:
    if not ref.segments:
        return str(ref)
    first = ref.segments[0]
    new_name = name_map.get(first.name, first.name)
    new_first = f"{new_name}:{first.index}" if first.index is not None else new_name
    rest = ".".join(str(seg) for seg in ref.segments[1:])
    return f"{new_first}.{rest}" if rest else new_first


def _rewrite_layer_refs(layers: Dict[str, BaseLayer], name_map: Dict[str, str]) -> None:
    for layer in layers.values():
        for field in ("inputs", "outputs"):
            defs = getattr(layer, field, None) or []
            for dd in defs:
                if dd.ref is None or not dd.ref.segments:
                    continue
                if dd.ref.segments[0].name in name_map:
                    dd.set_ref(_rewrite_ref_name(dd.ref, name_map))


def _rename_param_keys(params: Dict[str, Any], name_map: Dict[str, str]) -> Dict[str, Any]:
    renamed: Dict[str, Any] = {}
    for key, value in params.items():
        if "." not in key:
            renamed[key] = value
            continue
        layer_name, suffix = key.rsplit(".", 1)
        renamed[f"{name_map.get(layer_name, layer_name)}.{suffix}"] = value
    return renamed


BN_OP_IDS = {"batch_norm", "batch_norm1d", "batch_norm2d", "batch_norm3d"}


def _is_bn_layer(layer: BaseLayer) -> bool:
    op = getattr(layer, "op", None)
    return getattr(op, "op_id", None) in BN_OP_IDS


def _first_input_layer_name(layer: BaseLayer) -> Optional[str]:
    inputs = getattr(layer, "inputs", None) or []
    if not inputs or inputs[0].ref is None or not inputs[0].ref.segments:
        return None
    return inputs[0].ref.segments[0].name


def _rename_split_graph(
    split_ir: BaseIR,
    split_weights: Dict[str, Any],
    split_bn_params: Dict[str, Any],
) -> None:
    layers = split_ir.layers or {}
    topo = split_ir.topological_order()
    if len(topo) != len(layers):
        raise ValueError("Cannot rename split graph because topological order is incomplete")

    name_map: Dict[str, str] = {}
    prefix_counts: Dict[str, int] = {}
    used_names: set[str] = set()

    def unique_name(base: str) -> str:
        if base not in used_names:
            used_names.add(base)
            return base
        index = 1
        while f"{base}_{index}" in used_names:
            index += 1
        name = f"{base}_{index}"
        used_names.add(name)
        return name

    def next_numbered_name(prefix: str) -> str:
        index = prefix_counts.get(prefix, 0)
        prefix_counts[prefix] = index + 1
        return unique_name(f"{prefix}_{index}")

    for old_name in topo:
        layer = layers[old_name]
        layer_type = getattr(layer, "type", None)
        if old_name == "graph_input":
            new_name = unique_name("graph_input")
        elif old_name == "graph_output":
            new_name = unique_name("graph_output")
        elif layer_type == "input":
            new_name = next_numbered_name("Input")
        elif layer_type == "output":
            new_name = next_numbered_name("Output")
        elif _is_bn_layer(layer):
            producer = _first_input_layer_name(layer)
            producer_name = name_map.get(producer, producer) if producer else None
            new_name = unique_name(f"{producer_name}_bn") if producer_name else next_numbered_name("BatchNorm")
        else:
            new_name = next_numbered_name(_op_name_prefix(layer))
        name_map[old_name] = new_name

    _rewrite_layer_refs(layers, name_map)
    split_ir.layers = {name_map[name]: layers[name] for name in topo}

    renamed_weights = _rename_param_keys(split_weights, name_map)
    split_weights.clear()
    split_weights.update(renamed_weights)

    renamed_bn = _rename_param_keys(split_bn_params, name_map)
    split_bn_params.clear()
    split_bn_params.update(renamed_bn)


def split_model_for_xb(
    ir: BaseIR,
    xb: XBConfig,
    *,
    weight_store: Optional[Dict[str, Any]] = None,
    bn_store: Optional[Dict[str, Any]] = None,
    split_bn: bool = True,
) -> SplitResult:
    """
    Perform XB-aware column splitting for conv2d / linear layers in the IR.

    Args:
        ir:           Original IR (structure only; layer.weights contain shapes).
        xb:           XBConfig describing a single PE thread capacity.
        weight_store: Mapping 'layer.suffix' -> array (e.g. 'Conv_2.weight').
        bn_store:     Mapping for BN parameters (e.g. 'BN_2.running_mean').
        split_bn:     When True, if a BN layer immediately consumes the conv/linear output,
                      split/adjust BN params and insert BN right after each split conv/linear,
                      then remove the original BN so BN remains adjacent to conv/fc.

    Returns:
        SplitResult(new_ir, split_weights, split_bn_params)
    """
    # Default to IR-attached stores if caller does not provide explicit dicts.
    if weight_store is None:
        weight_store = getattr(ir, "weight_store", {}) or {}
    if bn_store is None:
        bn_store = getattr(ir, "bn_store", {}) or {}

    split_weights: Dict[str, Any] = {}
    split_bn_params: Dict[str, Any] = {}

    # 1) Deep copy IR to avoid mutating original
    split_ir = _clone_ir(ir)
    layers = split_ir.layers or {}

    # 2) Compute split plans
    plans = analyze_ir_for_xb_splits(split_ir, xb)
    if not plans:
        # No layers need splitting: just forward all weights/BNs
        split_weights.update(weight_store)
        split_bn_params.update(bn_store)
        return SplitResult(ir=split_ir, weights=split_weights, bn_params=split_bn_params)

    topo = split_ir.topological_order()

    for name in topo:
        if name not in layers:
            continue

        if name not in plans:
            # Layer not split: copy its weights / BN as-is
            for suffix in ("weight", "bias"):
                k = f"{name}.{suffix}"
                if k in weight_store:
                    split_weights[k] = weight_store[k]
            for suffix in ("weight", "bias", "running_mean", "running_var"):
                k = f"{name}.{suffix}"
                if k in bn_store:
                    split_bn_params[k] = bn_store[k]
            continue

        plan = plans[name]
        op_id: str = plan["op_id"]
        w_shape: List[int] = plan["weight_shape"]
        col_splits: int = plan["col_splits"]
        row_splits: int = plan["row_splits"]

        if col_splits <= 1 and row_splits <= 1:
            # No split needed
            for suffix in ("weight", "bias"):
                k = f"{name}.{suffix}"
                if k in weight_store:
                    split_weights[k] = weight_store[k]
            for suffix in ("weight", "bias", "running_mean", "running_var"):
                k = f"{name}.{suffix}"
                if k in bn_store:
                    split_bn_params[k] = bn_store[k]
            continue

        # ---- Row and/or column split ----
        layer: BaseLayer = layers[name]
        weights_def = getattr(layer, "weights", None)
        if not weights_def or "weight" not in weights_def:
            raise ValueError(f"Layer {name} has no weight shape for splitting")

        import numpy as np

        # Conv: row split = in_c, col split = out_c. Linear: same (row = input dim, col = output dim).
        # For linear, IR may store weight shape in ONNX order (in_c, out_c); use op.out_channel/in_channel.
        if op_id == "conv2d":
            total_out_c = w_shape[0]
            total_in_c = w_shape[1] if len(w_shape) >= 2 else 0
        else:
            op_attr = getattr(layer, "op", None)
            total_out_c = getattr(op_attr, "out_channel", None) or w_shape[0]
            total_in_c = getattr(op_attr, "in_channel", None) or (w_shape[1] if len(w_shape) >= 2 else 0)
        col_ranges = _split_ranges(total_out_c, col_splits)
        row_ranges = _split_ranges(total_in_c, row_splits) if row_splits > 1 else [(0, total_in_c)]

        w_key = f"{name}.weight"
        b_key = f"{name}.bias"
        if w_key not in weight_store:
            raise KeyError(f"Missing weight tensor for {w_key} in weight_store")
        w_arr = weight_store[w_key]
        w_arr = np.asarray(w_arr)
        b_arr = weight_store.get(b_key, None)

        # Linear weights may arrive as (in_c, out_c) from ONNX Gemm; normalize to (out_c, in_c).
        if op_id == "linear" and w_arr.ndim == 2:
            op_attr = getattr(layer, "op", None)
            op_in = int(getattr(op_attr, "in_channel", 0) or 0)
            op_out = int(getattr(op_attr, "out_channel", 0) or 0)
            if op_in and op_out and tuple(w_arr.shape) == (op_in, op_out):
                w_arr = w_arr.T.copy()
            total_out_c = int(w_arr.shape[0])
            total_in_c = int(w_arr.shape[1])
            col_ranges = _split_ranges(total_out_c, col_splits)
            row_ranges = _split_ranges(total_in_c, row_splits) if row_splits > 1 else [(0, total_in_c)]
        # else: keep total_out_c, total_in_c, col_ranges, row_ranges from above (conv or already set)

        orig_outs = getattr(layer, "outputs", None)
        if not orig_outs or not isinstance(orig_outs[0], DataDef):
            raise ValueError(f"Layer {name!r} has invalid outputs")
        orig_out_dd = orig_outs[0]
        orig_inputs = getattr(layer, "inputs", None)
        producer_ref = orig_inputs[0].ref if orig_inputs and orig_inputs[0].ref else None
        producer_name = producer_ref.segments[0].name if producer_ref and producer_ref.segments else None

        # Build grid of weight slices: [row_idx][col_idx] -> (w_slice, b_slice or None)
        # Conv: slice by in_c (row) then out_c (col). Linear: weight_store is (out_c, in_c), same slicing.
        grid_weights: List[List[Tuple[Any, Any]]] = []
        for ri, (in_s, in_e) in enumerate(row_ranges):
            row_list: List[Tuple[Any, Any]] = []
            if op_id == "conv2d":
                w_row = _slice_conv_weight_row(w_arr, [(in_s, in_e)])[0]
            elif op_id == "linear":
                w_row = _slice_fc_weight_row(w_arr, [(in_s, in_e)])[0]
            else:
                raise NotImplementedError(f"Unsupported op_id for split: {op_id}")
            for (out_s, out_e) in col_ranges:
                if op_id == "conv2d":
                    w_cell = w_row[out_s:out_e, ...].copy()
                else:
                    w_cell = w_row[out_s:out_e, :].copy()
                b_cell = None
                if b_arr is not None:
                    b_np = np.asarray(b_arr)
                    b_cell = b_np[out_s:out_e].copy()
                    if row_splits > 1:
                        b_cell = b_cell / float(row_splits)
                row_list.append((w_cell, b_cell))
            grid_weights.append(row_list)

        # 1) Optional: Split layer for row_splits > 1 (splits input along channel)
        split_name: Optional[str] = None
        if row_splits > 1 and producer_name:
            split_name = f"Split_0_{name}"
            if split_name in layers:
                raise ValueError(f"Split layer already exists: {split_name!r}")
            split_sizes = [e - s for (s, e) in row_ranges]
            split_op = make_op("split", axis=1, split=split_sizes)
            split_output_dds = [
                DataDef(
                    channel=e - s,
                    height=getattr(orig_inputs[0], "height", None) or 1,
                    width=getattr(orig_inputs[0], "width", None) or 1,
                    channel_last=getattr(orig_inputs[0], "channel_last", True),
                )
                for (s, e) in row_ranges
            ]
            split_input_dd = _data_def_with_shape(producer_name, layers)
            split_layer = OpLayer(
                type="op",
                op=split_op,
                inputs=[split_input_dd],
                outputs=split_output_dds,
            )
            layers[split_name] = split_layer
            # Rewire this layer's producer to feed Split; consumers of producer unchanged
            # (only we consume producer via Split now; our children will consume Split:i)

        # 2) Create grid of child layers: name_{ri}_{cj}
        child_grid: List[List[str]] = []
        for ri in range(row_splits):
            child_row: List[str] = []
            for cj in range(col_splits):
                child_name = f"{name}_{ri}_{cj}"
                if child_name in layers:
                    raise ValueError(f"Split child name already exists: {child_name!r}")
                w_sub, b_sub = grid_weights[ri][cj]
                in_chunk = row_ranges[ri][1] - row_ranges[ri][0]
                out_chunk = col_ranges[cj][1] - col_ranges[cj][0]

                child_layer = layer.clone()
                child_op = getattr(child_layer, "op", None)
                if child_op is not None:
                    if hasattr(child_op, "in_channel"):
                        child_op.in_channel = in_chunk
                    if hasattr(child_op, "out_channel"):
                        child_op.out_channel = out_chunk
                if child_layer.outputs and len(child_layer.outputs) == 1:
                    dd = child_layer.outputs[0]
                    if getattr(dd, "channel", None) is not None:
                        dd.channel = out_chunk
                child_w = child_layer.weights["weight"]
                shape = list(getattr(child_w, "shape", []) or [])
                if len(shape) == 4:
                    shape[0], shape[1] = out_chunk, in_chunk
                elif len(shape) == 2:
                    shape[0], shape[1] = out_chunk, in_chunk
                child_w.shape = shape
                if b_sub is not None and "bias" in child_layer.weights:
                    child_b = child_layer.weights["bias"]
                    b_shape = list(getattr(child_b, "shape", []) or [])
                    if b_shape:
                        b_shape[0] = out_chunk
                        child_b.shape = b_shape

                input_ref = f"{split_name}:{ri}" if split_name else producer_name
                child_input_dd = _data_def_with_shape(
                    input_ref, layers,
                    channel=in_chunk,
                    height=getattr(orig_inputs[0], "height", None) or 1,
                    width=getattr(orig_inputs[0], "width", None) or 1,
                )
                child_layer.set_attr("inputs", [child_input_dd])
                layers[child_name] = child_layer
                child_row.append(child_name)
                split_weights[f"{child_name}.weight"] = w_sub
                if b_sub is not None:
                    split_weights[f"{child_name}.bias"] = b_sub
            child_grid.append(child_row)

        # 3) BN adaptive split (default on): keep BN adjacent to each split conv/fc
        next_bn = _get_next_bn_consumer(layers, topo, name)
        if next_bn and split_bn:
            bn_layer = layers[next_bn]
            bn_op_id, epsilon = _bn_op_id_and_epsilon(bn_layer)

            # Insert BN after each split conv/fc (grid cell). For row-split reductions,
            # adjust bias and running_mean by dividing by row_splits so that after Add tree:
            # sum_i BN_i(x_i) ~= BN(sum_i x_i) for mean/bias terms (as required by mapping_steps.md).
            k = float(row_splits) if row_splits > 1 else 1.0
            for ri in range(row_splits):
                for cj in range(col_splits):
                    conv_child = child_grid[ri][cj]
                    out_s, out_e = col_ranges[cj]
                    ch = out_e - out_s
                    bn_child_name = f"{next_bn}_{ri}_{cj}"
                    if bn_child_name in layers:
                        raise ValueError(f"BN split child already exists: {bn_child_name!r}")

                    bn_input_dd = _data_def_with_shape(
                        conv_child,
                        layers,
                        channel=ch,
                        height=getattr(orig_out_dd, "height", 1),
                        width=getattr(orig_out_dd, "width", 1),
                        channel_last=getattr(orig_out_dd, "channel_last", True),
                    )
                    bn_out_dd = DataDef(
                        channel=ch,
                        height=getattr(orig_out_dd, "height", 1),
                        width=getattr(orig_out_dd, "width", 1),
                        channel_last=getattr(orig_out_dd, "channel_last", True),
                    )
                    bn_weights_shape = {
                        kw: {"shape": [ch]}
                        for kw in ("weight", "bias", "running_mean", "running_var")
                    }
                    bn_child_layer = OpLayer(
                        type="op",
                        op=make_op(bn_op_id, channel=ch, epsilon=epsilon),
                        inputs=[bn_input_dd],
                        outputs=[bn_out_dd],
                        weights=bn_weights_shape,
                    )
                    layers[bn_child_name] = bn_child_layer

                    # Export BN params: slice by output channels; adjust for row_splits if needed
                    for kw in ("weight", "bias", "running_mean", "running_var"):
                        src_key = f"{next_bn}.{kw}"
                        src_arr = _get_bn_param_array(
                            weight_store=weight_store,
                            bn_store=bn_store,
                            key=src_key,
                        )
                        if src_arr is None:
                            continue
                        arr = np.asarray(src_arr)[out_s:out_e].copy()
                        if row_splits > 1 and kw in ("bias", "running_mean"):
                            arr = arr / k
                        split_bn_params[f"{bn_child_name}.{kw}"] = arr

                    # Replace grid reference: downstream reduction/concat uses BN outputs
                    child_grid[ri][cj] = bn_child_name

            # Remove original BN node; its consumers will be rewired after we create merge node.
        else:
            # No BN split: just keep original BN params (if any) for export
            if next_bn:
                for suffix in ("weight", "bias", "running_mean", "running_var"):
                    k = f"{next_bn}.{suffix}"
                    v = _get_bn_param_array(weight_store=weight_store, bn_store=bn_store, key=k)
                    if v is not None:
                        split_bn_params[k] = v

        # 4) For each column j: build Add tree over row parts -> Add_j_name
        add_names: List[str] = []
        for cj in range(col_splits):
            row_output_refs = [child_grid[ri][cj] for ri in range(row_splits)]
            if len(row_output_refs) == 1:
                add_names.append(row_output_refs[0])
                continue
            current = row_output_refs[0]
            for ri in range(1, row_splits):
                add_layer_name = f"Add_{cj}_{ri}_{name}"
                add_op = make_op("add")
                ch = col_ranges[cj][1] - col_ranges[cj][0]
                left_dd = _data_def_with_shape(
                    current, layers,
                    channel=ch,
                    height=getattr(orig_out_dd, "height", 1),
                    width=getattr(orig_out_dd, "width", 1),
                    channel_last=getattr(orig_out_dd, "channel_last", True),
                )
                right_dd = _data_def_with_shape(
                    row_output_refs[ri], layers,
                    channel=ch,
                    height=getattr(orig_out_dd, "height", 1),
                    width=getattr(orig_out_dd, "width", 1),
                    channel_last=getattr(orig_out_dd, "channel_last", True),
                )
                add_out = DataDef(
                    channel=ch,
                    height=getattr(orig_out_dd, "height", 1),
                    width=getattr(orig_out_dd, "width", 1),
                    channel_last=getattr(orig_out_dd, "channel_last", True),
                )
                add_layer = OpLayer(type="op", op=add_op, inputs=[left_dd, right_dd], outputs=[add_out])
                layers[add_layer_name] = add_layer
                current = add_layer_name
            add_names.append(current)

        # 5) Concat over columns (if col_splits > 1) or single output
        if col_splits > 1:
            concat_name = f"Concat_0_{name}"
            if concat_name in layers:
                raise ValueError(f"Concat layer already exists: {concat_name!r}")
            concat_inputs = []
            for cj, add_name in enumerate(add_names):
                ch = col_ranges[cj][1] - col_ranges[cj][0]
                concat_inputs.append(
                    _data_def_with_shape(
                        add_name, layers,
                        channel=ch,
                        height=getattr(orig_out_dd, "height", 1),
                        width=getattr(orig_out_dd, "width", 1),
                        channel_last=getattr(orig_out_dd, "channel_last", True),
                    )
                )
            concat_op = make_op("concat", axis=1)
            concat_layer = OpLayer(
                type="op",
                op=concat_op,
                inputs=concat_inputs,
                outputs=[
                    DataDef(
                        channel=orig_out_dd.channel,
                        height=getattr(orig_out_dd, "height", 1),
                        width=getattr(orig_out_dd, "width", 1),
                        channel_last=getattr(orig_out_dd, "channel_last", True),
                    )
                ],
            )
            layers[concat_name] = concat_layer
            merge_output_name = concat_name
        else:
            merge_output_name = add_names[0]

        # Remove original BN and rewire its consumers to merged output (BN moved earlier)
        if next_bn and split_bn:
            _rewire_consumers_to(layers, next_bn, merge_output_name)
            layers.pop(next_bn, None)

        _rewire_consumers_to(layers, name, merge_output_name)
        layers.pop(name, None)

    split_ir.layers = layers
    _rename_split_graph(split_ir, split_weights, split_bn_params)
    return SplitResult(ir=split_ir, weights=split_weights, bn_params=split_bn_params)


def export_split_model(
    ir: BaseIR,
    xb: XBConfig,
    *,
    weight_store: Optional[Dict[str, Any]] = None,
    bn_store: Optional[Dict[str, Any]] = None,
    split_bn: bool = True,
    ir_file: Union[str, Path],
    weight_file: Union[str, Path],
    bn_file: Optional[Union[str, Path]] = None,
) -> None:
    """
    Convenience helper: run split_model_for_xb and export:
    - split_model_ir (YAML/JSON via save_ir)
    - single weight file: conv/fc weights and BN params (weight, bias, running_mean, running_var)
      are merged and exported to weight_file. bn_file is ignored (kept for backward compatibility).
    """
    result = split_model_for_xb(
        ir,
        xb,
        weight_store=weight_store,
        bn_store=bn_store,
        split_bn=split_bn,
    )

    # 1) IR
    save_ir(result.ir, file=str(ir_file))

    # 2) Weights + BN params in one file
    merged = dict(result.weights)
    merged.update(result.bn_params)
    export_weights(merged, weight_file)


__all__ = [
    "SplitResult",
    "split_model_for_xb",
    "export_split_model",
]

