"""
IR graph visualization utilities (Graphviz).

This module provides:
- to_dot(ir): build a DOT string for BaseIR.layers graph
- render_ir(ir, out_file, format): render graph to an image/PDF if python-graphviz is installed

Nodes:
- light blue rounded rectangles with layer names

Edges:
- directed arrows producer -> consumer
- edge label shows tensor shape (prefer producer output shape; fallback to consumer input shape)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Any

from .ir import BaseIR
from .datadef import DataDef


def _shape_str_from_datadef(dd: Optional[DataDef]) -> str:
    """
    Convert DataDef's known fields to a compact tuple string for edge labels.
    Priority:
      - dd.shape (if present): "(...)"
      - (channel, height, width) if any are present: "(c, h, w)" with unknowns omitted if all missing
      - empty string if unknown
    """
    if dd is None:
        return ""
    shape = getattr(dd, "shape", None)
    if shape:
        try:
            return str(tuple(int(x) for x in shape))
        except Exception:
            return str(shape)

    ch = getattr(dd, "channel", None)
    h = getattr(dd, "height", None)
    w = getattr(dd, "width", None)
    if ch is None and h is None and w is None:
        return ""
    # Use a tuple-like string. Fill missing dims with '?' only if some dims exist.
    c_s = str(int(ch)) if ch is not None else "?"
    h_s = str(int(h)) if h is not None else "?"
    w_s = str(int(w)) if w is not None else "?"
    return f"({c_s}, {h_s}, {w_s})"


def _resolve_producer_output_dd(
    ir: BaseIR,
    producer: str,
    output_index: int,
) -> Optional[DataDef]:
    layers = getattr(ir, "layers", None) or {}
    if producer not in layers:
        return None
    out_list = getattr(layers[producer], "outputs", None) or []
    if 0 <= output_index < len(out_list):
        return out_list[output_index]
    return None


def _parse_ref(ref_str: str) -> Tuple[str, int]:
    """
    Parse a ref like 'LayerName' or 'LayerName:2' -> (LayerName, 2).
    """
    if ":" not in ref_str:
        return ref_str, 0
    name, idx = ref_str.split(":", 1)
    try:
        return name, int(idx)
    except Exception:
        return name, 0


def to_dot(ir: BaseIR, *, rankdir: str = "TB") -> str:
    """
    Build DOT text for IR layers graph.
    Does not require the graphviz package.
    """
    layers = getattr(ir, "layers", None) or {}

    lines: list[str] = []
    lines.append("digraph IR {")
    lines.append(f'  rankdir="{rankdir}";')
    lines.append('  graph [fontsize=10, fontname="Times-Roman"];')
    # Style reference:
    # node_INP_A = dict(shape='box', style='rounded,filled', color='skyblue')
    # edge_INP_E = dict(penwidth='3', color='blue')
    lines.append('  node [shape=box, style="rounded,filled", color="skyblue", fontname="Times-Roman", fontsize=20];')
    lines.append('  edge [penwidth=3, color="blue", fontname="Times-Roman", fontsize=12, arrowsize=0.8];')

    # Nodes
    for name in layers.keys():
        safe = name.replace('"', '\\"')
        lines.append(f'  "{safe}" [label="{safe}"];')

    # Edges inferred from inputs
    for consumer_name, layer in layers.items():
        ins = getattr(layer, "inputs", None) or []
        for dd in ins:
            ref = getattr(dd, "ref", None)
            if ref is None:
                continue
            ref_str = str(ref)
            producer_name, out_idx = _parse_ref(ref_str.split(".", 1)[0])
            if producer_name not in layers:
                continue
            prod_out_dd = _resolve_producer_output_dd(ir, producer_name, out_idx)
            label = _shape_str_from_datadef(prod_out_dd) or _shape_str_from_datadef(dd)
            lbl = label.replace('"', '\\"')
            p = producer_name.replace('"', '\\"')
            c = consumer_name.replace('"', '\\"')
            if lbl:
                lines.append(f'  "{p}" -> "{c}" [label="{lbl}"];')
            else:
                lines.append(f'  "{p}" -> "{c}";')

    lines.append("}")
    return "\n".join(lines)


def render_ir(
    ir: BaseIR,
    out_file: str | Path,
    *,
    format: Optional[str] = None,
    rankdir: str = "LR",
    engine: str = "dot",
) -> str:
    """
    Render IR graph using python-graphviz (requires `graphviz` package and system Graphviz).

    Args:
        ir: BaseIR
        out_file: output path (e.g. 'out.png' or 'out.svg' or 'out.pdf')
        format: override format; if None infer from suffix
        rankdir: LR / TB / etc
        engine: graphviz engine, default 'dot'

    Returns:
        The rendered file path as string.
    """
    try:
        from graphviz import Source  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "render_ir requires the python package 'graphviz' and a Graphviz installation. "
            "Install the package (pip install graphviz) and ensure 'dot' is on PATH. "
            "You can always call to_dot(ir) to get DOT without dependencies."
        ) from e

    out_path = Path(out_file)
    if format is None:
        suf = out_path.suffix.lower().lstrip(".")
        format = suf or "png"

    dot = to_dot(ir, rankdir=rankdir)
    src = Source(dot, engine=engine, format=format)

    # graphviz.Source.render expects filename without suffix when format is set;
    # pass the full path without suffix to avoid duplicate extensions.
    stem_path = out_path.with_suffix("")
    rendered = src.render(filename=str(stem_path), cleanup=True)
    return str(rendered)


__all__ = ["to_dot", "render_ir"]

