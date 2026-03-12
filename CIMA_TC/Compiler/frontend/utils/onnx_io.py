"""
ONNX model load/save and value_info completion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

# Optional: keep onnx as optional for lightweight use
try:
    import onnx
    from onnx import helper
    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False


def load_onnx(path_or_model: str | Path | Any) -> Any:
    """
    Load ONNX model from path or return ModelProto as-is.
    """
    if not _ONNX_AVAILABLE:
        raise RuntimeError("onnx package is required; install with: pip install onnx")
    if isinstance(path_or_model, (str, Path)):
        return onnx.load(str(path_or_model))
    if hasattr(path_or_model, "graph") and hasattr(path_or_model, "ir_version"):
        return path_or_model
    raise TypeError("path_or_model must be path (str/Path) or ModelProto")


def save_onnx(model: Any, path: str | Path) -> None:
    """Save ONNX model to path."""
    if not _ONNX_AVAILABLE:
        raise RuntimeError("onnx package is required")
    onnx.save(model, str(path))


def add_value_info_for_constants(model: Any) -> Any:
    """
    Add ValueInfo for initializers so shape inference can use them.
    Mutates the model and returns it.
    """
    if not _ONNX_AVAILABLE:
        return model
    inputs_set = {i.name for i in model.graph.input}
    existing = {vi.name: vi for vi in model.graph.value_info}
    for init in model.graph.initializer:
        if init.name in inputs_set:
            continue
        vi = existing.get(init.name)
        if vi is None:
            vi = model.graph.value_info.add()
            vi.name = init.name
        tt = vi.type.tensor_type
        if tt.elem_type == onnx.TensorProto.UNDEFINED:
            tt.elem_type = init.data_type
        if not list(tt.shape.dim):
            for dim in init.dims:
                tt.shape.dim.add().dim_value = dim
    return model
