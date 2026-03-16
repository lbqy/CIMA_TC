"""
PyTorch model -> IR conversion via ONNX export.
Uses the ONNX frontend internally; keeps PyTorch-specific logic isolated.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Optional, Union

from CIMA_TC.Compiler.IR_tool.core.ir import BaseIR, save_ir
from CIMA_TC.Compiler.IR_tool.core.jsonable import SerializationConfig

from ..config import ConversionConfig
from ..converter import ConvertONNX
from ..utils.weight_export import export_weights as _export_weights_impl
from .config import TorchConversionConfig


def _check_torch() -> None:
    try:
        import torch
    except ImportError as e:
        raise RuntimeError("PyTorch is required for the PyTorch frontend. Install with: pip install torch") from e


def _export_to_onnx(
    model: Any,
    example_input: Any,
    *,
    opset_version: int = 11,
    dynamic_axes: Optional[dict[str, Any]] = None,
    do_constant_folding: bool = True,
    input_names: Optional[list[str]] = None,
    output_names: Optional[list[str]] = None,
) -> Any:
    """Export PyTorch model to ONNX ModelProto in memory."""
    import torch
    import onnx

    if input_names is None:
        input_names = ["input"]
    if output_names is None:
        output_names = ["output"]

    f = io.BytesIO()
    torch.onnx.export(
        model,
        example_input,
        f,
        opset_version=opset_version,
        dynamic_axes=dynamic_axes,
        do_constant_folding=do_constant_folding,
        input_names=input_names,
        output_names=output_names,
    )
    f.seek(0)
    return onnx.load(f)


class ConvertTorch:
    """
    Convert PyTorch nn.Module to IR.
    Exports to ONNX in memory, then uses the ONNX frontend.
    """

    def __init__(
        self,
        config: Optional["TorchConversionConfig"] = None,
        **kwargs: Any,
    ) -> None:
        _check_torch()
        if config is not None:
            self.config = config
        else:
            allowed = {
                "example_input", "ir_file", "fix_layer_name", "store_intermediate_model",
                "opset_version", "dynamic_axes", "do_constant_folding",
                "input_names", "output_names",
            }
            self.config = TorchConversionConfig(**{k: v for k, v in kwargs.items() if k in allowed})
        self.ir: Optional[BaseIR] = None
        self._state_dict: Optional[Dict[str, Any]] = None

    def convert(
        self,
        model: Any,
        example_input: Optional[Any] = None,
    ) -> BaseIR:
        """
        Convert PyTorch model to IR.
        Uses config.example_input if example_input is None.
        """
        cfg = self.config
        inp = example_input if example_input is not None else cfg.example_input
        if inp is None:
            raise ValueError("Provide example_input or set config.example_input")

        onnx_model = _export_to_onnx(
            model,
            inp,
            opset_version=cfg.opset_version,
            dynamic_axes=cfg.dynamic_axes,
            do_constant_folding=cfg.do_constant_folding,
            input_names=cfg.input_names,
            output_names=cfg.output_names,
        )

        onnx_config = ConversionConfig(
            ir_file=cfg.ir_file,
            fix_layer_name=cfg.fix_layer_name,
            store_intermediate_model=cfg.store_intermediate_model,
        )
        converter = ConvertONNX(onnx_config)
        self.ir = converter.convert(onnx_model=onnx_model)
        import torch
        self._state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        return self.ir

    def dump(
        self,
        ir_file: Optional[Union[str, Path]] = None,
        *,
        as_yaml: bool = True,
    ) -> Optional[str]:
        """Write IR to file (YAML by default). Returns string if ir_file is None."""
        if self.ir is None:
            raise RuntimeError("Run convert() first")
        path = ir_file or getattr(self.config, "ir_file", None)
        kwargs = {}
        if as_yaml:
            kwargs["config"] = SerializationConfig(default_flow_style=False)
        if path is not None:
            save_ir(self.ir, file=str(path), **kwargs)
            return None
        return save_ir(self.ir, **kwargs)

    def export_weights(
        self,
        path: Union[str, Path],
        state_dict: Optional[Dict[str, Any]] = None,
        *,
        format: Optional[str] = None,
    ) -> None:
        """
        Export PyTorch state_dict (conv/fc weights, BN parameters, etc.) to a separate file.
        format: None（按扩展名推断）/ "pt" / "npz" / "npy"（单数组时）。详见 utils.weight_export.export_weights。
        """
        sd = state_dict if state_dict is not None else self._state_dict
        if sd is None:
            raise RuntimeError("Run convert(model) first so state_dict is captured, or pass state_dict=...")
        _export_weights_impl(sd, path, format=format)
