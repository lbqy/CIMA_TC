"""
Test: export split_model_ir and split weights for CustomizedNet.

Run from repo root:
    python -m CIMA_TC.Compiler.test.CustomizedNet.test_split_customized_net
"""

from __future__ import annotations

import os

from ...frontend import ConvertONNX, ConversionConfig
from ...mapper.xb_split import XBConfig
from ...mapper.split_pass import export_split_model


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))

    onnx_path = os.path.join(script_dir, "CustomizedNet.onnx")
    ir_out = os.path.join(script_dir, "CustomizedNet_ir.yaml")

    split_ir_out = os.path.join(script_dir, "CustomizedNet_split_ir.yaml")
    split_weights_pt = os.path.join(script_dir, "CustomizedNet_split_weights.pt")

    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    # 1. Frontend: ONNX -> IR (structure)
    cfg = ConversionConfig(
        onnx_file=onnx_path,
        ir_file=ir_out,
        fix_layer_name=False,
        store_intermediate_model=False,
    )
    converter = ConvertONNX(cfg)
    print(f"Converting ONNX -> IR: {onnx_path}")
    ir = converter.convert()

    # 2. XB config: 576x128 XB, 4 XBs per thread along columns
    xb = XBConfig(rows=576, cols=128, max_xbs=4)

    # 3. Run split pass and export split_model_ir + split weights/BN
    print("Running XB-based split mapping ...")
    export_split_model(
        ir,
        xb,
        # Default split_bn=True: keep BN adjacent to split conv/fc.
        ir_file=split_ir_out,
        weight_file=split_weights_pt,
    )

    print(f"\nSplit IR written to: {split_ir_out}")
    print(f"Split weights (conv/fc + BN params) written to: {split_weights_pt}")


if __name__ == "__main__":
    main()

