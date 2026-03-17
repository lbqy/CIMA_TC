"""
Parse CustomizedNet.onnx and export IR to YAML.
Run from repo root: python -m CIMA_TC.Compiler.test.CustomizedNet.export_ir_customized_net
Or from this directory: python export_ir_customized_net.py
"""

import os
import sys

# We expect to be run via -m, so package imports work.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from ...frontend import ConvertONNX, ConversionConfig

ONNX_PATH = os.path.join(_SCRIPT_DIR, "CustomizedNet.onnx")
OUT_YAML = os.path.join(_SCRIPT_DIR, "CustomizedNet_ir.yaml")


def main() -> None:
    if not os.path.isfile(ONNX_PATH):
        print(f"ONNX not found: {ONNX_PATH}")
        sys.exit(1)

    config = ConversionConfig(
        onnx_file=ONNX_PATH,
        ir_file=OUT_YAML,
        fix_layer_name=False,
        store_intermediate_model=False,
    )
    converter = ConvertONNX(config)
    print(f"Converting: {ONNX_PATH}")
    ir = converter.convert()
    converter.dump(ir_file=OUT_YAML, as_yaml=True)
    print(f"IR (YAML) written: {OUT_YAML}")
    print(f"Layers: {len(ir.layers)}")


if __name__ == "__main__":
    main()
