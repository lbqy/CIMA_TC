"""
Parse ResNet50.onnx and export IR to YAML.
Run from repo root: python -m CIMA_TC.Compiler.frontend.test.export_ir_resnet50
Or from this directory: python export_ir_resnet50.py
"""

import os
import sys

# Ensure project root (parent of CIMA_TC) on path when run as script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from CIMA_TC.Compiler.frontend import ConvertONNX, ConversionConfig

ONNX_DIR = os.path.join(_SCRIPT_DIR, "ResNet50")
ONNX_PATH = os.path.join(ONNX_DIR, "ResNet50.onnx")
OUT_YAML = os.path.join(ONNX_DIR, "ResNet50_ir.yaml")


def main() -> None:
    if not os.path.isfile(ONNX_PATH):
        print(f"ONNX not found: {ONNX_PATH}")
        sys.exit(1)

    config = ConversionConfig(
        onnx_file=ONNX_PATH,
        ir_file=OUT_YAML,
        fix_layer_name=True,
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
