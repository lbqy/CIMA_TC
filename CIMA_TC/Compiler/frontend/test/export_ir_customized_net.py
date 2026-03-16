"""
Parse CustomizedNet.onnx and export IR to YAML.
Run from repo root: python -m CIMA_TC.Compiler.frontend.test.export_ir_customized_net
Or from this directory: python export_ir_customized_net.py
"""

import os
import sys

# Ensure project root (parent of CIMA_TC) on path when run as script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from CIMA_TC.Compiler.frontend import ConvertONNX, ConversionConfig

ONNX_DIR = os.path.join(_SCRIPT_DIR, "CustomizedNet")
ONNX_PATH = os.path.join(ONNX_DIR, "CustomizedNet.onnx")
OUT_YAML = os.path.join(ONNX_DIR, "CustomizedNet_ir.yaml")
OUT_WEIGHTS = os.path.join(ONNX_DIR, "CustomizedNet_weights.pt")


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

    # weights and BN parameters are exported separately (IR only keeps structure)
    converter.export_weights(OUT_WEIGHTS)
    print(f"Weights (conv/fc/BN etc.) written: {OUT_WEIGHTS}")
    # Usage: weights = torch.load(OUT_WEIGHTS)  # dict[str, Tensor], key is IR layer name.parameter name


if __name__ == "__main__":
    main()
