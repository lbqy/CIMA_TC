"""
Export a PyTorch nn.Module to IR via ONNX.
Run from repo root: python -m CIMA_TC.Compiler.frontend.test.export_ir_pytorch
Or from this directory: python export_ir_pytorch.py

Requires: pip install torch onnx
"""

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn as nn

from CIMA_TC.Compiler.frontend.pytorch import ConvertTorch, TorchConversionConfig


class SimpleCNN(nn.Module):
    """Small CNN for demo: Conv -> ReLU -> Conv -> ReLU -> AdaptiveAvgPool -> Flatten -> Linear."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)


OUT_DIR = os.path.join(_SCRIPT_DIR, "PyTorchExample")
OUT_YAML = os.path.join(OUT_DIR, "SimpleCNN_ir.yaml")
OUT_WEIGHTS = os.path.join(OUT_DIR, "SimpleCNN_weights.pt")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    model = SimpleCNN()
    example_input = torch.randn(1, 3, 32, 32)

    config = TorchConversionConfig(
        example_input=example_input,
        ir_file=OUT_YAML,
        fix_layer_name=True,
        opset_version=11,
    )
    converter = ConvertTorch(config)
    print("Converting PyTorch model -> ONNX -> IR")
    ir = converter.convert(model)
    converter.dump()
    print(f"IR (YAML) written: {OUT_YAML}")
    print(f"Layers: {len(ir.layers)}")

    # 权重与 BN 等参数单独导出（IR 仅保留结构；PyTorch -> ONNX -> IR 路径导出 state_dict）
    converter.export_weights(OUT_WEIGHTS)
    print(f"State_dict written: {OUT_WEIGHTS}")
    # 用法：state_dict = torch.load(OUT_WEIGHTS)  # PyTorch state_dict


if __name__ == "__main__":
    main()
