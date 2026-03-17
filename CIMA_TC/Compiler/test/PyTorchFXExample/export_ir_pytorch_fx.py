"""
Export PyTorch nn.Module to IR via torch.fx (direct, no ONNX).
Run from repo root: python -m CIMA_TC.Compiler.test.PyTorchFXExample.export_ir_pytorch_fx
Or from this directory: python export_ir_pytorch_fx.py

Requires: pip install torch
"""

import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import torch
import torch.nn as nn

from ...frontend.pytorch_fx import ConvertFX, FXConversionConfig


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


OUT_DIR = _SCRIPT_DIR
OUT_YAML = os.path.join(OUT_DIR, "SimpleCNN_fx_ir.yaml")
OUT_WEIGHTS = os.path.join(OUT_DIR, "SimpleCNN_fx_weights.pt")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    model = SimpleCNN()
    example_input = torch.randn(1, 3, 32, 32)

    config = FXConversionConfig(
        example_input=example_input,
        ir_file=OUT_YAML,
    )
    converter = ConvertFX(config)
    print("Converting PyTorch model -> IR (via torch.fx, no ONNX)")
    ir = converter.convert(model)
    converter.dump()
    print(f"IR (YAML) written: {OUT_YAML}")
    print(f"Layers: {len(ir.layers)}")

    # weights and BN parameters are exported separately (IR only keeps structure)
    converter.export_weights(OUT_WEIGHTS)
    print(f"State_dict written: {OUT_WEIGHTS}")
    # Usage: state_dict = torch.load(OUT_WEIGHTS)  # PyTorch state_dict, key is module path


if __name__ == "__main__":
    main()
