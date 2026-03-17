"""
Export a PyTorch nn.Module to IR via ONNX.
Run from repo root: python -m CIMA_TC.Compiler.test.PyTorchExample.export_ir_pytorch
Or from this directory: python export_ir_pytorch.py

Requires: pip install torch onnx
"""

import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import torch
import torch.nn as nn

from ...frontend.pytorch import ConvertTorch, TorchConversionConfig


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

    # weights and BN parameters are exported separately (IR only keeps structure; PyTorch -> ONNX -> IR path exports state_dict)
    converter.export_weights(OUT_WEIGHTS)
    print(f"State_dict written: {OUT_WEIGHTS}")
    # Usage: state_dict = torch.load(OUT_WEIGHTS)  # PyTorch state_dict


if __name__ == "__main__":
    main()
