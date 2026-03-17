"""
Auto-generated training script from IR.

IR: C:\Users\LvBuQY\CIMA_TC\CIMA_TC\Compiler\test\ResNet50\ResNet50_split_ir.yaml
Weights: C:\Users\LvBuQY\CIMA_TC\CIMA_TC\Compiler\test\ResNet50\ResNet50_split_weights.pt
"""

from __future__ import annotations

import torch

from CIMA_TC.Compiler.IR_tool.core.ir import BaseIR
from CIMA_TC.Compiler.backend.to_training_code.ir_to_torch import (
    build_torch_model_from_ir,
    load_weights_file,
    load_weights_into_model,
)


def _parse_shape(s: str):
    parts = [int(x) for x in s.split(",") if x.strip()]
    if len(parts) != 4:
        raise ValueError("example_shape must be 'N,C,H,W'")
    return parts


def main():
    ir = BaseIR.load_ir(file='C:\\Users\\LvBuQY\\CIMA_TC\\CIMA_TC\\Compiler\\test\\ResNet50\\ResNet50_split_ir.yaml')
    built = build_torch_model_from_ir(ir)
    model = built.model
    weights = load_weights_file('C:\\Users\\LvBuQY\\CIMA_TC\\CIMA_TC\\Compiler\\test\\ResNet50\\ResNet50_split_weights.pt')
    load_weights_into_model(model, weights, module_name_map=built.module_name_map, strict=False)

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    n, c, h, w = _parse_shape('1,3,224,224')
    x = torch.randn(n, c, h, w)

    for i in range(1):
        y = model(x)
        # dummy scalar loss
        loss = (y.float() ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"step={i} loss={loss.item():.6f}")


if __name__ == "__main__":
    main()
