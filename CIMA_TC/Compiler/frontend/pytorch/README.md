# PyTorch frontend

Converts PyTorch `nn.Module` to IR via ONNX export. Kept separate from the ONNX frontend; uses it internally after `torch.onnx.export`.

## Requirements

- PyTorch
- ONNX (for export)
- Existing ONNX frontend dependencies

## Usage

```python
import torch
import torch.nn as nn
from CIMA_TC.Compiler.frontend.pytorch import ConvertTorch, TorchConversionConfig

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))

model = MyModel()
example_input = torch.randn(1, 3, 224, 224)

config = TorchConversionConfig(
    example_input=example_input,
    ir_file="model_ir.yaml",
    fix_layer_name=True,
)
converter = ConvertTorch(config)
ir = converter.convert(model)
converter.dump()
```

Or with kwargs:

```python
converter = ConvertTorch(example_input=example_input, ir_file="model_ir.yaml")
ir = converter.convert(model)
converter.dump()
```

## Dynamic axes

For variable batch size:

```python
config = TorchConversionConfig(
    example_input=example_input,
    dynamic_axes={
        "input": {0: "batch"},
        "output": {0: "batch"},
    },
)
```

## Layout

- `config.py` – `TorchConversionConfig` (example_input, ir_file, opset_version, etc.)
- `convert.py` – `ConvertTorch` (export to ONNX, then call ONNX frontend)
