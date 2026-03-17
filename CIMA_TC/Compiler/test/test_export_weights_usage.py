"""
weights/state_dict export usage example and simple verification.

IR only keeps model structure; conv/fc weights and BN parameters are exported separately through export_weights.

Summary of usage:
---------
1) ONNX path (.onnx -> IR):     
   converter = ConvertONNX(config)
   converter.convert(onnx_file=...)
   converter.export_weights("weights.pt")   # from parser.weight_numpy, key is IR layer name.parameter name
   # Load: w = torch.load("weights.pt")   # dict[str, Tensor]

2) PyTorch -> ONNX -> IR path:
   converter = ConvertTorch(config)
   converter.convert(model, example_input=...)
   converter.export_weights("weights.pt")  # PyTorch state_dict
   # Load: sd = torch.load("weights.pt")   # state_dict, key is module path

3) PyTorch FX (direct trace -> IR):
   converter = ConvertFX(config)
   converter.convert(model)
   converter.export_weights("weights.pt")  # PyTorch state_dict
   # Load: sd = torch.load("weights.pt")

Optional: pass custom state_dict: converter.export_weights(path, state_dict=my_state_dict)

Numpy formats:
- .npz: converter.export_weights("weights.npz") or export_weights(..., format="npz")
  Load: d = np.load("weights.npz"); d.files lists keys ('.' is replaced by '_'); d["conv_weight"], etc.
- .npy: only supported for a single array; use .npz for multiple arrays.
"""

import os


def test_fx_export_weights_load() -> None:
    """FX path: export weights and verify torch.load result."""
    import torch
    import torch.nn as nn

    from ..frontend.pytorch_fx import ConvertFX, FXConversionConfig

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(2, 4, 2)
            self.fc = nn.Linear(4 * 3 * 3, 2)

        def forward(self, x):
            x = self.conv(x)
            x = x.flatten(1)
            return self.fc(x)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PyTorchFXExample")
    os.makedirs(out_dir, exist_ok=True)
    weights_path = os.path.join(out_dir, "test_fx_weights.pt")

    model = Tiny()
    example = torch.randn(1, 2, 4, 4)
    config = FXConversionConfig(example_input=example)
    converter = ConvertFX(config)
    converter.convert(model)
    converter.export_weights(weights_path)

    loaded = torch.load(weights_path)
    assert "conv.weight" in loaded and "fc.weight" in loaded
    assert loaded["conv.weight"].shape == (4, 2, 2, 2)
    assert loaded["fc.weight"].shape == (2, 4 * 3 * 3)
    print("FX export_weights + load OK")


def test_fx_export_weights_npz() -> None:
    """FX path: export .npz and verify np.load result."""
    import numpy as np
    import torch
    import torch.nn as nn

    from ..frontend.pytorch_fx import ConvertFX, FXConversionConfig

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(2, 4, 2)
            self.fc = nn.Linear(4 * 3 * 3, 2)

        def forward(self, x):
            x = self.conv(x)
            x = x.flatten(1)
            return self.fc(x)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PyTorchFXExample")
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, "test_fx_weights.npz")

    model = Tiny()
    example = torch.randn(1, 2, 4, 4)
    config = FXConversionConfig(example_input=example)
    converter = ConvertFX(config)
    converter.convert(model)
    converter.export_weights(npz_path, format="npz")

    d = np.load(npz_path)
    assert "conv_weight" in d.files and "fc_weight" in d.files
    assert d["conv_weight"].shape == (4, 2, 2, 2)
    assert d["fc_weight"].shape == (2, 4 * 3 * 3)
    print("FX export_weights(.npz) + np.load OK")


if __name__ == "__main__":
    test_fx_export_weights_load()
    test_fx_export_weights_npz()
    print("test_export_weights_usage done")
