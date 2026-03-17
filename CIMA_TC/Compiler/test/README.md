# Frontend test cases

## Cases

- **CustomizedNet**: `CustomizedNet/CustomizedNet.onnx`
- **ResNet50**: `ResNet50/ResNet50.onnx`

## Run ONNX → IR (YAML) export

From **project root** (parent of `CIMA_TC`):

```bash
# CustomizedNet
python -m CIMA_TC.Compiler.frontend.test.export_ir_customized_net

# ResNet50
python -m CIMA_TC.Compiler.frontend.test.export_ir_resnet50
```

Or from this `test` directory:

```bash
python export_ir_customized_net.py
python export_ir_resnet50.py
```

Output YAML files:

- `CustomizedNet/CustomizedNet_ir.yaml`
- `ResNet50/ResNet50_ir.yaml`
