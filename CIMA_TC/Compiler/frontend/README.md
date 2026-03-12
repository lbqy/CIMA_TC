# Frontend (ONNX → IR)

Refactored ONNX-to-IR conversion with:

- **Config-driven**: `ConversionConfig` holds options; conversion runs via `convert()`.
- **Op registry**: Handlers registered by ONNX `op_type`; extend via `@register_op("OpType")`.
- **Typed parser**: `OnnxParser` builds value_infos, parameters, nodes, predecessors/successors, weight_numpy.
- **MatMul split**: Static weight → `linear`/`fc` (one input); two dynamic inputs → `matmul`.
- **Preprocess**: Shape inference, add value_info for constants, MeaninglessOpPass (remove zero Pad/Add), optional name fixing.

## Usage

```python
from CIMA_TC.Compiler.frontend import ConvertONNX, ConversionConfig

config = ConversionConfig(onnx_file="model.onnx", ir_file="model.yaml", fix_layer_name=False)
converter = ConvertONNX(config)
ir = converter.convert()
converter.dump()
```

Or with kwargs:

```python
converter = ConvertONNX(onnx_file="model.onnx", ir_file="model.yaml")
ir = converter.convert()
```

## Layout

- `config.py` – `ConversionConfig` dataclass.
- `parser.py` – `OnnxParser` (value_infos, parameters, nodes, weight_numpy, constant, predecessors/successors).
- `preprocess.py` – `load_onnx_model` (shape inference, MeaninglessOpPass, optional fix_node_name).
- `converter.py` – `ConvertONNX` (convert(), dump()).
- `utils/` – shape_utils, attr_reader, onnx_io.
- `op_handlers/` – registry, common (resolve_ref, ir_inputs_for_node), handlers (Conv, MatMul, Gemm, Add, …).

## Adding an op

In `op_handlers/handlers.py`:

```python
from .registry import register_op

@register_op("NewOp")
def _new_op(ir, parser, node_name):
    node = parser.nodes[node_name]
    # ... build op, inputs, outputs
    ir.add_layer(node_name, type="op", op=op, inputs=inputs, outputs=outputs)
```

Then import `op_handlers.handlers` so the decorator runs (already done in `op_handlers/__init__.py`).
