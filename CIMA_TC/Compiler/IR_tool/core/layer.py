"""
Structured Computation Graph IR
"""

from typing import (
    Any, Dict, Iterator, Optional, Tuple, Type, Union, ClassVar, Callable
)
from .reg import RegistryMixin, RegistryEntry
from .jsonable import Jsonable
from .type_utils import to_typed_dict, is_integer, ValidationError
from .datadef import DataDef
from .ref import NameSegment, get_ref, require_ref
from .ns import ns_push
from .op import make_op, BaseOp  
# ============================================================
# Core IR Layer
# ============================================================

class IRLayer(Jsonable, RegistryMixin, RegistryEntry):
    """
    Base IR layer.
    """

    __registry_key__: ClassVar[str] = "type"
    __registry_default__: ClassVar[str] = "op"

    # 实例属性类型注解
    inputs: Optional[Dict[str, DataDef]]
    outputs: Optional[Dict[str, DataDef]]
    weights: Optional[Dict[str, DataDef]]

    def __init__(
        self,
        *,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        weights: Optional[Dict[str, Any]] = None,
        datadef: Type[DataDef] = DataDef,
        **kwargs: Any
    ) -> None:
        # allow Jsonable to consume extra kwargs
        super().__init__(**kwargs)

        self.set_attr("inputs", to_typed_dict(inputs, datadef))
        self.set_attr("outputs", to_typed_dict(outputs, datadef))
        self.set_attr("weights", to_typed_dict(weights, datadef))

    # --------------------------------------------------------

    def has_subgraph(self) -> bool:
        return False

    def iter_sublayers(self) -> Iterator[Tuple[str, "IRLayer"]]:
        return iter(())

    # ========================================================
    # Unified Validation Entry
    # ========================================================

    def validate(self) -> None:
        """
        Full validation entry.
        """

        # base-level validation hook
        super().validate()

        # structure check
        if self.inputs is not None and not isinstance(self.inputs, dict):
            raise ValidationError("inputs must be dict or None")

        if self.outputs is not None and not isinstance(self.outputs, dict):
            raise ValidationError("outputs must be dict or None")

        if self.weights is not None and not isinstance(self.weights, dict):
            raise ValidationError("weights must be dict or None")

        # recursive validation
        if self.has_subgraph():
            for _, layer in self.iter_sublayers():
                layer.validate()

    # --------------------------------------------------------

    def iter_inputs(self) -> Iterator[Tuple[str, DataDef]]:
        if self.inputs:
            for name, dd in self.inputs.items():
                with ns_push(f"inputs[{name!r}]"):
                    yield name, dd

    def iter_outputs(self) -> Iterator[Tuple[str, DataDef]]:
        if self.outputs:
            for name, dd in self.outputs.items():
                with ns_push(f"outputs[{name!r}]"):
                    yield name, dd

    def iter_weights(self) -> Iterator[Tuple[str, DataDef]]:
        if self.weights:
            for name, dd in self.weights.items():
                with ns_push(f"weights[{name!r}]"):
                    yield name, dd


# ============================================================
# Operator Layer
# ============================================================

class OpLayer(IRLayer):

    type: ClassVar[str] = "op"
    op: Optional[BaseOp] = None

    def __init__(self, *, op: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.set_attr("op", make_op(op), not_none=True)

    def validate(self) -> None:
        super().validate()

        # must have at least one input
        n = len(self.inputs or ())

        if n != self.op.num_inputs:
            raise ValidationError(f"Invalid number of inputs: {n}")


# ============================================================
# Graph Layer
# ============================================================

class GraphLayer(IRLayer):

    type: ClassVar[str] = "graph"
    layers: Optional[Dict[str, IRLayer]]

    def __init__(self, *, layers: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.set_attr(
            "layers",
            to_typed_dict(layers, IRLayer, IRLayer.create)
        )

    # --------------------------------------------------------

    def has_subgraph(self) -> bool:
        return True

    def iter_sublayers(self) -> Iterator[Tuple[str, IRLayer]]:
        if self.layers:
            for name, layer in self.layers.items():
                yield name, layer

    # --------------------------------------------------------

    def add_layer(
        self,
        name: str,
        layer: Optional[IRLayer] = None,
        **kwargs: Any
    ) -> None:

        NameSegment.parse(name)

        if self.layers is None:
            self.layers = {}

        if name in self.layers:
            raise ValueError(f"layer {name!r} already exists")

        if layer is None:
            self.layers[name] = IRLayer.create(kwargs)   # type: ignore
        elif isinstance(layer, IRLayer):
            self.layers[name] = layer.clone(**kwargs)
        else:
            raise TypeError("invalid layer")

    # --------------------------------------------------------

    def get_layer(self, ref: str) -> IRLayer:
        return get_ref(self, "layers", ref)   # type: ignore

    def require_layer(self, ref: str) -> IRLayer:
        return require_ref(self, "layers", ref)   # type: ignore

    # --------------------------------------------------------

    def validate(self) -> None:
        super().validate()

        if not self.layers:
            return

        inputs = []
        outputs = []

        for name, layer in self.layers.items():
            if isinstance(layer, InputLayer):
                inputs.append(name)
            if isinstance(layer, OutputLayer):
                outputs.append(name)

        if len(inputs) == 0:
            raise ValidationError("graph must contain at least one InputLayer")

        if len(outputs) == 0:
            raise ValidationError("graph must contain at least one OutputLayer")


# ============================================================
# Block Layer
# ============================================================

class BlockLayer(GraphLayer):

    type: ClassVar[str] = "block"
    repeat: int

    def __init__(self, *, repeat: Optional[int] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.set_attr("repeat", repeat)

    def validate(self) -> None:
        super().validate()

        if not is_integer(self.repeat, min_val=1):
            raise ValidationError(
                f"Invalid value for repeat: {self.repeat}"
            )

    def is_single(self) -> bool:
        return self.repeat == 1


# ============================================================
# IO Layers
# ============================================================

class IOLayer(IRLayer):

    __abstract__: ClassVar[bool] = True

    def validate(self) -> None:
        super().validate()

        if self.weights:
            raise ValidationError("IO layer cannot have weights")

        if self.has_subgraph():
            raise ValidationError("IO layer cannot have subgraphs")


class InputLayer(IOLayer):

    type: ClassVar[str] = "input"

    def validate(self) -> None:
        super().validate()

        if self.inputs:
            raise ValidationError("Input layer cannot have inputs")

        if not self.outputs:
            raise ValidationError("Input layer must have at least one output")


class OutputLayer(IOLayer):

    type: ClassVar[str] = "output"

    def validate(self) -> None:
        super().validate()

        if not self.inputs:
            raise ValidationError("Output layer must have at least one input")

        if self.outputs:
            raise ValidationError("Output layer cannot have outputs")


# ============================================================
# Factory Shortcut
# ============================================================

make_layer: Callable[..., IRLayer] = IRLayer.create