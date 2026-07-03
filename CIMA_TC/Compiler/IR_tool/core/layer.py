"""Structured computation graph IR layers."""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, ClassVar, Dict, Iterator, List, Optional, Set, Tuple, Type

from .datadef import DataDef
from .jsonable import Jsonable
from .ns import ns_push
from .op import BaseOp, make_op
from .ref import NameSegment, Ref, get_ref, require_ref
from .reg import RegistryEntry, RegistryMixin
from .type_utils import ConversionError, ValidationError, is_integer, to_typed_dict, to_typed_object


def _to_io_list(obj: Any, datadef: Type[DataDef] = DataDef) -> Optional[List[DataDef]]:
    """Convert layer input/output specs to a list of DataDef objects."""
    if obj is None:
        return None
    if not isinstance(obj, (list, tuple)):
        raise ConversionError(
            f"Cannot convert {type(obj).__name__} to input/output list; "
            "expect list of str or list of DataDef/dict"
        )

    out: List[DataDef] = []
    for item in obj:
        out.append(datadef(ref=item) if isinstance(item, str) else to_typed_object(item, datadef))
    return out


def _producer_name(ref: Ref) -> str:
    if not ref.segments:
        raise ValidationError("empty ref")
    return ref.segments[0].name


def _validate_input_ref(ref: Ref, consumer_name: str, layers: Dict[str, "BaseLayer"]) -> None:
    """Validate that an input ref points to a layer in this graph."""
    try:
        layer_name = _producer_name(ref)
    except ValidationError as exc:
        raise ValidationError(f"layer {consumer_name!r} has empty ref") from exc
    if layer_name not in layers:
        raise ValidationError(
            f"layer {consumer_name!r} inputs ref {ref!s} "
            f"(layer {layer_name!r}) is not a layer in this graph"
        )


def _iter_input_refs(layer: "BaseLayer") -> Iterator[Ref]:
    for dd in layer.inputs or []:
        if dd.ref is not None:
            yield dd.ref


def _ensure_default_output(name: str, layer: "BaseLayer") -> None:
    if not layer.outputs and not isinstance(layer, OutputLayer):
        layer.outputs = [DataDef(ref=name)]


def _validate_layer_inputs(name: str, layer: "BaseLayer", layers: Dict[str, "BaseLayer"]) -> None:
    for ref in _iter_input_refs(layer):
        _validate_input_ref(ref, name, layers)


class BaseLayer(Jsonable, RegistryMixin, RegistryEntry):
    """Base IR layer with optional inputs, outputs, and weights."""

    __registry_key__: ClassVar[str] = "type"
    __registry_default__: ClassVar[str] = "op"

    inputs: Optional[List[DataDef]]
    outputs: Optional[List[DataDef]]
    weights: Optional[Dict[str, DataDef]]

    def __init__(
        self,
        *,
        inputs: Optional[List[Any]] = None,
        outputs: Optional[List[Any]] = None,
        weights: Optional[Dict[str, Any]] = None,
        datadef: Type[DataDef] = DataDef,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if "type" not in self.__dict__ and hasattr(self.__class__, "type"):
            self.type = getattr(self.__class__, "type")
        self.set_attr("inputs", _to_io_list(inputs, datadef))
        self.set_attr("outputs", _to_io_list(outputs, datadef))
        self.set_attr("weights", to_typed_dict(weights, datadef))

    def has_subgraph(self) -> bool:
        return False

    def iter_sublayers(self) -> Iterator[Tuple[str, "BaseLayer"]]:
        return iter(())

    def validate(self) -> None:
        super().validate()
        if self.inputs is not None and not isinstance(self.inputs, list):
            raise ValidationError("inputs must be list or None")
        if self.outputs is not None and not isinstance(self.outputs, list):
            raise ValidationError("outputs must be list or None")
        if self.weights is not None and not isinstance(self.weights, dict):
            raise ValidationError("weights must be dict or None")
        for _, layer in self.iter_sublayers():
            layer.validate()

    def iter_inputs(self) -> Iterator[Tuple[str, DataDef]]:
        yield from self._iter_io("inputs", self.inputs)

    def iter_outputs(self) -> Iterator[Tuple[str, DataDef]]:
        yield from self._iter_io("outputs", self.outputs)

    def iter_weights(self) -> Iterator[Tuple[str, DataDef]]:
        for name, dd in (self.weights or {}).items():
            with ns_push(f"weights[{name!r}]"):
                yield name, dd

    @staticmethod
    def _iter_io(kind: str, values: Optional[List[DataDef]]) -> Iterator[Tuple[str, DataDef]]:
        for index, dd in enumerate(values or []):
            name = str(dd.ref) if dd.ref is not None else str(index)
            with ns_push(f"{kind}[{name!r}]"):
                yield name, dd


class OpLayer(BaseLayer):
    """Layer wrapping one registered IR operator."""

    type: ClassVar[str] = "op"
    op: Optional[BaseOp] = None

    def __init__(self, *, op: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.set_attr("op", make_op(op), not_none=True)

    def validate(self) -> None:
        super().validate()
        if self.op.num_inputs is not None and len(self.inputs or ()) != self.op.num_inputs:
            raise ValidationError(f"Invalid number of inputs: {len(self.inputs or ())}")


class GraphLayer(BaseLayer):
    """Layer containing a named subgraph."""

    type: ClassVar[str] = "graph"
    layers: Optional[Dict[str, BaseLayer]]

    def __init__(self, *, layers: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.set_attr("layers", to_typed_dict(layers, BaseLayer, BaseLayer.create))
        self._resolve_graph_connections()

    def _resolve_graph_connections(self) -> None:
        if not self.layers:
            return
        for name, layer in self.layers.items():
            _ensure_default_output(name, layer)
            _validate_layer_inputs(name, layer, self.layers)

    def has_subgraph(self) -> bool:
        return True

    def iter_sublayers(self) -> Iterator[Tuple[str, BaseLayer]]:
        yield from (self.layers or {}).items()

    def add_layer(self, name: str, layer: Optional[BaseLayer] = None, **kwargs: Any) -> None:
        NameSegment.parse(name)
        if self.layers is None:
            self.layers = {}
        if name in self.layers:
            raise ValueError(f"layer {name!r} already exists")

        added = BaseLayer.create(kwargs) if layer is None else self._clone_layer(layer, **kwargs)
        self.layers[name] = added
        _ensure_default_output(name, added)
        _validate_layer_inputs(name, added, self.layers)

    @staticmethod
    def _clone_layer(layer: BaseLayer, **kwargs: Any) -> BaseLayer:
        if not isinstance(layer, BaseLayer):
            raise TypeError("invalid layer")
        return layer.clone(**kwargs)

    def get_layer(self, ref: str) -> BaseLayer:
        return get_ref(self, "layers", ref)  # type: ignore[return-value]

    def require_layer(self, ref: str) -> BaseLayer:
        return require_ref(self, "layers", ref)  # type: ignore[return-value]

    def set_layer_inputs(self, layer_name: str, inputs: List[Any]) -> None:
        if self.layers is None or layer_name not in self.layers:
            raise ValueError(f"layer {layer_name!r} not in graph")
        layer = self.layers[layer_name]
        layer.set_attr("inputs", _to_io_list(inputs))
        _validate_layer_inputs(layer_name, layer, self.layers)

    def topological_order(self) -> List[str]:
        """Return layer names in producer-before-consumer order."""
        if not self.layers:
            return []
        successors, in_degree = self._dependency_graph()
        queue = deque(name for name in self.layers if in_degree[name] == 0)
        order: List[str] = []

        while queue:
            name = queue.popleft()
            order.append(name)
            for consumer in successors[name]:
                in_degree[consumer] -= 1
                if in_degree[consumer] == 0:
                    queue.append(consumer)
        return order

    def _dependency_graph(self) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
        layers = self.layers or {}
        successors: Dict[str, List[str]] = {name: [] for name in layers}
        in_degree: Dict[str, int] = {name: 0 for name in layers}

        for name, layer in layers.items():
            producers = self._input_producers(layer)
            in_degree[name] = len(producers)
            for producer in producers:
                successors[producer].append(name)
        return successors, in_degree

    def _input_producers(self, layer: BaseLayer) -> Set[str]:
        layers = self.layers or {}
        producers: Set[str] = set()
        for ref in _iter_input_refs(layer):
            producer = _producer_name(ref)
            if producer in layers:
                producers.add(producer)
        return producers

    def get_all_inputs(self) -> Dict[str, Optional[List[str]]]:
        if not self.layers:
            return {}
        out: Dict[str, Optional[List[str]]] = {}
        for name, layer in self.layers.items():
            if isinstance(layer, InputLayer):
                out[name] = None
            else:
                out[name] = [str(ref) for ref in _iter_input_refs(layer)]
        return out

    def get_all_outputs(self) -> Dict[str, Optional[List[str]]]:
        if not self.layers:
            return {}
        out: Dict[str, Optional[List[str]]] = {name: [] for name in self.layers}
        for name, layer in self.layers.items():
            if isinstance(layer, OutputLayer):
                out[name] = None
            for producer in self._input_producers(layer):
                consumers = out.get(producer)
                if consumers is not None and name not in consumers:
                    consumers.append(name)
        return out

    def validate(self) -> None:
        super().validate()
        if not self.layers:
            return
        if not any(isinstance(layer, InputLayer) for layer in self.layers.values()):
            raise ValidationError("graph must contain at least one InputLayer")
        if not any(isinstance(layer, OutputLayer) for layer in self.layers.values()):
            raise ValidationError("graph must contain at least one OutputLayer")
        for name, layer in self.layers.items():
            _validate_layer_inputs(name, layer, self.layers)


class BlockLayer(GraphLayer):
    """Repeatable graph block."""

    type: ClassVar[str] = "block"
    repeat: int

    def __init__(self, *, repeat: Optional[int] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.set_attr("repeat", repeat)

    def validate(self) -> None:
        super().validate()
        if not is_integer(self.repeat, min_val=1):
            raise ValidationError(f"Invalid value for repeat: {self.repeat}")

    def is_single(self) -> bool:
        return self.repeat == 1


class IOLayer(BaseLayer):
    """Base class for graph input/output marker layers."""

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
        if self.outputs is not None and len(self.outputs) == 0:
            raise ValidationError("Input layer must have at least one output when set")


class OutputLayer(IOLayer):
    type: ClassVar[str] = "output"

    def validate(self) -> None:
        super().validate()
        if not self.inputs:
            raise ValidationError("Output layer must have at least one input")
        if self.outputs:
            raise ValidationError("Output layer cannot have outputs")


make_layer: Callable[..., BaseLayer] = BaseLayer.create
