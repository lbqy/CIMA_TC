"""
Structured Computation Graph IR
"""

from CIMA_TC.Compiler.IR_tool.core.datadef import DataDef
from typing import (
    Any, Dict, Iterator, List, Optional, Set, Tuple, Type, Union, ClassVar, Callable
)
from .reg import RegistryMixin, RegistryEntry
from .jsonable import Jsonable
from .type_utils import to_typed_dict, to_typed_object, is_integer, ValidationError, ConversionError
from .datadef import DataDef
from .ref import NameSegment, Ref, get_ref, require_ref
from .ns import ns_push
from .op import make_op, BaseOp


def _to_io_list(
    obj: Any,
    datadef: Type[DataDef] = DataDef,
) -> Optional[List[DataDef]]:
    """
    Convert inputs/outputs to List[DataDef].
    - None -> None
    - ['in1', 'in2'] -> [DataDef(ref='in1'), DataDef(ref='in2')]
    - [dict, ...] or [DataDef, ...] -> list of DataDef via to_typed_object
    """
    if obj is None:
        return None
    if isinstance(obj, (list, tuple)):
        out: List[DataDef] = []
        for item in obj:
            if isinstance(item, str):
                out.append(datadef(ref=item))
            else:
                out.append(to_typed_object(item, datadef))
        return out
    raise ConversionError(
        f"Cannot convert {type(obj).__name__} to input/output list; expect list of str or list of DataDef/dict"
    )


def _validate_input_ref(
    ref: Ref,
    consumer_name: str,
    layers: Dict[str, "BaseLayer"],
) -> None:
    """
    Validate that an input ref (ref_name or ref_name:index) refers to a layer in the graph.
    Connection is defined by inputs only; ref_name is the layer name, optional index is the
    branch. We only check that ref_name exists as a layer; outputs need not be declared.
    """
    if not ref.segments:
        raise ValidationError(f"layer {consumer_name!r} has empty ref")
    seg = ref.segments[0]
    layer_name = seg.name
    if layer_name not in layers:
        raise ValidationError(
            f"layer {consumer_name!r} inputs ref {ref!s} (layer {layer_name!r}) is not a layer in this graph"
        )


# ============================================================
# Core IR Layer
# ============================================================

class BaseLayer(Jsonable, RegistryMixin, RegistryEntry):
    """
    Base IR layer.
    """

    __registry_key__: ClassVar[str] = "type"
    __registry_default__: ClassVar[str] = "op"

    # Instance attribute type annotations (inputs/outputs: list of DataDef with ref = name)
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
        **kwargs: Any
    ) -> None:
        # allow Jsonable to consume extra kwargs
        super().__init__(**kwargs)

        self.set_attr("inputs", _to_io_list(inputs, datadef))
        self.set_attr("outputs", _to_io_list(outputs, datadef))
        self.set_attr("weights", to_typed_dict(weights, datadef))

    # --------------------------------------------------------

    def has_subgraph(self) -> bool:
        return False

    def iter_sublayers(self) -> Iterator[Tuple[str, "BaseLayer"]]:
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
        if self.inputs is not None and not isinstance(self.inputs, list):
            raise ValidationError("inputs must be list or None")

        if self.outputs is not None and not isinstance(self.outputs, list):
            raise ValidationError("outputs must be list or None")

        if self.weights is not None and not isinstance(self.weights, dict):
            raise ValidationError("weights must be dict or None")

        # recursive validation
        if self.has_subgraph():
            for _, layer in self.iter_sublayers():
                layer.validate()

    # --------------------------------------------------------

    def iter_inputs(self) -> Iterator[Tuple[str, DataDef]]:
        if self.inputs:
            for i, dd in enumerate[DataDef](self.inputs):
                name = str(dd.ref) if dd.ref is not None else str(i)
                with ns_push(f"inputs[{name!r}]"):
                    yield name, dd

    def iter_outputs(self) -> Iterator[Tuple[str, DataDef]]:
        if self.outputs:
            for i, dd in enumerate[DataDef](self.outputs):
                name = str(dd.ref) if dd.ref is not None else str(i)
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

class OpLayer(BaseLayer):

    type: ClassVar[str] = "op"
    op: Optional[BaseOp] = None

    def __init__(self, *, op: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.set_attr("op", make_op(op), not_none=True)

    def validate(self) -> None:
        super().validate()

        n = len(self.inputs or ())

        # When num_inputs is None (e.g. Concat), allow variable inputs
        if self.op.num_inputs is not None and n != self.op.num_inputs:
            raise ValidationError(f"Invalid number of inputs: {n}")


# ============================================================
# Graph Layer
# ============================================================

class GraphLayer(BaseLayer):

    type: ClassVar[str] = "graph"
    layers: Optional[Dict[str, BaseLayer]]

    def __init__(self, *, layers: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.set_attr(
            "layers",
            to_typed_dict(layers, BaseLayer, BaseLayer.create)
        )
        if self.layers:
            self._resolve_graph_connections()

    def _resolve_graph_connections(self) -> None:
        """
        Resolve graph connections so that:
        - inputs like ["in", "mid"] or ["mid:0", "mid:1"] mean "consume output of layer 'in' / branch 0 of 'mid' / branch 1 of 'mid'"
          (ref_name:index = layer ref_name, output index).
        - If a layer has no outputs, set outputs = [DataDef(ref=layer_name)].
        """
        if not self.layers:
            return
        for name, layer in self.layers.items():
            if not layer.outputs and not isinstance(layer, OutputLayer):
                layer.outputs = [DataDef(ref=name)]
            if layer.inputs:
                for dd in layer.inputs:
                    if dd.ref is not None:
                        _validate_input_ref(dd.ref, name, self.layers)

    # --------------------------------------------------------

    def has_subgraph(self) -> bool:
        return True

    def iter_sublayers(self) -> Iterator[Tuple[str, BaseLayer]]:
        if self.layers:
            for name, layer in self.layers.items():
                yield name, layer

    # --------------------------------------------------------

    def add_layer(
        self,
        name: str,
        layer: Optional[BaseLayer] = None,
        **kwargs: Any
    ) -> None:

        NameSegment.parse(name)

        if self.layers is None:
            self.layers = {}

        if name in self.layers:
            raise ValueError(f"layer {name!r} already exists")

        if layer is None:
            self.layers[name] = BaseLayer.create(kwargs)   # type: ignore
        elif isinstance(layer, BaseLayer):
            self.layers[name] = layer.clone(**kwargs)
        else:
            raise TypeError("invalid layer")
        added = self.layers[name]
        if not added.outputs and not isinstance(added, OutputLayer):
            added.outputs = [DataDef(ref=name)]
        if added.inputs:
            for dd in added.inputs:
                if dd.ref is not None:
                    _validate_input_ref(dd.ref, name, self.layers)

    # --------------------------------------------------------

    def get_layer(self, ref: str) -> BaseLayer:
        return get_ref(self, "layers", ref)   # type: ignore

    def require_layer(self, ref: str) -> BaseLayer:
        return require_ref(self, "layers", ref)   # type: ignore

    def set_layer_inputs(self, layer_name: str, inputs: List[Any]) -> None:
        """
        Set inputs of an existing layer (e.g. to rewire after inserting a new layer).
        inputs: list of ref strings or DataDef, same as layer inputs.
        """
        if self.layers is None or layer_name not in self.layers:
            raise ValueError(f"layer {layer_name!r} not in graph")
        layer = self.layers[layer_name]
        layer.set_attr("inputs", _to_io_list(inputs))
        if layer.inputs:
            for dd in layer.inputs:
                if dd.ref is not None:
                    _validate_input_ref(dd.ref, layer_name, self.layers)

    # --------------------------------------------------------
    # Topological order and connection views (inferred from inputs only)
    # --------------------------------------------------------

    def topological_order(self) -> List[str]:
        """
        Return layer names in topological order (consumers after producers).
        Edges are inferred from inputs only: if layer B has A in its inputs, A comes before B.
        """
        if not self.layers:
            return []
        layers = self.layers
        # successors[producer] = list of consumers that have producer in their inputs
        successors: Dict[str, List[str]] = {name: [] for name in layers}
        in_degree: Dict[str, int] = {name: 0 for name in layers}
        for name, layer in layers.items():
            if not layer.inputs:
                continue
            producers: Set[str] = set()
            for dd in layer.inputs:
                if dd.ref is not None and dd.ref.segments:
                    prod = dd.ref.segments[0].name
                    if prod in layers:
                        producers.add(prod)
            for prod in producers:
                successors[prod].append(name)
            in_degree[name] = len(producers)
        queue: List[str] = [n for n in layers if in_degree[n] == 0]
        order: List[str] = []
        while queue:
            n = queue.pop(0)
            order.append(n)
            for m in successors[n]:
                in_degree[m] -= 1
                if in_degree[m] == 0:
                    queue.append(m)
        return order

    def get_all_inputs(self) -> Dict[str, Optional[List[str]]]:
        """
        Return each layer's input refs as strings.
        Keys: layer names. Values: list of input ref strings (e.g. ['in', 'mid:0']), or None for InputLayer.
        """
        if not self.layers:
            return {}
        out: Dict[str, Optional[List[str]]] = {}
        for name, layer in self.layers.items():
            if isinstance(layer, InputLayer):
                out[name] = None
            elif layer.inputs:
                out[name] = [str(dd.ref) for dd in layer.inputs if dd.ref is not None]
            else:
                out[name] = []
        return out

    def get_all_outputs(self) -> Dict[str, Optional[List[str]]]:
        """
        Return each layer's consumers (layers that use this layer as input), inferred from inputs only.
        Keys: layer names. Values: list of layer names that have this layer in their inputs, or None for OutputLayer.
        """
        if not self.layers:
            return {}
        out: Dict[str, Optional[List[str]]] = {name: [] for name in self.layers}
        for name, layer in self.layers.items():
            if isinstance(layer, OutputLayer):
                out[name] = None
            if not layer.inputs:
                continue
            for dd in layer.inputs:
                if dd.ref is not None and dd.ref.segments:
                    prod = dd.ref.segments[0].name
                    if prod in self.layers and out[prod] is not None and name not in out[prod]:
                        out[prod].append(name)
        return out

    # --------------------------------------------------------

    def validate(self) -> None:
        super().validate()

        if not self.layers:
            return

        input_layer_names = []
        output_layer_names = []

        for name, layer in self.layers.items():
            if isinstance(layer, InputLayer):
                input_layer_names.append(name)
            if isinstance(layer, OutputLayer):
                output_layer_names.append(name)

        if len(input_layer_names) == 0:
            raise ValidationError("graph must contain at least one InputLayer")

        if len(output_layer_names) == 0:
            raise ValidationError("graph must contain at least one OutputLayer")

        # Connection check: input refs (ref_name or ref_name:index) refer to layers in this graph
        for name, layer in self.layers.items():
            if layer.inputs:
                for dd in layer.inputs:
                    if dd.ref is not None:
                        _validate_input_ref(dd.ref, name, self.layers)


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

class IOLayer(BaseLayer):

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

        # outputs may be None when built as part of a graph; GraphLayer._resolve_graph_connections() fills them.
        # If outputs is explicitly provided as an empty list, treat it as invalid.
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


# ============================================================
# Factory Shortcut
# ============================================================

make_layer: Callable[..., BaseLayer] = BaseLayer.create