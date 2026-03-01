"""
Layer registration module based on the new generic registry framework.

This module defines a hierarchical layer system used to build
computational graphs.

Key features:
- Automatic subclass registration using RegistryMixin
- Factory-based object creation
- Graph validation utilities
- Nested layer tree support
- IO layer detection
- Block layer repetition support

All layer subclasses are automatically registered using
the class attribute specified by `__registry_key__ = "type"`.
"""

from contextlib import contextmanager

from .reg import RegistryMixin, RegistryEntry
from .jsonable import Jsonable
from .type_utils import to_obj_dict, is_integer
from .op import make_op
from .ref import DataDef
from .ref import is_valid_name, query_tree_ref, make_ref
from .ns import ns_push


# ============================================================
# Base Layer
# ============================================================

class BaseLayer(Jsonable, RegistryMixin, RegistryEntry):
    """
    Base class for all layer types.

    This class serves as the registry root for all layer subclasses.

    Registration behavior:
        - Each subclass must define a class attribute `type`
        - The value of `type` is used as the registry key
        - Subclasses are automatically registered

    Class-level registry configuration:
        __registry_key__ = "type"
        __registry_default__ = "op"

    Attributes:
        inputs (list or None): Input DataDef objects.
        outputs (list or None): Output DataDef objects.
        weights (dict or None): Weight DataDef mapping.
    """

    __registry_key__ = "type"
    __registry_default__ = "op"

    inputs = None
    outputs = None
    weights = None

    def __init__(self, *, inputs=None, outputs=None,
                 weights=None, datadef=DataDef, **kwargs):
        """
        Initialize a layer.

        Args:
            inputs (list, optional): Input data definitions.
            outputs (list, optional): Output data definitions.
            weights (dict, optional): Weight definitions.
            datadef (type): Data definition class used for conversion.
            **kwargs: Additional keyword arguments passed to Jsonable.
        """
        super().__init__(**kwargs)

        self.set_attr("inputs", to_obj_dict(inputs, datadef))
        self.set_attr("outputs", to_obj_dict(outputs, datadef))
        self.set_attr("weights", to_obj_dict(weights, datadef))

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    def validate(self):
        """
        Validate structural correctness of the layer.
        """
        super().validate()

        assert self.inputs is None or isinstance(self.inputs, list)
        assert self.outputs is None or isinstance(self.outputs, list)
        assert self.weights is None or isinstance(self.weights, dict)

    def validate_graph(self):
        """
        Validate graph-level semantics.

        This method should be overridden by subclasses
        that enforce graph constraints.
        """
        pass

    # --------------------------------------------------------
    # Type Identification
    # --------------------------------------------------------

    def is_layer_tree(self):
        """Return True if this layer is a nested LayerTree."""
        return False

    def is_io_layer(self):
        """Return True if this layer is an input/output layer."""
        return False

    # --------------------------------------------------------
    # Iterators
    # --------------------------------------------------------

    def iter_inputs(self):
        """
        Iterate over input DataDefs with namespace tracking.

        Yields:
            (index, DataDef)
        """
        if self.inputs:
            for i, dd in enumerate(self.inputs):
                with ns_push(f"inputs[{i}]"):
                    yield i, dd

    def iter_outputs(self):
        """
        Iterate over output DataDefs with namespace tracking.

        Yields:
            (index, DataDef)
        """
        if self.outputs:
            for i, dd in enumerate(self.outputs):
                with ns_push(f"outputs[{i}]"):
                    yield i, dd

    def iter_weights(self):
        """
        Iterate over weight DataDefs with namespace tracking.

        Yields:
            (name, DataDef)
        """
        if self.weights:
            for name, dd in self.weights.items():
                with ns_push(f"weights[{name!r}]"):
                    yield name, dd


# ============================================================
# Operator Layer
# ============================================================

class OpLayer(BaseLayer):
    """
    Layer wrapping a registered operator instance.

    type = "op" (default layer type)
    """

    type = "op"
    op = None

    def __init__(self, *, op, **kwargs):
        """
        Args:
            op (str | dict | BaseOp): Operator specification.
        """
        super().__init__(**kwargs)
        self.set_attr("op", make_op(op), not_none=True)

    def validate(self):
        super().validate()
        assert not hasattr(self, "layers")

    def validate_graph(self):
        """
        Validate number of inputs according to operator spec.
        """
        n = len(self.inputs or ())

        if self.op.num_inputs is None:
            assert n > 1, f"invalid {n} inputs"
        else:
            assert n == self.op.num_inputs, \
                f"invalid {n} inputs, expects {self.op.num_inputs}"


# ============================================================
# Layer Tree (Graph Container)
# ============================================================

class LayerTree(Jsonable):
    """
    A hierarchical container of layers.

    Supports:
    - Nested layer trees
    - Graph validation
    - IO layer detection
    - Topological sorting
    """

    layers = None

    def __init__(self, *, layers=None, **kwargs):
        super().__init__(**kwargs)

        if layers is None:
            layers = self.layers

        self.set_attr(
            "layers",
            to_obj_dict(layers, BaseLayer, BaseLayer.create)
        )

    def is_layer_tree(self):
        return True

    # --------------------------------------------------------
    # Layer Management
    # --------------------------------------------------------

    def add_layer(self, name, layer=None, **kwargs):
        """
        Add a new layer into the tree.

        Args:
            name (str): Layer name.
            layer (BaseLayer, optional): Existing layer instance.
            **kwargs: Arguments used to create a new layer.
        """
        if self.layers is None:
            self.layers = {}

        assert is_valid_name(name), f"invalid layer name={name!r}"
        assert name not in self.layers, f"layer name={name!r} exists"

        if layer is None:
            self.layers[name] = BaseLayer.create(kwargs)
        elif isinstance(layer, BaseLayer):
            self.layers[name] = layer.clone(**kwargs)
        else:
            raise TypeError(f"invalid layer={layer!r}")

    def get_layer(self, ref):
        """
        Retrieve layer by reference path.
        """
        return query_tree_ref(self, "layers", ref)

    # --------------------------------------------------------
    # Graph Validation
    # --------------------------------------------------------

    def validate_graph(self):
        """
        Validate structural graph correctness.
        """
        if not self.layers:
            return

        inps, oups = self.find_io_layers()

        assert len(inps) == 1, f"invalid {len(inps)} input layers"
        assert len(oups) == 1, f"invalid {len(oups)} output layers"

        for key, layer in self.layers.items():
            layer.validate_graph()

    # --------------------------------------------------------
    # Utilities
    # --------------------------------------------------------

    def find_io_layers(self):
        """
        Find input and output layer names.

        Returns:
            (list[str], list[str])
        """
        inps, oups = [], []

        if self.layers:
            for name, layer in self.layers.items():
                if layer.type == "input":
                    inps.append(name)
                elif layer.type == "output":
                    oups.append(name)

        return inps, oups


# ============================================================
# Block Layer
# ============================================================

class BlockLayer(LayerTree, BaseLayer):
    """
    A repeatable block layer.

    type = "block"
    """

    type = "block"
    number = 1

    def __init__(self, *, number=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr("number", number, is_integer, min_val=1)

    def validate(self):
        super().validate()
        assert self.number > 0

    def is_simple_graph(self):
        return self.number == 1


# ============================================================
# IO Layers
# ============================================================

class IOLayer(BaseLayer):
    """
    Base class for input/output layers.
    """

    def validate(self):
        super().validate()
        assert self.weights is None
        assert not hasattr(self, "layers")
        assert not hasattr(self, "op")

    def is_io_layer(self):
        return True

    def validate_graph(self):
        assert len(self.inputs or ()), "empty inputs is invalid"


class InputLayer(IOLayer):
    type = "input"


class OutputLayer(IOLayer):
    type = "output"


# ============================================================
# Reuse Layer
# ============================================================

class ReuseLayer(BaseLayer):
    """
    Layer that references another layer.
    """

    type = "reuse"
    layer = None

    def __init__(self, *, layer=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr("layer", layer, is_valid_name, not_none=True)

    def validate(self):
        super().validate()
        assert self.weights is None


# ============================================================
# Factory Shortcut
# ============================================================

make_layer = BaseLayer.create