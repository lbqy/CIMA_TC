"""
Structured Computation Graph IR
"""

from .reg import RegistryMixin, RegistryEntry
from .jsonable import Jsonable
from .type_utils import to_typed_dict, is_integer, ValidationError
from .datadef import DataDef
from .ref import NameSegment, get_ref, require_ref
from .ns import ns_push
from .op import make_op


# ============================================================
# Core IR Node
# ============================================================

class IRNode(Jsonable, RegistryMixin, RegistryEntry):
    """
    Base IR node.
    """

    __registry_key__ = "type"
    __registry_default__ = "op"

    inputs = None
    outputs = None
    weights = None

    def __init__(
        self,
        *,
        inputs=None,
        outputs=None,
        weights=None,
        datadef=DataDef,
        **kwargs
    ):
        # allow Jsonable to consume extra kwargs
        super().__init__(**kwargs)

        self.set_attr("inputs", to_typed_dict(inputs, datadef))
        self.set_attr("outputs", to_typed_dict(outputs, datadef))
        self.set_attr("weights", to_typed_dict(weights, datadef))

    # --------------------------------------------------------

    def has_subgraph(self) -> bool:
        return False

    def iter_subnodes(self):
        return ()

    # ========================================================
    # Unified Validation Entry
    # ========================================================

    def validate(self):
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
            for _, node in self.iter_subnodes():
                node.validate()

    # --------------------------------------------------------

    def iter_inputs(self):
        if self.inputs:
            for name, dd in self.inputs.items():
                with ns_push(f"inputs[{name!r}]"):
                    yield name, dd

    def iter_outputs(self):
        if self.outputs:
            for name, dd in self.outputs.items():
                with ns_push(f"outputs[{name!r}]"):
                    yield name, dd

    def iter_weights(self):
        if self.weights:
            for name, dd in self.weights.items():
                with ns_push(f"weights[{name!r}]"):
                    yield name, dd


# ============================================================
# Operator Node
# ============================================================

class OpNode(IRNode):

    type = "op"
    op = None

    def __init__(self, *, op, **kwargs):
        super().__init__(**kwargs)
        self.set_attr("op", make_op(op), not_none=True)

    def validate(self):
        super().validate()

        # must have at least one input
        n = len(self.inputs or {})
        if not is_integer(n, min_val=1):
            raise ValidationError(f"Invalid number of inputs: {n}")


# ============================================================
# Graph Node
# ============================================================

class GraphNode(IRNode):

    type = "graph"
    nodes = None

    def __init__(self, *, nodes=None, **kwargs):
        super().__init__(**kwargs)

        self.set_attr(
            "nodes",
            to_typed_dict(nodes, IRNode, IRNode.create)
        )

    # --------------------------------------------------------

    def has_subgraph(self):
        return True

    def iter_subnodes(self):
        if self.nodes:
            for name, node in self.nodes.items():
                yield name, node

    # --------------------------------------------------------

    def add_node(self, name, node=None, **kwargs):

        NameSegment.parse(name)

        if self.nodes is None:
            self.nodes = {}

        if name in self.nodes:
            raise ValueError(f"node {name!r} already exists")

        if node is None:
            self.nodes[name] = IRNode.create(kwargs)
        elif isinstance(node, IRNode):
            self.nodes[name] = node.clone(**kwargs)
        else:
            raise TypeError("invalid node")

    # --------------------------------------------------------

    def get_node(self, ref):
        return get_ref(self, "nodes", ref)

    def require_node(self, ref):
        return require_ref(self, "nodes", ref)

    # --------------------------------------------------------

    def validate(self):
        super().validate()

        if not self.nodes:
            return

        inputs = []
        outputs = []

        for name, node in self.nodes.items():
            if isinstance(node, InputNode):
                inputs.append(name)
            if isinstance(node, OutputNode):
                outputs.append(name)

        if len(inputs) == 0:
            raise ValidationError("graph must contain at least one InputNode")

        if len(outputs) == 0:
            raise ValidationError("graph must contain at least one OutputNode")


# ============================================================
# Block Node
# ============================================================

class BlockNode(GraphNode):

    type = "block"
    repeat = 1

    def __init__(self, *, repeat=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr("repeat", repeat)

    def validate(self):
        super().validate()

        if not is_integer(self.repeat, min_val=1):
            raise ValidationError(
                f"Invalid value for repeat: {self.repeat}"
            )

    def is_single(self):
        return self.repeat == 1


# ============================================================
# IO Nodes
# ============================================================

class IONode(IRNode):

    __abstract__ = True

    def validate(self):
        super().validate()

        if self.weights:
            raise ValidationError("IO node cannot have weights")

        if self.has_subgraph():
            raise ValidationError("IO node cannot have subgraphs")


class InputNode(IONode):

    type = "input"

    def validate(self):
        super().validate()

        if self.inputs:
            raise ValidationError("Input node cannot have inputs")

        if not self.outputs:
            raise ValidationError("Input node must have at least one output")


class OutputNode(IONode):

    type = "output"

    def validate(self):
        super().validate()

        if not self.inputs:
            raise ValidationError("Output node must have at least one input")

        if self.outputs:
            raise ValidationError("Output node cannot have outputs")


# ============================================================
# Factory Shortcut
# ============================================================

make_node = IRNode.create