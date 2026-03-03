"""
Operator registration module based on the generic registry framework.

This module defines the base class for all operators, which automatically
registers its subclasses using the registry system provided by
RegistryMixin and RegistryMeta. It also provides factory functions and
utilities for enumerating registered operator IDs.
"""

from .reg import RegistryMixin, RegistryEntry   # Assume these are imported from the current registry module
from .jsonable import Jsonable
from .type_utils import to_string_tokens, ValidationError

class BaseOp(Jsonable, RegistryMixin, RegistryEntry):
    """
    Abstract base class for all operators, also serving as the root registry.

    Subclasses must define a class attribute with the name specified by
    `__registry_key__` (i.e., `op_id`) containing a non‑empty string.
    This attribute is used as the registration key and will be automatically
    collected by the metaclass.

    Attributes (metadata) that can be overridden by subclasses:
        attrs (tuple): Names of attributes to be serialized.
        weights (tuple): Names of weight parameters.
        optional_weights (tuple): Names of optional weight parameters.
        unsigned_weights (tuple): Names of weights that are not signed.
        num_inputs (int or None): Expected number of input tensors.

    Class configuration (inherited from RegistryMixin):
        __registry_key__ = "op_id"          # Name of the class attribute used as registry key.
        __registry_default__ = None         # Optional default key for lookups.
    """

    __registry_key__ = "op_id"               # The class attribute that holds the operator ID.
    __registry_default__ = None               # No default key; must be provided explicitly.
    __abstract__ = True

    # Metadata defaults – can be overridden in concrete subclasses.
    attrs = ()
    weights = ()
    optional_weights = ()
    unsigned_weights = ()
    num_inputs = None

    def __init__(self, *, op_id=None, **kwargs):
        """
        Initialize an operator instance.

        Args:
            op_id (str, optional): Identifier of the operator. If not provided,
                it is taken from the class attribute `op_id` of the subclass.
            **kwargs: Additional arguments passed to the base class constructors
                (Jsonable and RegistryEntry) and potentially used by subclasses.

        Raises:
            AssertionError: If `op_id` cannot be determined (neither passed nor
                defined as a class attribute).
        """
        super().__init__(**kwargs)
        if op_id is None:
            # Retrieve the operator ID from the subclass's class attribute.
            if not hasattr(self.__class__, "op_id"):
                raise ValueError("op_id must be provided or defined in class")
            op_id = self.__class__.op_id
        self.op_id = op_id

    def validate(self) -> None:
        """
        Validate the operator instance after creation.

        This method is automatically called by the `create` factory method
        of RegistryMixin. It ensures that the operator has a valid non‑empty
        `op_id`.

        Raises:
            AssertionError: If `op_id` is empty or None.
        """
        if not self.op_id:
            raise ValidationError(f"invalid op_id={self.op_id}")

    def get_attrs(self) -> dict:
        """
        Collect the values of attributes listed in `self.attrs`.

        Returns:
            dict: Mapping from attribute name to its current value.
        """
        return {k: getattr(self, k) for k in self.attrs}

    def weight_shapes(self, **kwargs) -> dict:
        """
        Return the shapes of weight parameters.

        This method should be overridden by subclasses that define weights.
        The base implementation returns an empty dictionary if `self.weights`
        is empty; otherwise it raises NotImplementedError to indicate that
        the subclass must provide the shapes.

        Args:
            **kwargs: Additional arguments that may be used to determine shapes
                (e.g., input shapes).

        Returns:
            dict: Mapping from weight name to its shape (or a tensor/array spec).

        Raises:
            NotImplementedError: If the subclass has weights but does not
                override this method.
        """
        if not self.weights:
            return {}
        raise NotImplementedError


def make_op(obj, **kwargs):
    """
    Factory function to create an operator instance from various inputs.

    This is a convenience wrapper around `BaseOp.create`. It supports:
        - A string: interpreted as the operator ID.
        - A dictionary: must contain the key `op_id`.
        - An existing `BaseOp` instance: returned as is (no new instance created).
        - None with additional `kwargs`: treated as a dictionary of keyword arguments
          (which must contain `op_id`).

    Args:
        obj (str, dict, BaseOp, None): The source from which to create the operator.
        **kwargs: Extra keyword arguments that are merged with the source (if a dict)
            or passed to the constructor.

    Returns:
        BaseOp: An instance of the appropriate operator subclass.

    Raises:
        KeyError: If the operator ID is not registered.
        TypeError: If the source type is unsupported.
        ValueError: If validation fails or required keys are missing.
    """
    return BaseOp.create(obj, **kwargs)


def enum_op_ids():
    """
    Iterate over all registered operator IDs.

    Yields:
        str: Each registered operator key (lowercase version of the `op_id`
            values defined in subclasses).
    """
    for key, _ in BaseOp.all_registered():
        yield key

class UnaryOp(BaseOp):
    """
    Base class for unary operators (operators that take exactly one input).

    Sets `num_inputs = 1` by default. Subclasses should still define `op_id`.
    """
    num_inputs = 1
    __abstract__ = True

class BinaryOp(BaseOp):
    """
    Base class for binary operators (operators that take exactly two inputs).

    Sets `num_inputs = 2` by default. Subclasses should still define `op_id`.
    """
    num_inputs = 2
    __abstract__ = True  

# if __name__ == "__main__":
#     # ------------------------------------------------------------
#     # Example Usage
#     # ------------------------------------------------------------
#     # Define two concrete operator classes
#     class Add(BaseOp):
#         op_id = "add"

#         def execute(self, a: float, b: float) -> float:
#             return a + b

#     class Sub(BaseOp):
#         op_id = "sub"

#         def execute(self, a: float, b: float) -> float:
#             return a - b

#     print("Registered operator IDs:", list(enum_op_ids()))
#     # Output: Registered operator IDs: ['add', 'sub']

#     # Create instances using the factory
#     add_op = make_op("add")
#     sub_op = make_op({"op_id": "sub"})

#     print(f"Add(3, 4) = {add_op.execute(3, 4)}")   # Add(3, 4) = 7
#     print(f"Sub(3, 4) = {sub_op.execute(3, 4)}")   # Sub(3, 4) = -1

#     # Lookup the class by key
#     cls = BaseOp.get("add")
#     print(f"Class for 'add': {cls.__name__}")      # Class for 'add': Add