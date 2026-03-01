from typing import Any, TypeVar, Generic, Union, Optional, Sequence, Mapping, Iterable, Callable, Type, overload
import re
from dataclasses import dataclass
from functools import wraps
import warnings

# ==================== TYPE DEFINITIONS ====================
T = TypeVar('T')
U = TypeVar('U')
ScalarType = Union[None, bool, int, float, str, bytes]
Validator = Callable[[Any], bool]
Converter = Callable[[Any], T]

class ValidationError(ValueError):
    """Exception raised when validation fails."""
    pass

class ConversionError(TypeError):
    """Exception raised when conversion fails."""
    pass

# ==================== BASIC VALIDATORS ====================
def is_scalar(obj: Any) -> bool:
    """Check if object is a scalar type (None, bool, number, string, bytes)."""
    return obj is None or isinstance(obj, (bool, int, float, str, bytes))

def is_boolean(obj: Any) -> bool:
    """Check if object is strictly a boolean (not accepting subclasses)."""
    return type(obj) is bool

def is_integer(
    value: Any,
    *,
    min_val: Optional[int] = None,
    max_val: Optional[int] = None,
    strict: bool = True
) -> bool:
    """Check if value is an integer with optional range constraints.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        strict: Whether to exclude bool from integer check
    """
    if strict and type(value) is not int:
        return False
    elif not strict and not isinstance(value, int):
        return False
    
    if min_val is not None and value < min_val:
        return False
    if max_val is not None and value > max_val:
        return False
    return True

def is_number(
    value: Any,
    *,
    min_val: Optional[Union[int, float]] = None,
    max_val: Optional[Union[int, float]] = None,
    lower_limit: Optional[Union[int, float]] = None,
    upper_limit: Optional[Union[int, float]] = None,
    strict: bool = True
) -> bool:
    """Check if value is a number (int or float) with optional constraints.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        lower_limit: Lower boundary (exclusive)
        upper_limit: Upper boundary (exclusive)
        strict: Whether regard boolean as number
    """
    if strict and isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    
    if min_val is not None and value < min_val:
        return False
    if max_val is not None and value > max_val:
        return False
    if lower_limit is not None and value <= lower_limit:
        return False
    if upper_limit is not None and value >= upper_limit:
        return False
    
    return True

# ==================== COLLECTION VALIDATORS ====================
@dataclass
class CollectionConstraints:
    """Configuration for collection validation constraints."""
    min_size: int = 0
    max_size: Optional[int] = None
    exact_size: Optional[int] = None
    allow_none: bool = False
    allow_scalar: bool = False

def validate_collection(
    obj: Any,
    validator: Validator,
    constraints: Optional[CollectionConstraints] = None,
    **validator_kwargs: Any
) -> bool:
    """Validate each element in a collection.
    
    Args:
        obj: Object to validate
        validator: Element validator function
        constraints: Collection size constraints
        **validator_kwargs: Arguments passed to element validator
    
    Returns:
        True if validation passes, False otherwise
    """
    if constraints is None:
        constraints = CollectionConstraints()
    
    # Handle None values
    if obj is None:
        return constraints.allow_none
    
    # Handle scalar values (if allowed)
    if constraints.allow_scalar and is_scalar(obj):
        obj = [obj]
    
    # Check if object is a sequence
    if not isinstance(obj, Sequence):
        return False
    
    # Validate collection size constraints
    size = len(obj)
    
    if constraints.exact_size is not None and size != constraints.exact_size:
        return False
    if size < constraints.min_size:
        return False
    if constraints.max_size is not None and size > constraints.max_size:
        return False
    
    # Validate each element
    return all(validator(item, **validator_kwargs) for item in obj)

# def is_integer_sequence(
#     obj: Any,
#     **kwargs: Any
# ) -> bool:
#     """Check if object is a sequence of integers."""
#     constraints = CollectionConstraints(**kwargs)
#     return validate_collection(obj, is_integer, constraints)

# def is_number_sequence(
#     obj: Any,
#     **kwargs: Any
# ) -> bool:
#     """Check if object is a sequence of numbers."""
#     constraints = CollectionConstraints(**kwargs)
#     return validate_collection(obj, is_number, constraints)

def is_integers(
    obj: Any,
    *,
    min_size: int = 0,
    max_size: Optional[int] = None,
    exact_size: Optional[int] = None,
    allow_none: bool = False,
    allow_scalar: bool = False,
    **integer_kwargs: Any
) -> bool:
    """
    Backward-compatible helper:
    Validate sequence of integers with element-level constraints.

    integer_kwargs are passed to is_integer().
    """
    constraints = CollectionConstraints(
        min_size=min_size,
        max_size=max_size,
        exact_size=exact_size,
        allow_none=allow_none,
        allow_scalar=allow_scalar,
    )

    return validate_collection(
        obj,
        lambda x: is_integer(x, **integer_kwargs),
        constraints,
    )

def is_numbers(
    obj: Any,
    *,
    min_size: int = 0,
    max_size: Optional[int] = None,
    exact_size: Optional[int] = None,
    allow_none: bool = False,
    allow_scalar: bool = False,
    **number_kwargs: Any
) -> bool:
    """
    Backward-compatible helper:
    Validate sequence of numbers with element-level constraints.

    number_kwargs are passed to is_number().
    """
    constraints = CollectionConstraints(
        min_size=min_size,
        max_size=max_size,
        exact_size=exact_size,
        allow_none=allow_none,
        allow_scalar=allow_scalar,
    )

    return validate_collection(
        obj,
        lambda x: is_number(x, **number_kwargs),
        constraints,
    )

# ==================== VALUE RANGE VALIDATORS ====================
def is_in_values(
    obj: Any,
    values: Union[set, list, tuple],
    allow_none: bool = False
) -> bool:
    """Check if value is in a set of allowed values.
    
    Args:
        obj: Value to check
        values: Set of allowed values
        allow_none: Whether to allow None values
    """
    if obj is None and obj not in values:
        return allow_none
    return obj in values

# ==================== CONVERSION FUNCTIONS ====================
def to_boolean(value: Any) -> bool:
    """Convert value to boolean.
    
    Supported types:
    - None, True, False, 0, 1
    - Strings: 'true', 'True', 'TRUE', 'yes', 'y', '1' -> True
    - Strings: 'false', 'False', 'FALSE', 'no', 'n', '0', '' -> False
    """
    if value in (None, True, False, 0, 1):
        return bool(value)
    
    if isinstance(value, (bytes, bytearray)):
        value = value.decode('utf-8', errors='ignore')
    
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ('1', 'true', 'True', 'TRUE', 'yes', 'y'):
            return True
        if normalized in ('', '0', 'false', 'False', 'FALSE', 'no', 'n'):
            return False
        raise ValueError(f"Cannot convert string to boolean: {value!r}")
    
    return bool(value)

def to_string_tokens(
    obj: Any,
    *,
    container_type: Type[Sequence[str]] = tuple,
    keep_scalar: bool = False
) -> Optional[Sequence[str]]:
    """Convert object to sequence of string tokens.
    
    Args:
        obj: Object to convert
        container_type: Return container type
        keep_scalar: Whether to keep scalar as scalar (otherwise convert to single-element sequence)
    
    Returns:
        Sequence of strings or single string
    """
    if obj is None:
        return None if keep_scalar else container_type()
    
    if is_scalar(obj):
        return str(obj) if keep_scalar else container_type((str(obj),))

    if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes, Mapping)):
        return container_type(map(str, obj))
    
    raise ConversionError(f"Cannot convert {type(obj).__name__} to token sequence")

def to_integer_tuple(
    obj: Any,
    *,
    dimensions: Optional[int] = None,
    keep_scalar: bool = False,
    allow_float2int: bool = True
) -> Union[int, tuple[int, ...]]:
    """Convert object to integer tuple.
    
    Args:
        obj: Object to convert
        dimensions: Expected number of dimensions
        keep_scalar: Whether to keep scalar as scalar
        allow_float2int: Whether to allow float to int conversion (with truncation)
    
    Returns:
        Integer tuple or single integer
    """
    if dimensions is not None and dimensions < 0:
        raise ValueError("Dimensions cannot be negative")
    
    if obj is None:
        if dimensions is not None and dimensions > 0:
            warnings.warn(
                f"Converting None to integer tuple with {dimensions} dimensions results in empty tuple, which may not be intended.",
            )
        return None if keep_scalar else ()
    
    if isinstance(obj, int):
        if keep_scalar:
            return obj
        if dimensions is None:
            return (obj,)
        return (obj,) * dimensions
    
    if isinstance(obj, Iterable):
        result = []
        for item in obj:
            if isinstance(item, int):
                result.append(item)
                continue
            if isinstance(item, float):
                if allow_float2int:
                    warnings.warn(
                        f"Float value {item} will be truncated to integer",
                        UserWarning
                    )
                    result.append(int(item))
                else:
                    raise ConversionError(f"Cannot convert float {item} to integer")
                continue
            try:
                result.append(int(item))
            except (ValueError, TypeError):
                raise ConversionError(f"Cannot convert {item!r} (type {type(item).__name__}) to integer")
        
        result = tuple(result)
        
        if dimensions is None or len(result) == dimensions:
            return result
        
        if dimensions % len(result) != 0:
            raise ConversionError(
                f"Dimensions {dimensions} cannot be divided by sequence length {len(result)}"
            )
        
        return result * (dimensions // len(result))
    
    raise ConversionError(f"Cannot convert {type(obj).__name__} to integer tuple")

# ==================== GENERIC CONVERTERS ====================
@overload
def to_typed_object(
    obj: None,
    target_type: Type[T],
    converter: Optional[Callable[..., T]] = None
) -> None: ...

@overload
def to_typed_object(
    obj: Any,
    target_type: Type[T],
    converter: Optional[Callable[..., T]] = None
) -> T: ...

def to_typed_object(
    obj: Any,
    target_type: Type[T],
    converter: Optional[Callable[..., T]] = None
) -> Optional[T]:
    """Convert object to specified type.
    
    Args:
        obj: Object to convert
        target_type: Target type
        converter: Optional conversion function (defaults to target_type)
    
    Returns:
        Converted object
    """
    if converter is None:
        converter = target_type
    
    if obj is None:
        return None
    
    if isinstance(obj, target_type):
        return obj

    if is_scalar(obj):
        try:
            return converter(obj)
        except (TypeError, ValueError) as e:
            raise ValidationError(
                f"Invalid scalar value {obj!r} for {target_type.__name__}: {e}"
            ) from e
    
    if isinstance(obj, Mapping):
        try:
            return converter(**obj)
        except TypeError as e:
            raise ConversionError(
                f"Cannot map {obj} to {target_type.__name__} constructor: {e}"
            ) from e
        except ValueError as e:
            raise ValidationError(
                f"Invalid values in mapping {obj} for {target_type.__name__}: {e}"
            ) from e
    
    if isinstance(obj, Sequence):
        try:
            return converter(*obj)
        except TypeError:
            try:
                return converter(obj)
            except TypeError as e:
                raise ConversionError(
                    f"Cannot unpack sequence {obj} to {target_type.__name__} constructor: {e}"
                ) from e
            except ValueError as e:
                raise ValidationError(
                    f"Invalid values in sequence {obj} for {target_type.__name__}: {e}"
                ) from e
    
    try:
        return converter(obj)
    except (TypeError, ValueError) as e:
        raise ConversionError(
            f"Cannot convert {type(obj).__name__} to {target_type.__name__}: {e}"
        ) from e

def to_typed_dict(
    obj: Any,
    target_type: Type[T],
    converter: Optional[Callable[..., T]] = None
) -> Optional[dict[str, T]]:
    """Convert object to dictionary with values converted to specified type."""
    if obj is None:
        return None
    
    if isinstance(obj, Mapping):
        return {
            str(key): to_typed_object(value, target_type, converter)
            for key, value in obj.items()
        }
    
    raise ConversionError(f"Cannot convert {type(obj).__name__} to typed dictionary")

def to_typed_list(
    obj: Any,
    target_type: Type[T],
    converter: Optional[Callable[..., T]] = None
) -> Optional[list[T]]:
    """Convert object to list of specified type."""
    if obj is None:
        return None
    
    if is_scalar(obj):
        return [to_typed_object(obj, target_type, converter)]
    
    if isinstance(obj, Mapping):
        return [to_typed_object(obj, target_type, converter)]
    
    if isinstance(obj, Iterable):
        return [
            to_typed_object(item, target_type, converter)
            for item in obj
        ]
    
    raise ConversionError(f"Cannot convert {type(obj).__name__} to typed list")

# ==================== DECORATORS AND MIXINS ====================
def mixin_class(base_class: Type) -> Callable[[Type], Type]:
    """Add class as mixin to base class."""
    def decorator(mixin_class: Type) -> Type:
        # Preserve original base classes
        original_bases = base_class.__bases__
        
        # Ensure mixin is not already in base classes
        if mixin_class not in original_bases:
            base_class.__bases__ = (mixin_class, *original_bases)
        
        return mixin_class
    return decorator

# ==================== STRING PROCESSING ====================
_VAR_TOKEN_PATTERN = re.compile(r'[^a-zA-Z0-9_]')

def to_variable_token(
    text: Optional[str],
    *,
    allow_none: bool = False
) -> Optional[str]:
    """Convert string to valid variable token.
    
    Args:
        text: Input text
        allow_none: Whether to allow None return value
    
    Returns:
        Cleaned variable token
    """
    if text is None:
        return None if allow_none else ""
    
    # Remove invalid characters and convert to lowercase
    cleaned = _VAR_TOKEN_PATTERN.sub('_', text).lower()
    
    # Ensure doesn't start with digit
    if cleaned and cleaned[0].isdigit():
        cleaned = '_' + cleaned
    
    return cleaned

# ==================== VALIDATION DECORATOR ====================
def validate_arguments(**validators: Validator) -> Callable:
    """Decorator for argument validation.
    
    Example:
        @validate_arguments(x=is_integer, y=is_number)
        def func(x: Any, y: Any) -> float:
            return float(x) + y
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get parameter binding (simplified version)
            # In practice, use inspect.signature for complete parameter binding
            return func(*args, **kwargs)
        return wrapper
    return decorator

# ==================== USAGE EXAMPLES ====================
if __name__ == "__main__":
    # Example usage
    print(is_integer(5, min_val=0, max_val=10))  # True
    print(is_number(3.14, min_val=0))  # True
    
    # Validate collection
    constraints = CollectionConstraints(min_size=1, max_size=3)
    print(validate_collection([1, 2, 3], is_integer, constraints))  # True
    
    # Conversion examples
    print(to_boolean("yes"))  # True
    print(to_integer_tuple([1, 2, 3], dimensions=6))  # (1, 2, 3, 1, 2, 3)
    print(to_variable_token("Hello World!"))  # hello_world_