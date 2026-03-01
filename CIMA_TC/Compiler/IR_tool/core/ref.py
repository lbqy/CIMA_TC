from __future__ import annotations

import re
from dataclasses import dataclass
from typing import (
    Optional,
    Iterable,
    Dict,
    Any,
    List,
    Protocol,
    runtime_checkable,
)


# ============================================================
# Exceptions
# ============================================================


class RefError(Exception):
    """Base class for all reference-related errors."""


class InvalidNameError(RefError):
    """Raised when a name segment is invalid."""


class InvalidRefError(RefError):
    """Raised when a reference string is invalid."""


class RefResolutionError(RefError):
    """Raised when a reference cannot be resolved."""


# ============================================================
# Name validation
# ============================================================

RE_NAME: re.Pattern[str] = re.compile(
    r'^[a-zA-Z][a-zA-Z0-9]*(?:[_\-][a-zA-Z0-9]+)*(?:\:\d+)?$'
)


@dataclass(frozen=True)
class NameSegment:
    """
    Represents a single segment in a reference path.

    Example:
        layer
        layer:2
    """

    name: str
    index: Optional[int] = None

    @classmethod
    def parse(cls, raw: str) -> NameSegment:
        """
        Parse a raw string segment into a NameSegment object.
        """
        if not isinstance(raw, str) or RE_NAME.fullmatch(raw) is None:
            raise InvalidNameError(f"Invalid name segment: {raw!r}")

        parts: List[str] = raw.split(":")
        if len(parts) == 1:
            return cls(parts[0], None)

        if len(parts) == 2:
            return cls(parts[0], int(parts[1]))

        raise InvalidNameError(f"Invalid name format: {raw!r}")

    def __str__(self) -> str:
        if self.index is None:
            return self.name
        return f"{self.name}:{self.index}"


# ============================================================
# Ref class
# ============================================================


@dataclass(frozen=True)
class Ref:
    """
    Immutable structured reference.

    Example:
        Ref.parse("encoder.layer:2.attn")
    """

    segments: tuple[NameSegment, ...]

    # ------------------------
    # Construction
    # ------------------------

    @classmethod
    def parse(cls, raw: str) -> Ref:
        """
        Parse a dot-separated reference string.
        """
        if not isinstance(raw, str):
            raise InvalidRefError("Reference must be a string")

        parts = raw.split(".")
        try:
            segments = tuple(NameSegment.parse(p) for p in parts)
        except InvalidNameError as e:
            raise InvalidRefError(str(e)) from e

        return cls(segments)

    @classmethod
    def from_segments(
        cls,
        segments: Iterable[NameSegment],
    ) -> Ref:
        return cls(tuple(segments))

    # ------------------------
    # Representation
    # ------------------------

    def __str__(self) -> str:
        return ".".join(str(seg) for seg in self.segments)

    def __repr__(self) -> str:
        return f"Ref({str(self)!r})"

    def __iter__(self):
        return iter(self.segments)

    def __len__(self) -> int:
        return len(self.segments)


# ============================================================
# Tree protocol
# ============================================================


@runtime_checkable
class RefContainer(Protocol):
    """
    Protocol describing minimal container interface.

    Requirements:
        - obj.<key> returns Dict[str, Any]
        - child object may define `.number` for index bounds
    """

    number: int  # optional, default assumed 0 if missing


# ============================================================
# Core resolution logic
# ============================================================


def resolve_ref(
    root: Any,
    key: str,
    ref: Ref | str,
    *,
    strict: bool = False,
) -> Optional[Any]:
    """
    Core reference resolution function.

    Parameters
    ----------
    root : Any
        Root object to start traversal.
    key : str
        Attribute name storing children dictionary.
    ref : Ref | str
        Structured Ref or raw reference string.
    strict : bool
        If True, raises RefResolutionError on failure.
        If False, returns None on failure.

    Returns
    -------
    Optional[Any]
        Resolved object, or None if not found (strict=False).

    Raises
    ------
    RefResolutionError
        If strict=True and resolution fails.
    """

    if isinstance(ref, str):
        ref = Ref.parse(ref)

    current: Any = root

    for segment in ref:

        container: Optional[Dict[str, Any]] = getattr(current, key, None)

        if container is None:
            if strict:
                raise RefResolutionError(
                    f"Object {current!r} has no attribute '{key}'"
                )
            return None

        current = container.get(segment.name)

        if current is None:
            if strict:
                raise RefResolutionError(
                    f"Segment '{segment.name}' not found"
                )
            return None

        if segment.index is not None:
            max_number: int = getattr(current, "number", 0)
            if not (0 <= segment.index < max_number):
                if strict:
                    raise RefResolutionError(
                        f"Index {segment.index} out of range "
                        f"(0 <= idx < {max_number})"
                    )
                return None

    return current


# ============================================================
# Public APIs
# ============================================================


def get_ref(
    root: Any,
    key: str,
    ref: Ref | str,
) -> Optional[Any]:
    """
    Resolve reference.

    Returns None if not found.
    """
    return resolve_ref(root, key, ref, strict=False)


def require_ref(
    root: Any,
    key: str,
    ref: Ref | str,
) -> Any:
    """
    Resolve reference.

    Raises RefResolutionError if not found.
    """
    result = resolve_ref(root, key, ref, strict=True)
    assert result is not None  # for type checker
    return result