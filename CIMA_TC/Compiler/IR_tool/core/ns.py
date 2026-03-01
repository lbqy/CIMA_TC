from contextlib import contextmanager
import threading
from typing import Iterator, Tuple


_local = threading.local()


@contextmanager
def ns_push(name: str) -> Iterator[Tuple[str, ...]]:
    """
    Push a name onto the thread-local namespace stack.

    The namespace is maintained as a stack local to each thread.
    The yielded value is an immutable snapshot of the current namespace.
    """
    if not isinstance(name, str):
        raise TypeError(f"namespace element must be str, got {type(name).__name__}")

    ns = getattr(_local, "namespace", None)
    if ns is None:
        ns = []
        _local.namespace = ns

    ns.append(name)
    try:
        # Yield an immutable snapshot to avoid accidental mutation
        yield tuple(ns)
    finally:
        ns.pop()
        # Clean up empty namespace to avoid stale state
        if not ns:
            delattr(_local, "namespace")


def ns_get(sep: str = "/") -> str:
    """
    Get the current thread-local namespace as a joined string.
    """
    ns = getattr(_local, "namespace", None)
    if not ns:
        return ""
    return sep.join(ns)
