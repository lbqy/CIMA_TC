import threading
import time
import pytest

from .ns import ns_push, ns_get


# ----------------------------------------------------------------------
# Basic semantics
# ----------------------------------------------------------------------

def test_ns_get_empty():
    """ns_get should return empty string when no namespace is active."""
    assert ns_get() == ""


def test_single_push():
    """Single push/pop should correctly set and clear namespace."""
    with ns_push("A"):
        assert ns_get() == "A"
    assert ns_get() == ""


def test_nested_push():
    """Nested pushes should form a stack-like namespace."""
    with ns_push("A"):
        assert ns_get() == "A"
        with ns_push("B"):
            assert ns_get() == "A/B"
        assert ns_get() == "A"
    assert ns_get() == ""


# ----------------------------------------------------------------------
# Exception safety
# ----------------------------------------------------------------------

def test_exception_inside_context():
    """
    Exception raised inside ns_push context must not leak namespace state.
    """
    with pytest.raises(RuntimeError):
        with ns_push("A"):
            assert ns_get() == "A"
            raise RuntimeError("boom")

    assert ns_get() == ""


def test_nested_exception_unwind():
    """
    Nested contexts must unwind correctly when an inner exception occurs.
    """
    with pytest.raises(ValueError):
        with ns_push("A"):
            with ns_push("B"):
                assert ns_get() == "A/B"
                raise ValueError("fail")

    assert ns_get() == ""


# ----------------------------------------------------------------------
# Thread isolation
# ----------------------------------------------------------------------

def test_thread_isolation():
    """
    Namespaces must be isolated across threads.
    """
    results = {}

    def worker(name: str):
        with ns_push(name):
            time.sleep(0.05)
            results[name] = ns_get()

    t1 = threading.Thread(target=worker, args=("T1",))
    t2 = threading.Thread(target=worker, args=("T2",))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert results["T1"] == "T1"
    assert results["T2"] == "T2"

    # Main thread must remain clean
    assert ns_get() == ""


def test_thread_nested_independence():
    """
    Nested namespaces in different threads must not interfere.
    """
    results = []

    def worker(prefix: str):
        with ns_push(prefix):
            with ns_push("inner"):
                results.append(ns_get())

    threads = [
        threading.Thread(target=worker, args=("A",)),
        threading.Thread(target=worker, args=("B",)),
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(results) == ["A/inner", "B/inner"]


# ----------------------------------------------------------------------
# Defensive behavior
# ----------------------------------------------------------------------

def test_invalid_namespace_type():
    """
    Non-string namespace elements should fail immediately.
    """
    with pytest.raises(TypeError):
        with ns_push(123):  # type: ignore
            pass


# ----------------------------------------------------------------------
# Lifecycle cleanup
# ----------------------------------------------------------------------

def test_namespace_attribute_cleanup():
    """
    Namespace state must be fully cleaned up after exiting context.
    """
    with ns_push("A"):
        assert ns_get() == "A"

    # After context exit, namespace should be completely cleared
    assert ns_get() == ""


# ----------------------------------------------------------------------
# Optional stress / robustness test
# ----------------------------------------------------------------------

def test_many_nested_pushes():
    """
    Deep nesting should behave correctly and fully unwind.
    """
    depth = 50
    contexts = [ns_push(str(i)) for i in range(depth)]

    for ctx in contexts:
        ctx.__enter__()

    assert ns_get() == "/".join(str(i) for i in range(depth))

    for ctx in reversed(contexts):
        ctx.__exit__(None, None, None)

    assert ns_get() == ""
