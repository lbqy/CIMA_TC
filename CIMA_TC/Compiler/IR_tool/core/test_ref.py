"""
Comprehensive test suite for the Ref resolution system.

This test file verifies:

1. NameSegment parsing
2. Ref parsing and formatting
3. Error handling for invalid names and refs
4. resolve_ref behavior (strict and non-strict modes)
5. get_ref and require_ref wrappers
6. Index bounds validation
7. Behavior when attributes are missing
8. Behavior when children dictionary is missing

Designed for production-grade reliability.
"""

import pytest

from .ref import (
    NameSegment,
    Ref,
    resolve_ref,
    get_ref,
    require_ref,
    InvalidNameError,
    InvalidRefError,
    RefResolutionError,
)


# ============================================================
# Helper Tree Structure
# ============================================================


class Node:
    """
    Simple tree node used for testing.
    """

    def __init__(self, number: int = 0):
        self.children = {}
        self.number = number


def build_sample_tree():
    """
    Build the following tree:

    root
     └── encoder
          └── layer (number=3)
               └── attn
    """
    root = Node()
    root.children["encoder"] = Node()

    encoder = root.children["encoder"]
    encoder.children["layer"] = Node(number=3)

    layer = encoder.children["layer"]
    layer.children["attn"] = Node()

    return root


# ============================================================
# NameSegment Tests
# ============================================================


@pytest.mark.parametrize(
    "raw,expected_name,expected_index",
    [
        ("layer", "layer", None),
        ("layer:2", "layer", 2),
        ("block_1", "block_1", None),
        ("conv-2:10", "conv-2", 10),
    ],
)
def test_name_segment_parse_valid(raw, expected_name, expected_index):
    seg = NameSegment.parse(raw)
    assert seg.name == expected_name
    assert seg.index == expected_index
    assert str(seg) == raw


@pytest.mark.parametrize(
    "raw",
    [
        "1layer",        # starts with digit
        "layer::2",      # double colon
        "layer:abc",     # non-integer index
        "layer:",        # missing index
    ],
)
def test_name_segment_parse_invalid(raw):
    with pytest.raises(InvalidNameError):
        NameSegment.parse(raw)


# ============================================================
# Ref Tests
# ============================================================


def test_ref_parse_and_str():
    ref = Ref.parse("encoder.layer:2.attn")

    assert len(ref) == 3
    assert str(ref) == "encoder.layer:2.attn"

    segments = list(ref)
    assert segments[0].name == "encoder"
    assert segments[1].index == 2


def test_ref_parse_invalid():
    with pytest.raises(InvalidRefError):
        Ref.parse("encoder..layer")


# ============================================================
# resolve_ref - Non Strict Mode
# ============================================================


def test_resolve_success_non_strict():
    root = build_sample_tree()

    result = resolve_ref(
        root,
        key="children",
        ref="encoder.layer:2.attn",
        strict=False,
    )

    assert result is not None


def test_resolve_missing_segment_non_strict():
    root = build_sample_tree()

    result = resolve_ref(
        root,
        key="children",
        ref="encoder.unknown",
        strict=False,
    )

    assert result is None


def test_resolve_index_out_of_range_non_strict():
    root = build_sample_tree()

    result = resolve_ref(
        root,
        key="children",
        ref="encoder.layer:10.attn",
        strict=False,
    )

    assert result is None


def test_resolve_missing_children_attr_non_strict():
    root = build_sample_tree()

    # Remove children attribute from encoder
    del root.children["encoder"].children

    result = resolve_ref(
        root,
        key="children",
        ref="encoder.layer",
        strict=False,
    )

    assert result is None


# ============================================================
# resolve_ref - Strict Mode
# ============================================================


def test_resolve_missing_segment_strict():
    root = build_sample_tree()

    with pytest.raises(RefResolutionError):
        resolve_ref(
            root,
            key="children",
            ref="encoder.unknown",
            strict=True,
        )


def test_resolve_index_out_of_range_strict():
    root = build_sample_tree()

    with pytest.raises(RefResolutionError):
        resolve_ref(
            root,
            key="children",
            ref="encoder.layer:10",
            strict=True,
        )


def test_resolve_missing_children_attr_strict():
    root = build_sample_tree()

    del root.children["encoder"].children

    with pytest.raises(RefResolutionError):
        resolve_ref(
            root,
            key="children",
            ref="encoder.layer",
            strict=True,
        )


# ============================================================
# get_ref Wrapper
# ============================================================


def test_get_ref_returns_none():
    root = build_sample_tree()

    result = get_ref(
        root,
        key="children",
        ref="encoder.invalid",
    )

    assert result is None


# ============================================================
# require_ref Wrapper
# ============================================================


def test_require_ref_success():
    root = build_sample_tree()

    result = require_ref(
        root,
        key="children",
        ref="encoder.layer:1",
    )

    assert result is not None


def test_require_ref_failure():
    root = build_sample_tree()

    with pytest.raises(RefResolutionError):
        require_ref(
            root,
            key="children",
            ref="encoder.invalid",
        )


# ============================================================
# Ref Object Input Instead of String
# ============================================================


def test_ref_object_input():
    root = build_sample_tree()

    ref = Ref.parse("encoder.layer:1")

    result = resolve_ref(
        root,
        key="children",
        ref=ref,
        strict=False,
    )

    assert result is not None