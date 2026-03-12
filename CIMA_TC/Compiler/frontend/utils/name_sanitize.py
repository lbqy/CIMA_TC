"""
Sanitize ONNX node names to IR-compatible layer names.
IR NameSegment allows: [a-zA-Z][a-zA-Z0-9]*(?:[_\-][a-zA-Z0-9]+)*(?::\d+)?$
So no leading slash, no path separators; must start with a letter.
"""

from __future__ import annotations

import re
from typing import Dict, List

# Pattern for valid segment (same idea as ref.RE_NAME): letter start, then alphanumeric/underscore/hyphen
_RE_VALID = re.compile(r"^[a-zA-Z][a-zA-Z0-9]*(?:[_\-][a-zA-Z0-9]+)*(?::\d+)?$")


def sanitize_layer_name(raw: str) -> str:
    """
    Convert an ONNX node name to a valid IR layer name.
    Replaces / \\ and other invalid chars with _; ensures first char is a letter.
    """
    if not raw:
        return "layer_0"
    s = str(raw)
    # Replace path separators and other invalid chars with underscore
    s = re.sub(r"[/\\.,\s]+", "_", s)
    # Remove leading/trailing underscores, collapse multiple underscores
    s = re.sub(r"_+", "_", s).strip("_")
    # Ensure starts with letter (IR requirement)
    if not s:
        return "layer_0"
    if not s[0].isalpha():
        s = "n_" + s
    if not _RE_VALID.fullmatch(s):
        # Fallback: keep only alphanumeric, underscore, hyphen
        s = re.sub(r"[^a-zA-Z0-9_\-]", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        if not s or not s[0].isalpha():
            s = "n_" + (s or "0")
    return s or "layer_0"


def build_layer_name_map(node_names: List[str]) -> Dict[str, str]:
    """
    Build original -> sanitized name map with uniqueness: if two nodes sanitize
    to the same string, append _0, _1, ...
    """
    name_map: Dict[str, str] = {}
    seen: Dict[str, int] = {}
    for orig in node_names:
        base = sanitize_layer_name(orig)
        if base in seen:
            seen[base] += 1
            name_map[orig] = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
            name_map[orig] = base
    return name_map
