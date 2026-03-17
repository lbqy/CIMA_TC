"""
Mapping passes for CIMA compiler.

This package currently includes:
- XB-based split analysis and rewrite for conv/linear (xb_split, split_pass).
- Example tests for CustomizedNet mapping.
"""

from .xb_split import XBConfig, compute_conv_fc_split_plan, analyze_ir_for_xb_splits
from .split_pass import SplitResult, split_model_for_xb, export_split_model

__all__ = [
    "XBConfig",
    "compute_conv_fc_split_plan",
    "analyze_ir_for_xb_splits",
    "SplitResult",
    "split_model_for_xb",
    "export_split_model",
]

