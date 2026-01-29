"""
WSI Models for Multiple Instance Learning
==========================================

This package contains MIL models for WSI-level prediction.

Models:
- ABMIL: Attention-based Multiple Instance Learning
- TransMIL: Transformer-based Multiple Instance Learning
"""

from .mil_model import ABMIL, TransMIL, GatedAttention, MILModelConfig

__all__ = [
    "ABMIL",
    "TransMIL",
    "GatedAttention",
    "MILModelConfig",
]
